"""Torch-free SOMA-X hand stage: HaMeR MANO params -> SOMA-X hand mesh, in numpy.

Re-implements, from a single exported .npz (see mano2soma-x
`scripts/export_numpy_assets.py`):

  1. the mano2soma-x ResMLP regressor (MANO finger rotations + betas ->
     SOMA identity, bone scales, 25 absolute-local rotations, root translation);
  2. SOMA-X `SOMAHandLayer.prepare_identity()` -- identity PCA rest shape,
     closed-form skeleton transfer (joint positions by a linear regressor,
     joint orientations by Procrustes alignment with SOMA's Newton-Schulz
     polar decomposition + Kabsch fallback), re-posing to the bind pose;
  3. `SOMAHandLayer.pose()` -- bone-scaled forward kinematics + linear blend
     skinning.

The regressor MLP can run either in numpy or as a CoreML model on the Neural
Engine (`regressor_mlpackage=`, exported by mano2soma-x
`scripts/export_coreml_regressor.py`: 1x1-conv layout, fp16). Everything else
is a handful of numpy matmuls per batch of hands; no torch / warp / py-soma-x
at runtime. Verified against the reference torch/Warp implementation
(fasthamer/tests/test_soma_np.py).
"""
import ctypes
import hashlib
import os
import subprocess
from typing import Dict, Optional, Tuple

import numpy as np

from .assets import cache_dir

_C_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soma_stage.c")
_c_lib = None


def _load_c_lib() -> ctypes.CDLL:
    """Compile (once, cached in ~/.cache/fasthamer) and load the C SOMA stage."""
    global _c_lib
    if _c_lib is not None:
        return _c_lib
    with open(_C_SRC, "rb") as f:
        tag = hashlib.sha256(f.read()).hexdigest()[:16]
    so_path = os.path.join(cache_dir(), f"soma_stage_{tag}.so")
    if not os.path.exists(so_path):
        os.makedirs(os.path.dirname(so_path), exist_ok=True)
        cmd = ["cc", "-O3", "-shared", "-fPIC", _C_SRC, "-o", so_path]
        subprocess.run(cmd, check=True, capture_output=True)
    lib = ctypes.CDLL(so_path)
    D, I = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    lib.soma_create.restype = ctypes.c_void_p
    lib.soma_create.argtypes = [ctypes.c_int] * 4 + [D, D, ctypes.c_double, D, D, I, I, I, I, D, D, D,
                                I, I, I, I, D, D, D, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
    lib.soma_destroy.argtypes = [ctypes.c_void_p]
    lib.soma_destroy.restype = None
    lib.soma_pose.argtypes = [ctypes.c_void_p, ctypes.c_int, D, D, D, D, D, D]
    lib.soma_pose.restype = None
    _c_lib = lib
    return lib

NUM_JOINTS = 25
NUM_IDENTITY = 20
NUM_SCALES = 24
ROT6D_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


# ----------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------
def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 (max abs error 1.5e-7), vectorised, no scipy."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-a * a)
    return s * y


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Zhou et al. 6D -> (..., 3, 3); the 6 numbers are the first two ROWS."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    a2p = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2p / np.maximum(np.linalg.norm(a2p, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def _det3(M: np.ndarray) -> np.ndarray:
    return (M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1])
            - M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0])
            + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0]))


def _se3_inverse(T: np.ndarray) -> np.ndarray:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    out = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.zeros(R.shape[:-2] + (4, 4), dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


# ----------------------------------------------------------------------------
# SOMA-X rotation fitting (soma.geometry.transforms, numpy port)
# ----------------------------------------------------------------------------
class _Align:
    def __init__(self, prior_strength: float, rank_threshold: float, ns_iters: int, eps: float = 1e-8):
        self.prior_strength = float(prior_strength)
        self.rank_threshold = float(rank_threshold)
        self.ns_iters = int(ns_iters)
        self.eps = eps

    @staticmethod
    def rodrigues(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Shortest-arc rotation R with R @ b ~= a; a, b (..., 3)."""
        a_u = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), eps)
        b_u = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), eps)
        dot = np.clip((a_u * b_u).sum(-1, keepdims=True), -1.0, 1.0)
        v = np.cross(b_u, a_u)
        z = np.zeros_like(v[..., 0])
        K = np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1),
                      np.stack([v[..., 2], z, -v[..., 0]], -1),
                      np.stack([-v[..., 1], v[..., 0], z], -1)], -2)
        R = np.eye(3) + K + (1.0 / (1.0 + dot[..., None])) * (K @ K)
        anti = dot[..., 0] < -1.0 + 1e-6
        if np.any(anti):
            bb = b_u[anti]
            w = np.where((np.abs(bb[..., :1]) > 0.6), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
            axis = np.cross(bb, w)
            axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
            R[anti] = 2.0 * axis[..., :, None] * axis[..., None, :] - np.eye(3)
        return R

    def virtual_normal(self, A2: np.ndarray, B2: np.ndarray) -> np.ndarray:
        """SOMA's synthetic normal correspondence built from the first two
        points of each set; A2, B2 (..., 2, 3) -> (..., 3, 3) contribution."""
        p0, p1, q0, q1 = A2[..., 0, :], A2[..., 1, :], B2[..., 0, :], B2[..., 1, :]
        n_src, n_dst = np.cross(p0, p1), np.cross(q0, q1)
        ln_s = np.linalg.norm(n_src, axis=-1, keepdims=True)
        ln_d = np.linalg.norm(n_dst, axis=-1, keepdims=True)
        v_src = n_src * (np.linalg.norm(p0, axis=-1, keepdims=True) / (ln_s + self.eps))
        v_dst = n_dst * (np.linalg.norm(q0, axis=-1, keepdims=True) / (ln_d + self.eps))
        valid = (ln_s[..., 0] > 1e-9) & (ln_d[..., 0] > 1e-9)
        return np.where(valid[..., None, None], v_src[..., :, None] * v_dst[..., None, :], 0.0)

    def covariance(self, A: np.ndarray, B: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """H = sum_n a_n b_n^T (+ SOMA's virtual-normal term from the first two
        correspondences). A, B (..., N, 3); mask (..., N) selects valid rows."""
        if mask is not None:
            A = A * mask[..., None]
            B = B * mask[..., None]
        H = np.einsum("...ni,...nj->...ij", A, B)
        if A.shape[-2] >= 2:
            H = H + self.virtual_normal(A[..., :2, :], B[..., :2, :])
        return H

    def rotation_from_covariance(self, H: np.ndarray) -> np.ndarray:
        """align_vectors(method='auto'): SOMA regularises H with a weak identity
        prior, then takes its polar factor (Newton-Schulz with a Kabsch fallback
        in the reference). We take the polar factor directly with a batched SVD
        -- the same rotation, one LAPACK call instead of 30 matmul rounds."""
        prior_scale = np.maximum(np.abs(H).sum(-1).max(-1), self.eps)
        vol = np.abs(_det3(H)) / prior_scale ** 3
        rank_w = np.clip((self.rank_threshold - vol) / self.rank_threshold, 0.0, 1.0)
        H = H + (self.prior_strength * rank_w * prior_scale)[..., None, None] * np.eye(3)
        U, _, Vh = np.linalg.svd(H)
        UVt = U @ np.swapaxes(Vh, -1, -2)
        sign = np.where(_det3(UVt) < 0, -1.0, 1.0)
        U = U.copy()
        U[..., :, 2] *= sign[..., None]
        return U @ Vh

    def align(self, A: np.ndarray, B: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Rotation R with R @ b_n ~= a_n (SciPy align_vectors convention)."""
        if A.shape[-2] == 1:
            return self.rodrigues(A[..., 0, :], B[..., 0, :], self.eps)
        return self.rotation_from_covariance(self.covariance(A, B, mask))


# ----------------------------------------------------------------------------
# the stage
# ----------------------------------------------------------------------------
class SomaStageNumpy:
    """MANO params (B hands) -> SOMA-X right-hand vertices / joints, numpy only."""

    def __init__(self, npz_path: str, regressor_mlpackage: Optional[str] = None,
                 compute_units: str = "CPU_AND_NE", backend: str = "auto"):
        """`backend`: "c" (compiled soma_stage.c, ~0.1 ms), "numpy", or "auto"
        (C when a compiler is available, else numpy)."""
        with np.load(npz_path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        self.faces = d["faces"].astype(np.int64)
        self.num_verts = int(d["num_verts"])
        self.lod = str(d["lod"]) if "lod" in d else "mid"
        # regressor: weights pre-transposed to (in, out) so x @ W is one GEMM each
        self.W = {}
        for k in d:
            if k.startswith("net."):
                a = d[k].astype(np.float32)
                self.W[k[4:]] = np.ascontiguousarray(a.T) if k.endswith(".weight") and a.ndim == 2 else a
        self._nb_pose = 1 + max(int(k.split(".")[2]) for k in self.W if k.startswith("pose_net.blocks."))
        self._nb_id = 1 + max(int(k.split(".")[2]) for k in self.W if k.startswith("id_net.blocks."))
        self.coreml = None
        if regressor_mlpackage:
            self._load_coreml_regressor(regressor_mlpackage, compute_units)
        self.use_scales = bool(d["use_scales"])
        self.ref_orient = d["ref_orient"].astype(np.float64)                 # (25,3,3)
        self.ref_orient_parent_T = np.swapaxes(d["ref_orient_parent"], -1, -2).astype(np.float64)
        self.wrist_offset = d["wrist_offset"].astype(np.float64)
        # identity
        self.pca_mean = d["pca_mean"].astype(np.float64)                      # (V*3,)
        self.pca_matrix_scaled = (d["pca_matrix"].astype(np.float64)
                                  * np.sqrt(d["eigenvalues"].astype(np.float64))[:, None])  # (20, V*3)
        self.rest_scale = float(d["rest_scale"])                              # cm -> output unit
        # skeleton transfer
        self.parents = d["parents"].astype(np.int64)
        self.children = [list(np.flatnonzero(self.parents == j)) for j in range(NUM_JOINTS)]
        self.children = [[c for c in ch if c != j] for j, ch in enumerate(self.children)]
        self.Wpos = d["joint_pos_W"].astype(np.float64)                       # (25, V)
        self.bpos = d["joint_pos_b"].astype(np.float64)                       # (25, 3)
        self.T0 = d["st_bind_world"].astype(np.float64)                       # (25,4,4) skeleton-transfer bind
        self.R0 = self.T0[:, :3, :3]
        self.p0 = self.T0[:, :3, 3]
        self.bind_shape = d["st_bind_shape"].astype(np.float64)               # (V,3)
        st_W = d["st_skin_weights"].astype(np.float64)                        # (V,25)
        self.align = _Align(float(d["prior_strength"]), float(d["rank_threshold"]), int(d["ns_iters"]))
        self.skip_endjoints = bool(d["skip_endjoints"])
        # Per-joint skinned vertex sets S_j = {v : W[v, j] > 0.01}. The Procrustes
        # covariance H_j = sum_{v in S_j} (rest_v - t_j)(bind_v - p0_j)^T expands
        # into terms that are all products with the (F, V) indicator matrix, so
        # one matmul gives every joint's covariance at once (no padded gathers).
        self.fit_joints = [j for j in range(NUM_JOINTS)
                           if self.children[j] or not self.skip_endjoints]
        fj = np.array(self.fit_joints)
        self.fj = fj
        M = (st_W[:, fj] > 0.01).T.astype(np.float64)                        # (F, V)
        self.M = M
        self.M_bind = M @ self.bind_shape                                      # (F, 3)   sum_v bind_v
        self.M_count = M.sum(1)                                                # (F,)
        self.Mb = [np.ascontiguousarray(M * self.bind_shape[None, :, k]) for k in range(3)]  # (F, V) m_jv*bind_v[k]
        # the virtual-normal term uses the first two vertices (lowest ids) of each set
        first2 = np.stack([np.flatnonzero(M[i])[:2] for i in range(len(fj))])   # (F, 2)
        self.first2 = first2
        self.orig_first2 = self.bind_shape[first2] - self.p0[fj][:, None, :]   # (F, 2, 3)
        # children groups: joints with one child (rodrigues) and with >1 (procrustes)
        self.one_child = [(j, self.children[j][0]) for j in self.fit_joints if len(self.children[j]) == 1]
        self.multi_child = [(j, self.children[j]) for j in self.fit_joints if len(self.children[j]) > 1]
        self.end_joints = [j for j in range(NUM_JOINTS) if not self.children[j] and self.skip_endjoints]
        # skinning
        self.skin_W = d["skin_weights"].astype(np.float64)                    # (V,25)
        self.bind_local_R = d["bind_pose_local"].astype(np.float64)[:, :3, :3]
        self.bind_root_t = d["bind_pose_local"].astype(np.float64)[0, :3, 3]
        self.levels = self._levels(self.parents)
        self.skin_W_T = np.ascontiguousarray(self.skin_W)                      # (V, J) for the LBS matmul
        # MANO -> SOMA barycentric correspondence (fit metric), optional
        self.bary = None
        if "bary_F_src" in d:
            self.bary = dict(F=d["bary_F_src"].astype(np.int64),
                             tet=d["bary_F_src_tet"].astype(np.int64)[d["bary_face_ids"].astype(np.int64)],  # (S, 4)
                             bc=d["bary_coords"].astype(np.float64))                                      # (S, 4)
        self.valid_verts = d["valid_verts"].astype(bool) if "valid_verts" in d else None
        # C backend
        self._c = None
        self._ctx = None
        self.c_error: Optional[Exception] = None
        if backend in ("auto", "c"):
            try:
                self._init_c(st_W)
            except Exception as e:  # noqa: BLE001 -- no compiler etc.: numpy fallback
                if backend == "c":
                    raise
                self.c_error = e
        self.backend = "c" if self._c is not None else "numpy"

    def _init_c(self, st_W: np.ndarray) -> None:
        lib = _load_c_lib()
        J, V, F = NUM_JOINTS, self.num_verts, len(self.fj)
        d64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
        i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)
        sets = [np.flatnonzero(st_W[:, j] > 0.01) for j in self.fit_joints]
        set_off = i32(np.concatenate([[0], np.cumsum([len(s) for s in sets])]))
        set_vids = i32(np.concatenate(sets))
        child_off = i32(np.concatenate([[0], np.cumsum([len(c) for c in self.children])]))
        child_ids = i32(np.concatenate([np.asarray(c, dtype=np.int64) for c in self.children]) if any(self.children) else [])
        nz = [np.flatnonzero(self.skin_W[v] > 0) for v in range(V)]
        skin_off = i32(np.concatenate([[0], np.cumsum([len(n) for n in nz])]))
        skin_j = i32(np.concatenate(nz))
        skin_w = d64(np.concatenate([self.skin_W[v, n] for v, n in enumerate(nz)]))
        keep = [d64(self.pca_mean), d64(self.pca_matrix_scaled), d64(self.Wpos), d64(self.bpos), i32(self.parents),
                i32(self.fit_joints), set_off, set_vids, d64(self.p0), d64(self.R0), d64(self.bind_shape),
                child_off, child_ids, skin_off, skin_j, skin_w, d64(self.bind_local_R), d64(self.bind_root_t)]
        self._c_keep = keep  # the C context only holds pointers
        D, I = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        ptr = lambda a: a.ctypes.data_as(D if a.dtype == np.float64 else I)
        (pca_mean, pca_matrix, Wpos, bpos, parents, fit_joints, set_off, set_vids, p0, R0, bind_shape,
         child_off, child_ids, skin_off, skin_j, skin_w, bind_local_R, bind_root_t) = keep
        self._ctx = lib.soma_create(V, J, F, NUM_IDENTITY, ptr(pca_mean), ptr(pca_matrix), self.rest_scale,
                                    ptr(Wpos), ptr(bpos), ptr(parents), ptr(fit_joints), ptr(set_off), ptr(set_vids),
                                    ptr(p0), ptr(R0), ptr(bind_shape), ptr(child_off), ptr(child_ids),
                                    ptr(skin_off), ptr(skin_j), ptr(skin_w), ptr(bind_local_R), ptr(bind_root_t),
                                    self.align.prior_strength, self.align.rank_threshold, self.align.ns_iters,
                                    int(self.skip_endjoints))
        self._c = lib

    def __del__(self):
        try:
            if getattr(self, "_c", None) is not None and self._ctx:
                self._c.soma_destroy(self._ctx)
        except Exception:  # noqa: BLE001
            pass

    def _pose_c(self, p: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        B = p["identity"].shape[0]
        D = ctypes.POINTER(ctypes.c_double)
        pose = np.ascontiguousarray(p["pose"], dtype=np.float64).reshape(B, NUM_JOINTS, 9)
        root_t = np.ascontiguousarray(p["root_translation"], dtype=np.float64).reshape(B, 3)
        ident = np.ascontiguousarray(p["identity"], dtype=np.float64).reshape(B, NUM_IDENTITY)
        sc = p.get("scale_params")
        scales = None if sc is None else np.ascontiguousarray(sc, dtype=np.float64).reshape(B, NUM_SCALES)
        verts = np.empty((B, self.num_verts, 3), dtype=np.float64)
        joints = np.empty((B, NUM_JOINTS, 3), dtype=np.float64)
        self._c.soma_pose(self._ctx, B, pose.ctypes.data_as(D), root_t.ctypes.data_as(D), ident.ctypes.data_as(D),
                          None if scales is None else scales.ctypes.data_as(D),
                          verts.ctypes.data_as(D), joints.ctypes.data_as(D))
        return verts, joints

    def _load_coreml_regressor(self, mlpackage: str, compute_units: str) -> None:
        import coremltools as ct
        from .inference import _load_prediction_model
        self.coreml = _load_prediction_model(mlpackage, getattr(ct.ComputeUnit, compute_units))
        self._coreml_batches = None
        try:  # enumerated batch sizes the model was exported with
            desc = ct.models.MLModel(mlpackage, skip_model_load=True).get_spec().description
            enum = desc.input[0].type.multiArrayType.enumeratedShapes.shapes
            self._coreml_batches = sorted(int(s.shape[0]) for s in enum) or None
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _levels(parents):
        depth = np.zeros(len(parents), dtype=np.int64)
        for j in range(len(parents)):
            p = parents[j]
            depth[j] = 0 if p < 0 or p == j else depth[p] + 1
        return [np.flatnonzero(depth == k) for k in range(depth.max() + 1)]

    # ---- regressor -------------------------------------------------------
    def _resmlp(self, prefix: str, x: np.ndarray, nblocks: int) -> np.ndarray:
        W = self.W
        h = x @ W[f"{prefix}.inp.weight"] + W[f"{prefix}.inp.bias"]
        for i in range(nblocks):
            p = f"{prefix}.blocks.{i}"
            y = _layer_norm(h, W[f"{p}.norm.weight"], W[f"{p}.norm.bias"])
            y = _gelu(y @ W[f"{p}.fc1.weight"] + W[f"{p}.fc1.bias"])
            h = h + y @ W[f"{p}.fc2.weight"] + W[f"{p}.fc2.bias"]
        h = _layer_norm(h, W[f"{prefix}.norm.weight"], W[f"{prefix}.norm.bias"])
        return h @ W[f"{prefix}.out.weight"] + W[f"{prefix}.out.bias"]

    def _heads(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(B,100) -> raw pose head (B,153), raw identity head (B,44)."""
        if self.coreml is None:
            return self._resmlp("pose_net", x, self._nb_pose), self._resmlp("id_net", x[:, 90:], self._nb_id)
        B = x.shape[0]
        sizes = self._coreml_batches
        if sizes is None or B in sizes:
            chunks = [x]
        else:  # split into supported batch sizes (largest first)
            chunks, s = [], 0
            while s < B:
                n = max([b for b in sizes if b <= B - s] or [min(sizes)])
                chunks.append(x[s:s + n]); s += n
        po, io = [], []
        for c in chunks:
            n = c.shape[0]
            if sizes is not None and n not in sizes:  # pad up to the smallest supported size
                n2 = min(b for b in sizes if b >= n)
                c = np.concatenate([c, np.zeros((n2 - n, c.shape[1]), np.float32)])
            out = self.coreml.predict({"x": np.ascontiguousarray(c.reshape(c.shape[0], 100, 1, 1), dtype=np.float32)})
            po.append(np.asarray(out["pose_raw"], np.float32)[:min(n, c.shape[0])])
            io.append(np.asarray(out["id_raw"], np.float32)[:min(n, c.shape[0])])
        return np.concatenate(po)[:B], np.concatenate(io)[:B]

    def regress(self, hand_pose: np.ndarray, betas: np.ndarray,
                global_orient: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        B = betas.shape[0]
        hp = np.asarray(hand_pose, np.float32).reshape(B, 15, 3, 3)
        betas = np.asarray(betas, np.float32).reshape(B, 10)
        x = np.concatenate([hp[:, :, :2, :].reshape(B, -1), betas], axis=1)  # rot6d = first two rows
        o, i = self._heads(x)
        d6 = o[:, :NUM_JOINTS * 6].reshape(B, NUM_JOINTS, 6) + ROT6D_IDENTITY
        rel = _rot6d_to_matrix(d6.astype(np.float64))
        root_t = o[:, NUM_JOINTS * 6:].astype(np.float64) + self.W["root_t_bias"]
        identity = i[:, :NUM_IDENTITY].astype(np.float64)
        scales = np.exp(i[:, NUM_IDENTITY:].astype(np.float64)) if self.use_scales else None
        absr = self.ref_orient_parent_T @ rel @ self.ref_orient
        wrist = self.wrist_offset @ absr[:, 0]
        rot = np.concatenate([wrist[:, None], absr[:, 1:]], axis=1)
        if global_orient is not None:
            go = np.asarray(global_orient, np.float64).reshape(B, 3, 3)
            rot = np.concatenate([(go @ rot[:, 0])[:, None], rot[:, 1:]], axis=1)
            root_t = np.einsum("bij,bj->bi", go, root_t)
        return dict(pose=rot, root_translation=root_t, identity=identity, scale_params=scales)

    # ---- SOMA geometry --------------------------------------------------------
    def rest_shape(self, identity: np.ndarray) -> np.ndarray:
        flat = identity @ self.pca_matrix_scaled + self.pca_mean
        return flat.reshape(identity.shape[0], -1, 3) * self.rest_scale

    def fit_skeleton(self, rest: np.ndarray) -> np.ndarray:
        """SkeletonTransfer.fit: (B,V,3) rest shape -> (B,25,4,4) bind transforms."""
        B = rest.shape[0]
        t = (self.Wpos @ rest) + self.bpos                                     # (B,25,3)
        R = np.tile(self.R0, (B, 1, 1, 1))                                     # (B,25,3,3)
        fj = self.fj
        tj, pj = t[:, fj], self.p0[fj]                                         # (B,F,3), (F,3)
        # H_j = sum_v m_jv (rest_v - t_j)(bind_v - p0_j)^T
        #     = sum_v m_jv rest_v bind_v^T - (sum_v m_jv rest_v) p0_j^T - t_j (sum_v m_jv bind_v)^T + n_j t_j p0_j^T
        H = self._cov_term(rest)                                               # (B,F,3,3)
        M_rest = self.M @ rest                                                 # (B,F,3)  sum_v m_jv rest_v
        H = H - M_rest[..., :, None] * pj[None, :, None, :] \
            - tj[..., :, None] * self.M_bind[None, :, None, :] \
            + self.M_count[None, :, None, None] * tj[..., :, None] * pj[None, :, None, :]
        # SOMA's virtual-normal conditioning term from the first two correspondences
        A2 = rest[:, self.first2] - tj[:, :, None, :]                          # (B,F,2,3)
        B2 = np.broadcast_to(self.orig_first2[None], A2.shape)
        H = H + self.align.virtual_normal(A2, B2)
        R_init = self.align.rotation_from_covariance(H)                        # (B,F,3,3)
        R_new = np.empty_like(R_init)
        # joints with a single child: Rodrigues on the child direction
        if self.one_child:
            js = np.array([j for j, _ in self.one_child]); cs = np.array([c for _, c in self.one_child])
            idx = np.array([self.fit_joints.index(j) for j in js])
            po = np.einsum("bjik,jk->bji", R_init[:, idx], self.p0[cs] - self.p0[js])
            pn = t[:, cs] - t[:, js]
            A = self.align.rodrigues(pn, po, self.align.eps)
            R_new[:, idx] = A @ R_init[:, idx] @ self.R0[js]
        # joints with several children: Procrustes over the child directions
        for j, ch in self.multi_child:
            k = self.fit_joints.index(j); ch = np.array(ch)
            po = np.einsum("bik,ck->bci", R_init[:, k], self.p0[ch] - self.p0[j])
            pn = t[:, ch] - t[:, j][:, None, :]
            A = self.align.align(pn, po)
            R_new[:, k] = A @ R_init[:, k] @ self.R0[j]
        R[:, fj] = R_new
        for j in self.end_joints:          # end joints inherit the (fitted) parent orientation
            R[:, j] = R[:, self.parents[j]]
        return _se3(R, t)

    def _cov_term(self, rest: np.ndarray) -> np.ndarray:
        """sum_v m_jv rest_v bind_v^T for every fitted joint -> (B,F,3,3), three (F,V)@(V,3) matmuls."""
        out = np.empty((rest.shape[0], len(self.fj), 3, 3))
        for k in range(3):
            out[:, :, :, k] = self.Mb[k] @ rest                                # (B,F,3): sum_v m_jv bind_v[k] rest_v
        return out

    def _fk(self, local_R: np.ndarray, local_t: np.ndarray) -> np.ndarray:
        T_local = _se3(local_R, local_t)
        world = T_local.copy()
        for lv in self.levels[1:]:
            world[:, lv] = world[:, self.parents[lv]] @ T_local[:, lv]
        return world

    def _local_translations(self, T_world: np.ndarray) -> np.ndarray:
        inv_parent = _se3_inverse(T_world[:, self.parents])
        local = inv_parent @ T_world
        local[:, 0] = T_world[:, 0]
        return local[:, :, :3, 3]

    @staticmethod
    def _lbs(rest: np.ndarray, W: np.ndarray, bone: np.ndarray) -> np.ndarray:
        """rest (B,V,3), W (V,J), bone (B,J,4,4) -> (B,V,3)."""
        B, J = bone.shape[:2]
        Tv = (W @ bone[:, :, :3, :].reshape(B, J, 12)).reshape(B, -1, 3, 4)  # (B,V,3,4) blended transforms
        return (Tv[..., :3] * rest[:, :, None, :]).sum(-1) + Tv[..., 3]

    def forward(self, hand_pose: np.ndarray, betas: np.ndarray,
                global_orient: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(B,15,3,3), (B,10), (B,3,3) -> vertices (B,V,3), joints (B,25,3) in metres,
        MANO wrist at the origin (same frame as SOMAHandLayer in mano2soma-x)."""
        p = self.regress(hand_pose, betas, global_orient)
        return self.pose(p)

    def pose(self, p: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if self._c is not None:
            return self._pose_c(p)
        return self.pose_numpy(p)

    def pose_numpy(self, p: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        B = p["identity"].shape[0]
        # prepare_identity
        rest = self.rest_shape(p["identity"])
        T_fit = self.fit_skeleton(rest)
        local_t = self._local_translations(T_fit)
        # repose to the canonical bind pose
        lt = local_t.copy(); lt[:, 0] = self.bind_root_t
        T_bind = self._fk(np.broadcast_to(self.bind_local_R, (B, NUM_JOINTS, 3, 3)), lt)
        rest_bind = self._lbs(rest, self.skin_W, T_bind @ _se3_inverse(T_fit))
        # pose
        local_t = self._local_translations(T_bind)
        if p.get("scale_params") is not None:
            local_t = local_t.copy()
            local_t[:, 1:] *= p["scale_params"][:, :, None]
        local_t[:, 0] = p["root_translation"]
        T_world = self._fk(p["pose"], local_t)
        verts = self._lbs(rest_bind, self.skin_W, T_world @ _se3_inverse(T_bind))
        return verts, T_world[:, :, :3, 3]

    # ---- fit metric ----------------------------------------------------------
    def mano_to_soma(self, mano_verts: np.ndarray) -> np.ndarray:
        """SOMA-X BarycentricInterpolator: MANO verts (B,778,3) -> SOMA topology (B,V,3)."""
        if self.bary is None:
            raise RuntimeError("this asset file has no MANO->SOMA correspondence")
        F = self.bary["F"]
        f0, f1, f2 = mano_verts[:, F[:, 0]], mano_verts[:, F[:, 1]], mano_verts[:, F[:, 2]]
        apex = f0 + np.cross(f1 - f0, f2 - f0)                                 # tet_normal_scale == "area"
        V_tet = np.concatenate([mano_verts, apex], axis=1)
        tet, bc = self.bary["tet"], self.bary["bc"]
        return (V_tet[:, tet] * bc[None, :, :, None]).sum(2)

    def fit_error_mm(self, soma_verts: np.ndarray, mano_verts_wrist_local: np.ndarray) -> np.ndarray:
        """Mean distance (mm) between SOMA verts and the MANO mesh transferred
        onto SOMA topology with SOMA-X's barycentric correspondence."""
        target = self.mano_to_soma(np.asarray(mano_verts_wrist_local, np.float64))
        d = np.linalg.norm(soma_verts - target, axis=-1)
        if self.valid_verts is not None:
            d = d[:, self.valid_verts]
        return d.mean(-1) * 1000.0
