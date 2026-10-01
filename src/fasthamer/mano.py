"""The MANO hand model (right hand) in plain numpy, built from the user's own
MANO_RIGHT.pkl.

fasthamer ships no MANO data. The CoreML model stops at the MANO *parameters*
(global_orient, hand_pose, betas, cam); this module turns them into the
778-vertex mesh and the 21 OpenPose-ordered joints exactly like HaMeR's
smplx-based MANO layer does (rotation-matrix input, `pose2rot=False`).

The pkl is parsed without chumpy or scipy: a small unpickler shim stands in
for their classes and the arrays are reconstructed by hand. The result is
cached as a plain .npz (`MANO_NPZ_NAME`) so later loads need no unpickling.
"""
import os
import pickle
import zipfile
from typing import Dict, Tuple

import numpy as np

NUM_VERTS = 778
NUM_JOINTS = 16
NUM_BETAS = 10
NUM_FACES = 1538
PKL_NAME = "MANO_RIGHT.pkl"
MANO_NPZ_NAME = "mano_right.npz"
NPZ_KEYS = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights",
            "parents", "faces")

# Fingertip vertices appended as extra joints (smplx `vertex_ids['mano']`:
# thumb, index, middle, ring, pinky), as in hamer.models.mano_wrapper.MANO.
EXTRA_JOINT_IDXS = np.array([744, 320, 443, 554, 671], dtype=np.int64)
# HaMeR's `mano_to_openpose`: [16 MANO joints + 5 tips] -> OpenPose order.
JOINT_MAP = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18,
                      10, 11, 12, 19, 7, 8, 9, 20], dtype=np.int64)


# ----------------------------------------------------------------------------
# chumpy/scipy-free pkl loading
# ----------------------------------------------------------------------------
class _Shim:
    """Stand-in for chumpy / scipy.sparse classes while unpickling: just keeps
    the pickled state so the arrays can be rebuilt by `_to_array`."""

    _origin = ("", "")

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy") or module.startswith("scipy.sparse"):
            return type(name, (_Shim,), {"_origin": (module, name)})
        return super().find_class(module, name)


def _to_array(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, (list, tuple, int, float)):
        return np.asarray(obj)
    if isinstance(obj, _Shim):
        d = obj.__dict__
        if "indptr" in d and "indices" in d and "data" in d:  # scipy CSC/CSR
            shape = tuple(d.get("_shape") or d.get("shape"))
            dense = np.zeros(shape, dtype=np.asarray(d["data"]).dtype)
            indptr, indices, data = (np.asarray(d[k]) for k in ("indptr", "indices", "data"))
            is_csr = "csr" in obj._origin[1].lower()
            for j in range(len(indptr) - 1):
                sl = slice(indptr[j], indptr[j + 1])
                if is_csr:
                    dense[j, indices[sl]] = data[sl]
                else:
                    dense[indices[sl], j] = data[sl]
            return dense
        if "idxs" in d and "a" in d:  # chumpy.reordering.Select (a.ravel()[idxs])
            src = _to_array(d["a"]).ravel()
            return src[np.asarray(d["idxs"])].reshape(tuple(d["preferred_shape"]))
        if "x" in d:  # chumpy.ch.Ch wrapping an ndarray
            return _to_array(d["x"])
    raise ValueError(f"cannot convert pickled object of type {type(obj).__name__} "
                     f"({getattr(obj, '_origin', '')}) to an array")


def load_mano_pkl(path: str) -> Dict[str, np.ndarray]:
    """Parse MANO_RIGHT.pkl into plain numpy arrays (no chumpy/scipy needed).

    Returns float64 arrays laid out like smplx's MANO buffers:
      v_template (778,3), shapedirs (778,3,10), posedirs (135, 2334),
      J_regressor (16,778), weights (778,16), parents (16,), faces (1538,3).
    """
    try:
        with open(path, "rb") as f:
            raw = _Unpickler(f, encoding="latin1").load()
        if not isinstance(raw, dict):
            raise ValueError("not a dict")
    except Exception as e:  # noqa: BLE001 -- anything that isn't a MANO pickle
        raise ValueError(f"{path} is not a MANO model pickle ({type(e).__name__}: {e}); "
                         "expected MANO_RIGHT.pkl from mano_v1_2.zip") from None
    try:
        v_template = np.asarray(_to_array(raw["v_template"]), dtype=np.float64)
        shapedirs = np.asarray(_to_array(raw["shapedirs"]), dtype=np.float64)
        posedirs = np.asarray(_to_array(raw["posedirs"]), dtype=np.float64)
        J_regressor = np.asarray(_to_array(raw["J_regressor"]), dtype=np.float64)
        weights = np.asarray(_to_array(raw["weights"]), dtype=np.float64)
        kintree = np.asarray(_to_array(raw["kintree_table"])).astype(np.int64)
        faces = np.asarray(_to_array(raw["f"])).astype(np.int64)
    except KeyError as e:
        raise ValueError(f"{path} does not look like a MANO model file (missing {e})") from None

    if shapedirs.ndim == 3 and shapedirs.shape[-1] > NUM_BETAS:
        shapedirs = shapedirs[:, :, :NUM_BETAS]
    posedirs = posedirs.reshape(-1, posedirs.shape[-1]).T  # (P, V*3) like smplx
    parents = kintree[0].copy()
    parents[0] = -1

    data = dict(v_template=v_template, shapedirs=shapedirs, posedirs=posedirs,
                J_regressor=J_regressor, weights=weights, parents=parents, faces=faces)
    _validate(data, path)
    return data


def _validate(data: Dict[str, np.ndarray], src: str) -> None:
    expect = {"v_template": (NUM_VERTS, 3), "shapedirs": (NUM_VERTS, 3, NUM_BETAS),
              "posedirs": (9 * (NUM_JOINTS - 1), NUM_VERTS * 3),
              "J_regressor": (NUM_JOINTS, NUM_VERTS), "weights": (NUM_VERTS, NUM_JOINTS),
              "parents": (NUM_JOINTS,), "faces": (NUM_FACES, 3)}
    for k, shp in expect.items():
        if tuple(data[k].shape) != shp:
            raise ValueError(f"{src}: {k} has shape {tuple(data[k].shape)}, expected {shp} "
                             "(is this MANO_RIGHT.pkl from mano_v1_2?)")


def save_mano_npz(data: Dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{k: data[k] for k in NPZ_KEYS})
    os.replace(tmp, path)


def load_mano_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        data = {k: z[k] for k in NPZ_KEYS}
    _validate(data, path)
    return data


def extract_right_pkl(zip_path: str, dest_pkl: str) -> str:
    """Pull MANO_RIGHT.pkl out of mano_v1_2.zip into `dest_pkl`."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if os.path.basename(n) == PKL_NAME]
        if not members:
            raise ValueError(f"{zip_path} contains no {PKL_NAME}")
        os.makedirs(os.path.dirname(os.path.abspath(dest_pkl)), exist_ok=True)
        with zf.open(members[0]) as src, open(dest_pkl, "wb") as dst:
            dst.write(src.read())
    return dest_pkl


# ----------------------------------------------------------------------------
# forward model
# ----------------------------------------------------------------------------
class ManoRight:
    """Right-hand MANO forward pass in numpy (float64).

        mano = ManoRight(load_mano_npz(path))
        vertices, joints = mano(global_orient, hand_pose, betas)

    global_orient (3,3) and hand_pose (15,3,3) are rotation matrices (HaMeR's
    head output), betas (10,). Returns vertices (778,3) and the 21 joints in
    HaMeR/OpenPose order (16 MANO joints + 5 fingertips, reordered), in the
    MANO root frame -- identical to HaMeR's `mano_output.vertices/.joints`.
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        _validate(data, "mano data")
        self.v_template = np.ascontiguousarray(data["v_template"], dtype=np.float64)
        self.shapedirs = np.ascontiguousarray(data["shapedirs"], dtype=np.float64)
        self.posedirs = np.ascontiguousarray(data["posedirs"], dtype=np.float64)
        self.J_regressor = np.ascontiguousarray(data["J_regressor"], dtype=np.float64)
        self.weights = np.ascontiguousarray(data["weights"], dtype=np.float64)
        self.parents = np.asarray(data["parents"], dtype=np.int64)
        self.faces = np.ascontiguousarray(data["faces"], dtype=np.int64)
        self._eye3 = np.eye(3)

    def __call__(self, global_orient: np.ndarray, hand_pose: np.ndarray,
                 betas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = np.concatenate([np.asarray(global_orient, np.float64).reshape(1, 3, 3),
                            np.asarray(hand_pose, np.float64).reshape(NUM_JOINTS - 1, 3, 3)])
        betas = np.asarray(betas, np.float64).reshape(NUM_BETAS)

        # shape blend shapes + joint locations in the rest pose
        v_shaped = self.v_template + self.shapedirs @ betas
        J = self.J_regressor @ v_shaped                            # (16, 3)
        # pose blend shapes (pose feature = R - I for the 15 non-root joints)
        pose_feature = (R[1:] - self._eye3).reshape(-1)           # (135,)
        v_posed = v_shaped + (pose_feature @ self.posedirs).reshape(NUM_VERTS, 3)

        # rigid transforms down the kinematic chain
        rel = J.copy()
        rel[1:] -= J[self.parents[1:]]
        local = np.tile(np.eye(4), (NUM_JOINTS, 1, 1))
        local[:, :3, :3] = R
        local[:, :3, 3] = rel
        G = np.empty_like(local)
        G[0] = local[0]
        for i in range(1, NUM_JOINTS):
            G[i] = G[self.parents[i]] @ local[i]
        posed_joints = G[:, :3, 3].copy()
        A = G.copy()                                               # relative to rest joints
        A[:, :3, 3] -= np.einsum("jab,jb->ja", G[:, :3, :3], J)

        # linear blend skinning
        T = (self.weights @ A.reshape(NUM_JOINTS, 16)).reshape(NUM_VERTS, 4, 4)
        verts = np.einsum("vab,vb->va", T[:, :3, :3], v_posed) + T[:, :3, 3]

        joints = np.concatenate([posed_joints, verts[EXTRA_JOINT_IDXS]])[JOINT_MAP]
        return verts, joints
