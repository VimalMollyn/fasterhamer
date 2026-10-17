"""Optional SOMA-X hand overlay.

Turns HaMeR's MANO parameters into a SOMA-X hand mesh with the mano2soma-x
regressor (https://github.com/VimalMollyn/mano2soma-x) so the two meshes can
be compared on the live image. Needs torch, py-soma-x, warp-lang and the
`mano_to_somax` package; all of them are imported lazily, so fasthamer itself
stays torch-free.

    soma = SomaOverlay("checkpoints/mano2somax_resmlp512x4.pt")
    soma_result, fit_mm = soma.convert(result)          # same frame as result
    img = MeshRenderer(soma.faces, color=SOMA_COLOR, close_wrist=False).render(img, soma_result)

`fit_mm` is the mean per-vertex distance (mm) between the regressed SOMA hand
and HaMeR's MANO mesh transferred onto SOMA topology with SOMA-X's own
barycentric correspondence -- the metric mano2soma-x was trained and
evaluated with. It needs the user's MANO_RIGHT.pkl (default: the copy in the
fasthamer cache); without it `convert` still returns the meshes, with None
errors.
"""
import dataclasses
import os
from typing import List, Optional, Tuple

import numpy as np

from .assets import mano_cache_dir
from .mano import PKL_NAME
from .types import HandMeshResult

SOMA_COLOR = (0.96, 0.56, 0.16)  # orange, against fasthamer's steel-blue MANO


class SomaOverlay:
    def __init__(self, checkpoint: str, lod: str = "mid", device: str = "cpu",
                 mano_pkl: Optional[str] = None):
        try:
            import torch
            from soma.hand import SOMAHandLayer
            from mano_to_somax.infer import ManoToSomaX
        except ImportError as e:
            raise ImportError(
                "the SOMA overlay needs torch, py-soma-x, warp-lang and the "
                "mano2soma-x package (pip install -e <mano2soma-x checkout>)") from e
        self.torch = torch
        self.device = torch.device(device)
        self.lod = lod
        self.net = ManoToSomaX.from_checkpoint(checkpoint, device=device)
        self.layer = SOMAHandLayer(hand_type="right", device=device, identity_model_type="soma",
                                   lod=lod, correctives_model_path=None)
        self.faces = self.layer.faces.detach().cpu().numpy().astype(np.int64)
        self.num_verts = int(self.faces.max()) + 1

        # Exact fit metric: MANO mesh -> SOMA topology via SOMA-X's MANO identity
        # backend (needs the user's MANO pkl). 38 wrist-boundary verts have no
        # MANO correspondence and are excluded, as in mano2soma-x's training.
        self._interp = None
        self._valid = None
        if mano_pkl is None:
            cand = os.path.join(mano_cache_dir(), PKL_NAME)
            mano_pkl = cand if os.path.isfile(cand) else None
        if mano_pkl:
            ml = SOMAHandLayer(hand_type="right", device=device, identity_model_type="mano",
                               identity_model_kwargs={"model_path": mano_pkl}, lod=lod,
                               correctives_model_path=None)
            self._interp = ml.identity_model._to_soma_interp
            valid = np.ones(self.num_verts, dtype=bool)
            excl = ml.excluded_vert_ids
            if excl is not None and len(excl):
                valid[np.asarray(excl, dtype=np.int64)] = False
            self._valid = torch.from_numpy(valid).to(self.device)

        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy conversion so Warp's kernels are loaded before the first frame."""
        torch = self.torch
        eye = torch.eye(3, device=self.device)
        with torch.no_grad():
            p = self.net(eye.repeat(1, 15, 1, 1), torch.zeros(1, 10, device=self.device),
                         global_orient=eye[None])
            self.layer.prepare_identity(p["identity"], scale_params=p["scale_params"])
            self.layer.pose(p["pose"], pose2rot=False, absolute_pose=True,
                            global_translation=p["root_translation"])
            if self._interp is not None:
                self._interp(torch.zeros(1, 778, 3, device=self.device))

    @property
    def can_measure(self) -> bool:
        return self._interp is not None

    @staticmethod
    def _right_frame(x: np.ndarray, is_right: bool) -> np.ndarray:
        """Undo fasthamer's left-hand x-flip: coordinates in the right-hand MANO frame."""
        x = np.array(x, dtype=np.float64, copy=True)
        if not is_right:
            x[..., 0] = -x[..., 0]
        return x

    def convert(self, result: HandMeshResult) -> Tuple[HandMeshResult, List[Optional[float]]]:
        """SOMA-X meshes for every hand in `result` (same hand-centred frame, so
        `cam_t` / `vertices_camera` keep working) plus per-hand fit error in mm
        (None when no MANO pkl is available)."""
        torch = self.torch
        hands = [h for h in result.hands
                 if h.hand_pose is not None and h.betas is not None and h.global_orient is not None]
        out_result = HandMeshResult(hands=[], focal=result.focal, cx=result.cx, cy=result.cy,
                                    image_size=result.image_size)
        if not hands:
            return out_result, []
        B = len(hands)
        f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(self.device)
        hp = f32(np.stack([h.hand_pose for h in hands])).reshape(B, 15, 3, 3)
        go = f32(np.stack([h.global_orient for h in hands])).reshape(B, 3, 3)
        betas = f32(np.stack([h.betas for h in hands])).reshape(B, 10)
        # SOMA output lives in the right-hand MANO frame with the MANO wrist joint
        # at the origin; fasthamer's meshes keep MANO's own origin.
        wrist = np.stack([self._right_frame(h.keypoints_3d[0], h.is_right) for h in hands])  # (B, 3)

        with torch.no_grad():
            p = self.net(hp, betas, global_orient=go)
            self.layer.prepare_identity(p["identity"], scale_params=p["scale_params"])
            soma = self.layer.pose(p["pose"], pose2rot=False, absolute_pose=True,
                                   global_translation=p["root_translation"])["vertices"]  # (B, V, 3)
            errs: List[Optional[float]] = [None] * B
            if self._interp is not None:
                mano = np.stack([self._right_frame(h.vertices, h.is_right) for h in hands])
                target = self._interp(f32(mano - wrist[:, None, :]))
                d = (soma - target).norm(dim=-1)[:, self._valid].mean(dim=1) * 1000.0
                errs = [float(x) for x in d.cpu()]

        soma_np = soma.cpu().numpy().astype(np.float64)
        for i, h in enumerate(hands):
            v = soma_np[i] + wrist[i]
            if not h.is_right:
                v[:, 0] = -v[:, 0]
            out_result.hands.append(dataclasses.replace(h, vertices=v))
        return out_result, errs
