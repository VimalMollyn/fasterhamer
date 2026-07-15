"""Dev parity tests — run from the hamer-realtime repo root with its venv:

    FASTHAMER_MODEL_DIR=_DATA/fasthamer_bundle python fasthamer/tests/test_parity.py

Needs the full hamer repo + torch (not shipped with fasthamer). Checks that:
  A. fasthamer's torch-free preprocessing matches hamer's ViTDetDataset crops
     and the end-to-end outputs match realtime_demo's FullHamerCoreML.
  B. the returned MANO parameters reproduce the returned vertices when pushed
     through the reference smplx MANO layer.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())  # hamer repo root

import fasthamer
from fasthamer.preprocess import preprocess_hands
from fasthamer.assets import resolve_model_dir

import realtime_demo as rt
from hamer.datasets.vitdet_dataset import ViTDetDataset

IMAGES = ["example_data/test1.jpg", "example_data/test3.jpg", "example_data/test5.jpg"]


def main():
    bundle = resolve_model_dir()
    model_cfg = rt.load_cfg(rt.DEFAULT_CHECKPOINT)
    ref_engine = rt.FullHamerCoreML(os.path.join(bundle, "hamer_mano.mlpackage"),
                                    os.path.join(bundle, "mano_faces.npy"), model_cfg)
    hm = fasthamer.load(mode="image", model_dir=bundle)

    mano = None  # lazy: torch MANO layer for test B
    ok = True
    for path in IMAGES:
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes, is_right = hm.detector.detect(rgb, 0)
        if not boxes:
            print(f"{path}: no hands detected, skipping")
            continue

        # --- A1: crop parity vs ViTDetDataset ---
        ds = ViTDetDataset(model_cfg, bgr, np.stack(boxes),
                           np.array(is_right, np.float32), rescale_factor=2.0)
        crops = preprocess_hands(rgb, np.stack(boxes), np.array(is_right),
                                 rescale_factor=2.0)
        crop_err = max(float(np.abs(np.asarray(ds[i]["img"]) - crops[i].img).max())
                       for i in range(len(ds)))

        # --- A2: end-to-end parity vs realtime_demo.FullHamerCoreML ---
        ref = ref_engine.run(bgr, boxes, is_right, rescale_factor=2.0)
        res = hm(rgb)
        vert_err = max(float(np.abs(rv - h.vertices).max())
                       for rv, h in zip(ref["verts"], res.hands))
        camt_err = max(float(np.abs(rc - h.cam_t).max())
                       for rc, h in zip(ref["cam_t"], res.hands))
        focal_err = abs(ref["focal"] - res.focal)

        # --- B: MANO params reproduce vertices ---
        if mano is None:
            import torch
            from hamer.models import MANO
            from hamer.configs import get_config
            cfg = get_config(os.path.join(os.path.dirname(
                os.path.dirname(rt.DEFAULT_CHECKPOINT)), "model_config.yaml"),
                update_cachedir=True)
            mano = MANO(**{k.lower(): v for k, v in dict(cfg.MANO).items()})
        import torch
        mano_err = 0.0
        for h in res.hands:
            with torch.no_grad():
                out = mano(
                    global_orient=torch.from_numpy(h.global_orient).float().reshape(1, 1, 3, 3),
                    hand_pose=torch.from_numpy(h.hand_pose).float().reshape(1, 15, 3, 3),
                    betas=torch.from_numpy(h.betas).float().reshape(1, 10),
                    pose2rot=False)
            v = out.vertices[0].numpy().astype(np.float64)
            if not h.is_right:
                v[:, 0] = -v[:, 0]
            mano_err = max(mano_err, float(np.abs(v - h.vertices).max()))

        line = (f"{path}: {len(res)} hands | crop={crop_err:.2e} "
                f"verts={vert_err:.2e} cam_t={camt_err:.2e} "
                f"focal={focal_err:.2e} mano_recon={mano_err*1000:.3f}mm")
        # Crops: bitwise except images that trigger the anti-alias blur (cv2 vs
        # skimage Gaussian, <1e-3 px-value delta). Verts/cam_t: identical when
        # crops are bitwise; blur-path crop deltas pass through the fp16 model
        # at its own noise floor (~1e-3 m). MANO recon: fp16/palettization
        # noise of the on-device mesh vs fp32 smplx.
        passed = crop_err < 1e-3 and vert_err < 2e-3 and camt_err < 5e-2 \
            and focal_err < 1e-6 and mano_err < 2e-3
        ok &= passed
        print(("PASS " if passed else "FAIL ") + line)

    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
