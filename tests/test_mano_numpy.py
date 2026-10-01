"""Tests for fasthamer.mano (chumpy/scipy-free MANO loading + numpy forward).

    python fasthamer/tests/test_mano_numpy.py

Needs a MANO_RIGHT.pkl: FASTHAMER_MANO_PATH, ./_DATA/data/mano/, or the
fasthamer cache. If torch + smplx + the hamer repo are importable (run from
the hamer-realtime root with its venv), also checks parity against HaMeR's
reference MANO layer; otherwise only self-consistency.
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.getcwd())

from fasthamer.assets import mano_npz_path  # noqa: E402
from fasthamer.mano import (EXTRA_JOINT_IDXS, JOINT_MAP, ManoRight,  # noqa: E402
                            load_mano_npz, load_mano_pkl, save_mano_npz)


def find_pkl():
    for c in (os.environ.get("FASTHAMER_MANO_PATH"), "_DATA/data/mano/MANO_RIGHT.pkl",
              os.path.join(os.path.dirname(mano_npz_path()), "MANO_RIGHT.pkl")):
        if c and os.path.isfile(c):
            return c
    return None


def rand_rot(rng, n, scale):
    aa = rng.normal(size=(n, 3)) * scale
    th = np.linalg.norm(aa, axis=1, keepdims=True)
    k = aa / np.maximum(th, 1e-9)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    return np.eye(3) + np.sin(th)[:, :, None] * K + (1 - np.cos(th))[:, :, None] * (K @ K)


def main():
    pkl = find_pkl()
    if pkl is None:
        print("SKIP: no MANO_RIGHT.pkl available")
        return 0
    assert "chumpy" not in sys.modules and "scipy" not in sys.modules
    data = load_mano_pkl(pkl)
    assert "chumpy" not in sys.modules and "scipy" not in sys.modules, \
        "pkl loading must not import chumpy/scipy"

    with tempfile.TemporaryDirectory() as tmp:
        npz = os.path.join(tmp, "m.npz")
        save_mano_npz(data, npz)
        data2 = load_mano_npz(npz)
    for k in data:
        assert np.array_equal(data[k], data2[k]), k

    m = ManoRight(data)
    rng = np.random.default_rng(0)
    # rest pose: joints come from the regressor, tips from the template
    v, j = m(np.eye(3), np.tile(np.eye(3), (15, 1, 1)), np.zeros(10))
    assert np.allclose(v, data["v_template"]), "rest pose must be the template"
    J = data["J_regressor"] @ data["v_template"]
    ref = np.concatenate([J, data["v_template"][EXTRA_JOINT_IDXS]])[JOINT_MAP]
    assert np.allclose(j, ref)
    assert j.shape == (21, 3) and v.shape == (778, 3) and m.faces.shape == (1538, 3)
    # a rigid global rotation rotates everything rigidly
    R = rand_rot(rng, 1, 2.0)[0]
    v2, j2 = m(R, np.tile(np.eye(3), (15, 1, 1)), np.zeros(10))
    assert np.allclose(v2 - J[0], (v - J[0]) @ R.T, atol=1e-9)
    print(f"self-consistency OK ({pkl})")

    t0 = time.perf_counter()
    for _ in range(200):
        m(R, rand_rot(rng, 15, 0.5), rng.normal(size=10))
    print(f"numpy MANO forward: {(time.perf_counter() - t0) / 200 * 1e3:.2f} ms/hand")

    try:
        import torch
        import realtime_demo as rt
        from hamer.configs import get_config
        from hamer.models import MANO
    except Exception as e:  # noqa: BLE001
        print(f"reference parity SKIPPED ({type(e).__name__}: {e})")
        return 0
    cfg = get_config(os.path.join(os.path.dirname(os.path.dirname(rt.DEFAULT_CHECKPOINT)),
                                  "model_config.yaml"), update_cachedir=True)
    ref = MANO(**{k.lower(): v for k, v in dict(cfg.MANO).items()})
    assert np.array_equal(np.asarray(ref.faces).astype(np.int64), m.faces)
    worst = 0.0
    for _ in range(50):
        go, hp, betas = rand_rot(rng, 1, 3.0)[0], rand_rot(rng, 15, 0.8), rng.normal(size=10) * 2
        v, j = m(go, hp, betas)
        with torch.no_grad():
            out = ref(global_orient=torch.from_numpy(go).float().reshape(1, 1, 3, 3),
                      hand_pose=torch.from_numpy(hp).float().reshape(1, 15, 3, 3),
                      betas=torch.from_numpy(betas).float().reshape(1, 10), pose2rot=False)
        worst = max(worst, float(np.abs(out.vertices[0].numpy() - v).max()),
                    float(np.abs(out.joints[0].numpy() - j).max()))
    print(f"vs HaMeR/smplx MANO (50 random poses): max abs diff {worst:.2e} m")
    assert worst < 1e-6
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
