"""fasthamer.soma_np (torch-free SOMA-X hand stage) vs the reference torch/Warp implementation.

    FASTHAMER_SOMA_NPZ=/path/to/mano2somax_resmlp512x4_numpy.npz \
    FASTHAMER_SOMA_CKPT=/path/to/mano2somax_resmlp512x4.pt \
        python fasthamer/tests/test_soma_np.py

Needs the mano2soma-x venv (torch, py-soma-x, warp, mano_to_somax). Skips when the
assets or the reference stack are missing. Also checks the optional CoreML regressor
(`<npz dir>/mano2somax_regressor_fp32.mlpackage`) when present.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())

from fasthamer.soma_np import SomaStageNumpy  # noqa: E402

NPZ = os.environ.get("FASTHAMER_SOMA_NPZ")
CKPT = os.environ.get("FASTHAMER_SOMA_CKPT")


def main():
    if not (NPZ and os.path.isfile(NPZ)):
        print("SKIP: set FASTHAMER_SOMA_NPZ")
        return 0
    try:
        import torch
        from soma.hand import SOMAHandLayer
        from mano_to_somax.infer import ManoToSomaX
        from mano_to_somax.rotations import axis_angle_to_matrix
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: reference stack unavailable ({type(e).__name__}: {e})")
        return 0
    if not (CKPT and os.path.isfile(CKPT)):
        print("SKIP: set FASTHAMER_SOMA_CKPT")
        return 0

    net = ManoToSomaX.from_checkpoint(CKPT, device="cpu")
    layer = SOMAHandLayer(hand_type="right", device="cpu", identity_model_type="soma",
                          lod="mid", correctives_model_path=None)

    def ref(hp, b, go):
        with torch.no_grad():
            p = net(hp, b, global_orient=go)
            layer.prepare_identity(p["identity"], scale_params=p["scale_params"])
            o = layer.pose(p["pose"], pose2rot=False, absolute_pose=True,
                           global_translation=p["root_translation"])
        return o["vertices"].numpy(), o["joints"].numpy()

    def rand_inputs(seed, B):
        g = torch.Generator().manual_seed(seed)
        return (axis_angle_to_matrix(torch.randn(B, 15, 3, generator=g) * (0.5 if seed % 2 else 1.0)),
                torch.randn(B, 10, generator=g), axis_angle_to_matrix(torch.randn(B, 3, generator=g)))

    engines = {"numpy": SomaStageNumpy(NPZ, backend="numpy")}
    st_c = SomaStageNumpy(NPZ, backend="auto")
    if st_c.backend == "c":
        engines["C geometry"] = st_c
    else:
        print(f"note: C backend unavailable ({st_c.c_error}); numpy only")
    fp32 = os.path.join(os.path.dirname(os.path.abspath(NPZ)), "mano2somax_regressor_fp32.mlpackage")
    if os.path.isdir(fp32):
        engines["coreml-fp32 heads + " + st_c.backend] = SomaStageNumpy(NPZ, regressor_mlpackage=fp32, compute_units="CPU_ONLY")
    ok = True
    if st_c.backend == "c":  # C and numpy geometry must agree to round-off on identical params
        hp, b, go = rand_inputs(3, B=2)
        p = st_c.regress(hp.numpy(), b.numpy(), go.numpy())
        d = max(np.abs(a - b_).max() for a, b_ in zip(st_c.pose(p), st_c.pose_numpy(p)))
        passed = d < 1e-7
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} C vs numpy geometry on identical params: max |d| = {d*1000:.2e} mm")
    for name, st in engines.items():
        worst = 0.0
        for seed in range(1, 31):
            hp, b, go = rand_inputs(seed, B=1 if seed % 3 else 2)
            v_ref, j_ref = ref(hp, b, go)
            v, j = st.forward(hp.numpy(), b.numpy(), go.numpy())
            worst = max(worst, np.abs(v - v_ref).max(), np.abs(j - j_ref).max())
        hp, b, go = rand_inputs(7, B=1)
        a = (hp.numpy(), b.numpy(), go.numpy())
        for _ in range(10):
            st.forward(*a)
        t0 = time.perf_counter()
        for _ in range(100):
            st.forward(*a)
        ms = (time.perf_counter() - t0) / 100 * 1e3
        passed = worst < 2e-4  # 0.2 mm; measured ~0.05 mm (fp32 reference noise)
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {name}: max |dv| vs torch/warp = {worst*1000:.4f} mm | {ms:.2f} ms/hand")
    # fit metric: barycentric transfer vs torch interpolator (needs the MANO pkl)
    st = engines["numpy"]
    pkl = os.path.expanduser("~/.cache/fasthamer/mano/MANO_RIGHT.pkl")
    if st.bary is not None and os.path.isfile(pkl):
        ml = SOMAHandLayer(hand_type="right", device="cpu", identity_model_type="mano",
                           identity_model_kwargs={"model_path": pkl}, lod="mid", correctives_model_path=None)
        mv = np.random.default_rng(0).normal(size=(2, 778, 3)) * 0.05
        with torch.no_grad():
            ti = ml.identity_model._to_soma_interp(torch.from_numpy(mv.astype(np.float32))).numpy()
        err = np.abs(st.mano_to_soma(mv) - ti).max()
        passed = err < 1e-5
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} MANO->SOMA barycentric transfer: max |d| = {err*1000:.5f} mm")
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
