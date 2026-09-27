"""PyTorch backend tests (pytest).

    pytest tests/test_torch_backend.py -v

Needs torch and a torch bundle (fasthamer-convert-torch, or FASTHAMER_TORCH_MODEL_DIR /
FASTHAMER_HAMER_CKPT); everything is skipped otherwise. The reference-parity test additionally
needs the `hamer` package importable (HAMER_ROOT env pointing at a hamer checkout with _DATA).
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fasthamer.assets import resolve_torch_model  # noqa: E402
from fasthamer.inference_torch import TorchHamer  # noqa: E402
from fasthamer.preprocess import preprocess_hands  # noqa: E402
from fasthamer.torch_model import BACKBONE_INPUT, HamerTorch  # noqa: E402


def _bundle():
    try:
        return resolve_torch_model(interactive=False)
    except (FileNotFoundError, RuntimeError) as e:
        pytest.skip(f"no torch bundle: {e}")


def _synthetic_hand_image(w=640, h=480, seed=0):
    """A deterministic image with a bright hand-like blob; enough to exercise the network."""
    rng = np.random.default_rng(seed)
    img = (rng.uniform(40, 80, (h, w, 3))).astype(np.uint8)
    import cv2
    cv2.ellipse(img, (w // 2, h // 2), (70, 90), 20, 0, 360, (200, 170, 150), -1)
    for i in range(5):
        cv2.line(img, (w // 2 - 50 + 25 * i, h // 2 - 60), (w // 2 - 60 + 30 * i, h // 2 - 170), (200, 170, 150), 18)
    box = np.array([w // 2 - 110, h // 2 - 190, w // 2 + 110, h // 2 + 100], np.float32)
    return img, box


def test_architecture_matches_checkpoint_keys():
    model = HamerTorch()
    keys = set(model.state_dict().keys())
    assert "backbone.pos_embed" in keys and "mano_head.decpose.weight" in keys and "mano.posedirs" in keys
    assert model.state_dict()["backbone.pos_embed"].shape == (1, 193, 1280)
    assert model.state_dict()["mano_head.init_hand_pose"].shape == (1, 96)


def test_predict_shapes_and_reprojection():
    engine = TorchHamer(_bundle(), device="auto", dtype="auto")
    img, box = _synthetic_hand_image()
    crops = preprocess_hands(img, box[None], np.array([1]), rescale_factor=2.0)
    pred = engine.predict(crops[0])
    assert pred["vertices"].shape == (778, 3) and pred["keypoints3d"].shape == (21, 3)
    assert pred["global_orient"].shape == (3, 3) and pred["hand_pose"].shape == (15, 3, 3) and pred["betas"].shape == (10,)
    assert np.all(np.isfinite(pred["vertices"]))
    # rotation matrices are orthonormal
    R = pred["hand_pose"]
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-3)
    # hand-centred mesh is hand-sized (MANO: ~10 cm palm, ~20 cm span)
    extent = pred["vertices"].max(0) - pred["vertices"].min(0)
    assert 0.08 < extent.max() < 0.30
    # batch == single
    preds = engine.predict_batch(crops * 3)
    assert len(preds) == 3
    assert np.allclose(preds[1]["vertices"], pred["vertices"], atol=2e-3)


def test_end_to_end_with_boxes():
    import fasthamer
    hands = fasthamer.load(backend="torch", detector="mediapipe") if _mediapipe() else None
    if hands is None:
        # detector-free path still needs an engine; build via reconstruct
        hands = fasthamer.HandMesh.__new__(fasthamer.HandMesh)
        pytest.skip("mediapipe not installed; end-to-end path covered by test_predict_shapes_and_reprojection")
    img, box = _synthetic_hand_image()
    res = hands(img, boxes=[box], is_right=[1])
    assert len(res) == 1
    h = res.hands[0]
    assert h.is_right and h.keypoints_2d.shape == (21, 2)
    # projected joints must land inside the padded box
    pad = 0.5 * (box[2] - box[0])
    assert (h.keypoints_2d[:, 0] > box[0] - pad).all() and (h.keypoints_2d[:, 0] < box[2] + pad).all()


def _mediapipe():
    try:
        import mediapipe  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not os.environ.get("HAMER_ROOT"), reason="set HAMER_ROOT to a hamer checkout for the parity test")
def test_parity_with_reference_hamer():
    root = os.environ["HAMER_ROOT"]
    sys.path.insert(0, root)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        from hamer.models import load_hamer, DEFAULT_CHECKPOINT
        ref, _ = load_hamer(DEFAULT_CHECKPOINT)
    except ImportError as e:  # hamer and its deps (smplx, lightning, ...) are not installed
        pytest.skip(f"reference hamer package not importable: {e}")
    finally:
        os.chdir(cwd)
    ref = ref.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    engine = TorchHamer(_bundle(), dtype="float32")
    img, box = _synthetic_hand_image()
    crop = preprocess_hands(img, box[None], np.array([1]), rescale_factor=2.0)[0]
    x = torch.from_numpy(crop.img[None]).to(next(ref.parameters()).device)
    with torch.no_grad():
        out = ref({"img": x})
    pred = engine.predict(crop)
    dv = np.abs(out["pred_vertices"][0].cpu().numpy() - pred["vertices"]).max()
    dc = np.abs(out["pred_cam"][0].cpu().numpy() - pred["cam"]).max()
    assert dv < 1e-3, f"vertex mismatch {dv * 1000:.2f} mm"
    assert dc < 1e-3
