# fasthamer

Realtime [HaMeR](https://github.com/geopavlakos/hamer) 3D hand mesh recovery on
the Apple Neural Engine. Give it an image, get MANO parameters, 3D joints,
camera parameters, and 2D joints per hand — a MediaPipe-Hands-style API, but
with a full 3D hand mesh behind it.

- **Fast**: whole model (ViT-H backbone + MANO head + MANO mesh) runs as one
  CoreML program on the ANE — ~30 FPS end-to-end with two hands at 640px on a
  fanless M4 MacBook Air, torch-free at runtime.
- **Simple**: `fasthamer.load()` → `result = hands(rgb)` → done.
- **Complete outputs**: MANO `global_orient` / `hand_pose` / `betas`
  (rotation matrices *and* axis-angle), 778-vertex mesh, 21 3D joints,
  camera intrinsics + per-hand translation, projected 2D joints.
- **Batteries included**: hand detection via
  [fasthands](https://github.com/VimalMollyn/Mediapipe-Hands-PyTorch-CoreML)
  (CoreML/ANE, ~1 ms) and a tiny C mesh rasterizer for overlays (~10x faster
  than pyrender, no GPU/OpenGL).

Requires macOS on Apple Silicon.

## Install

> **MANO license required.** fasthamer is built on the
> [MANO](https://mano.is.tue.mpg.de) hand model, which is free for
> non-commercial research but license-gated. Before installing, create an
> account at https://mano.is.tue.mpg.de and **sign/accept the MANO license**
> there. The CoreML model bundle has MANO-derived data baked into its weights,
> so by using fasthamer you agree to use it only under the terms of that
> license.

```bash
pip install fasthamer
fasthamer-setup
```

`fasthamer-setup` runs once: it asks you to confirm you've accepted the MANO
license, then downloads the prebuilt CoreML model bundle (~470 MB) into
`~/.cache/fasthamer`. If you skip this step, the same prompt runs on your
first `fasthamer.load()`.

The **first** `fasthamer.load()` also compiles the model for your device
(~30 s, one-time) and caches the compiled `.mlmodelc`; every load after that
takes a couple of seconds.

Non-interactive environments (CI, scripts): set
`FASTHAMER_ACCEPT_MANO_LICENSE=1` to acknowledge the license, e.g.
`FASTHAMER_ACCEPT_MANO_LICENSE=1 fasthamer-setup`.

## Quickstart

```python
import cv2
import fasthamer

hands = fasthamer.load()                       # mode="image" by default
rgb = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
result = hands(rgb)

for hand in result:
    hand.is_right          # bool (left hands: see note below)
    hand.keypoints_2d      # (21, 2)  pixels in the input image
    hand.keypoints_3d      # (21, 3)  meters, hand-centered
    hand.keypoints_camera  # (21, 3)  meters, camera frame (= keypoints_3d + cam_t)
    hand.vertices          # (778, 3) MANO mesh, hand-centered
    hand.cam_t             # (3,)     hand -> camera translation
    hand.global_orient     # (3, 3)   MANO global orientation (rotation matrix)
    hand.hand_pose         # (15, 3, 3) MANO pose  (also .hand_pose_aa / .global_orient_aa)
    hand.betas             # (10,)    MANO shape

result.focal, result.cx, result.cy   # pinhole camera: u = f*X/Z + cx
overlay = hands.render(rgb, result)               # mesh overlay (C rasterizer)
overlay = fasthamer.draw_landmarks(overlay, result)  # 2D skeleton
```

## Video mode

`mode="video"` enables temporal hand tracking in the detector (faster and
smoother than per-frame detection) — feed frames sequentially:

```python
hands = fasthamer.load(mode="video")
while True:
    ok, frame = cap.read()
    result = hands(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

Examples:

- [`examples/image_demo.py`](examples/image_demo.py) — single image in, mesh +
  skeleton overlay out.
- [`examples/webcam_demo.py`](examples/webcam_demo.py) — minimal synchronous
  webcam loop (easiest to read).
- [`examples/webcam_threaded_demo.py`](examples/webcam_threaded_demo.py) —
  **low-latency** webcam demo: a background camera thread always serves the
  freshest frame, and a worker/display split keeps the window responsive while
  the mesh stays aligned to the frame it was computed on. Use this one if the
  simple demo feels laggy (`cv2.VideoCapture` otherwise hands you buffered,
  stale frames). Supports `--no-display --max-frames N` for benchmarking.

## Conventions

- Input images are **RGB uint8** (`H×W×3`).
- 3D outputs use a camera-style frame: x right, y down, z forward, meters.
  `hand.vertices` / `hand.keypoints_3d` are hand-centered; add `hand.cam_t`
  (or use the `*_camera` properties) for camera-frame coordinates. Projection
  uses the pinhole camera in the result (`result.project(pts)`).
- Joint order is MANO/OpenPose-style: wrist, then 4 joints each for thumb,
  index, middle, ring, pinky (`fasthamer.HAND_EDGES` has the skeleton).
- **Left hands**: HaMeR mirrors left-hand crops, so MANO parameters always
  describe a *right* MANO hand. `vertices`/`keypoints_3d` are already
  un-mirrored. To re-pose a left hand yourself, apply the params to a
  right-hand MANO model and negate x.
- `hands.faces` gives the (1538, 3) triangle faces (flip winding for left
  hands).

## Options

```python
fasthamer.load(
    mode="image",             # or "video"
    max_hands=2,
    detector="fasthands",     # or "mediapipe" (pip install "fasthamer[mediapipe]")
    model_dir=None,           # local bundle dir (skips download/license check)
    rescale_factor=2.0,       # hand-box padding before cropping
    swap_handedness=False,    # if handedness looks inverted (mirrored inputs)
    compute_units="CPU_AND_NE",
)
```

`FASTHAMER_MODEL_DIR` (env) points at a local model bundle;
`FASTHAMER_ASSETS_URL` overrides where the bundle is downloaded from.

## How it works

MediaPipe-style palm detection (fasthands, ANE) finds hand boxes; each box is
cropped HaMeR-style and run through a single CoreML mlprogram containing the
ViT-H backbone (192×144 input, interpolated position embeddings), the MANO
transformer head, and the MANO mesh — 6-bit palettized weights with the MANO
buffers kept at fp16. Outputs match the reference HaMeR torch pipeline to
<1 mm (fp16 noise floor). The overlay renderer is a small C rasterizer with
supersampled anti-aliasing, compiled once on first use.

## License

Code: MIT. The model weights are derived from the HaMeR checkpoint and the
MANO model; MANO is licensed by the Max Planck Institute for non-commercial
scientific research — you must register at https://mano.is.tue.mpg.de (which
the first-run setup asks you to confirm) and comply with its
[license](https://mano.is.tue.mpg.de/license.html). Cite
[HaMeR](https://arxiv.org/abs/2312.05251) and
[MANO](https://mano.is.tue.mpg.de) in academic work.
