# fasthamer

Realtime [HaMeR](https://github.com/geopavlakos/hamer) 3D hand mesh recovery on
the Apple Neural Engine. Give it an image, get MANO parameters, 3D joints,
camera parameters, and 2D joints per hand — a MediaPipe-Hands-style API, but
with a full 3D hand mesh behind it.

- **Fast**: whole model (ViT-H backbone + MANO head + MANO mesh) runs as one
  CoreML program on the ANE — ~30 FPS end-to-end with two hands at 640px on a
  fanless M4 MacBook Air, torch-free at runtime. On Linux/Windows the same
  model runs on PyTorch/CUDA (150 hands/s batched on a TITAN V).
- **Simple**: `fasthamer.load()` → `result = hands(rgb)` → done.
- **Complete outputs**: MANO `global_orient` / `hand_pose` / `betas`
  (rotation matrices *and* axis-angle), 778-vertex mesh, 21 3D joints,
  camera intrinsics + per-hand translation, projected 2D joints.
- **Batteries included**: hand detection via
  [fasthands](https://github.com/VimalMollyn/Mediapipe-Hands-PyTorch-CoreML)
  (CoreML/ANE, ~1 ms) and a tiny C mesh rasterizer for overlays (~10x faster
  than pyrender, no GPU/OpenGL).

The CoreML/Neural Engine path requires macOS on Apple Silicon. Everywhere else
(Linux, Windows, Intel Macs) fasthamer runs the same model on **PyTorch**, with
CUDA when available — see [Linux / CUDA](#linux--cuda-pytorch-backend).

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
fasthamer-webcam --mirror     # live demo: 3D mesh overlay from your webcam
```

`fasthamer-setup` runs once: it asks you to confirm you've accepted the MANO
license, then downloads the prebuilt CoreML model bundle (~470 MB) into
`~/.cache/fasthamer`. If you skip this step, the same prompt runs on your
first `fasthamer.load()`.

The **first** `fasthamer.load()` also compiles the model for your device
(one-time, typically 10-30 s) and caches the compiled `.mlmodelc`; every load
after that takes a couple of seconds.

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

## Linux / CUDA (PyTorch backend)

Off macOS, `fasthamer.load()` picks the PyTorch backend (`backend="torch"`)
and Google's MediaPipe Tasks hand detector (`detector="mediapipe"`)
automatically. It is the same network (ViT-H + MANO head + MANO skinning),
re-implemented in plain torch (no `hamer` / `smplx` / `timm` / `einops`
dependencies) and loaded from the official HaMeR checkpoint.

```bash
pip install torch            # CUDA build from https://pytorch.org if you have a GPU
pip install "fasthamer[torch]"
# one-time: build the torch bundle from the official checkpoint
# (hamer.ckpt from https://github.com/geopavlakos/hamer, fetch_demo_data.sh
#  -> _DATA/hamer_ckpts/checkpoints/hamer.ckpt; MANO license applies)
fasthamer-convert-torch --ckpt /path/to/hamer.ckpt      # -> ~/.cache/fasthamer/torch-v1 (1.3 GB, fp16)
```

```python
hands = fasthamer.load()                     # == backend="torch", detector="mediapipe" on Linux
hands = fasthamer.load(backend="torch", device="cuda:1", dtype="float32")   # explicit
result = hands(rgb)                          # same API and outputs as on macOS
```

Options (all `fasthamer.load()` arguments): `backend="auto"|"coreml"|"torch"`,
`device="auto"|"cuda"|"cuda:N"|"cpu"|"mps"`, `dtype="auto"|"float16"|"bfloat16"|"float32"`
(auto = float16 on GPUs with tensor cores, float32 otherwise; the MANO head and
skinning always run in float32), `input_size=(H, W)` (default `(256, 192)`, the
reference resolution). `model_dir=` may also point at a `hamer_torch.pt` or
straight at `hamer.ckpt`; `FASTHAMER_TORCH_MODEL_DIR` / `FASTHAMER_HAMER_CKPT`
do the same from the environment. `fasthamer-webcam --backend torch` works too.

**Bring your own detections.** If you already have hand boxes (e.g. from
MediaPipe landmarks you recorded), skip the detector:

```python
result = hands(rgb, boxes=[xyxy], is_right=[1])          # one image
results = hands.reconstruct_many(frames, boxes_per_frame, is_right_per_frame)  # offline batch
```

`reconstruct_many` runs every hand of every frame through the network in one
batched pass — the fast way to process recordings on a GPU.

**Accuracy vs the reference torch HaMeR** (same crops, official checkpoint):
float32 max vertex error 0.3 mm, float16 backbone ~1 mm (fp16 noise, like the
Neural Engine build). **Speed** on an NVIDIA TITAN V (Volta, 12 GB), 256×192
input: float16 ≈ 7 ms/hand batched (150 hands/s), ≈ 45-60 ms for a single
hand call (launch-bound); float32 ≈ 30 ms/hand batched. Peak GPU memory ≈ 3 GB.
On a Pascal-class GPU (no tensor cores) float32 is the faster choice, which
`dtype="auto"` picks for you. CPU works but is slow (seconds per hand).

The MANO mesh overlay (`hands.render`) compiles its tiny C rasterizer with the
system `cc` on Linux as well.

## Video mode

`mode="video"` enables temporal hand tracking in the detector (faster and
smoother than per-frame detection) — feed frames sequentially:

```python
hands = fasthamer.load(mode="video")
while True:
    ok, frame = cap.read()
    result = hands(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

## Webcam demo

`fasthamer-webcam` (installed with the package) opens your webcam and overlays
the reconstructed meshes in realtime:

```bash
fasthamer-webcam --mirror              # selfie view
fasthamer-webcam --skeleton            # 2D joints instead of the mesh
fasthamer-webcam --camera 0            # pick a camera (auto-probed by default)
fasthamer-webcam --stabilize           # lock Right/Left per hand across frames
fasthamer-webcam --camera clip.mp4     # run on a video file instead of a camera
fasthamer-webcam --record out.mp4      # also save the annotated output
fasthamer-webcam --no-display --max-frames 120   # headless benchmark, prints FPS
```

Keys: `q` / ESC quit, `m` toggle mesh, `s` toggle skeleton. It uses the same
low-latency loop as the threaded example below (freshest-frame capture thread
+ worker/display split), so the FPS shown is the real end-to-end rate — it is
capped by your camera's frame rate, and only drops below that when the model
is the bottleneck (hold your hands in view when benchmarking; with no hands
only the ~1 ms detector runs). See `fasthamer-webcam --help` for the full
option list (`--max-hands`, `--alpha`, `--force-handedness`,
`--fasthands-detector`, `--compute-units`, `--width/--height`, ...).

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
    detector="fasthands",     # detection stack: "fasthands" or "mediapipe"
    fasthands_detector=None,  # model inside fasthands: "whim" or "mediapipe"
    model_dir=None,           # local bundle dir (skips download/license check)
    rescale_factor=2.0,       # hand-box padding before cropping
    swap_handedness=False,    # if handedness looks inverted (mirrored inputs)
    stabilize_handedness=False,  # video: lock R/L per hand across frames
    handedness_iou=0.3,       # IoU to treat a detection as the same hand
    handedness_ttl=10,        # drop a track after N unseen frames
    handedness_switch_frames=5,  # switch only after N consecutive disagreements
    force_handedness=None,    # "right" / "left" to pin it outright
    compute_units="CPU_AND_NE",
    backend="auto",           # "coreml" (macOS/ANE) or "torch" (CUDA/CPU); auto by platform
    device="auto",            # torch backend: "cuda", "cuda:1", "cpu", "mps"
    dtype="auto",             # torch backend: float16 on tensor-core GPUs, else float32
    input_size=None,          # torch backend: (H, W) backbone input, default (256, 192)
)
```

### Choosing a hand detector

Two independent choices, and confusingly both offer something called
"mediapipe" — they are not the same thing:

- **`detector`** picks the detection *stack*: `"fasthands"` (default on
  macOS) runs MediaPipe Hands ported to CoreML on the Neural Engine, ~1 ms/frame;
  `"mediapipe"` (default elsewhere) runs Google's official MediaPipe **Tasks**
  API on TFLite/XNNPACK/GPU (needs `pip install "fasthamer[mediapipe]"`, and
  exposes `det_conf` / `track_conf`). `"auto"` picks by platform.
- **`fasthands_detector`** picks the detector model *inside* fasthands
  (requires `fasthands >= 0.4.0`): `"whim"` — the WHIM-fine-tuned
  full-hand-box detector, steadier than the palm detector; `"mediapipe"` —
  the original palm detector. Left as `None`, fasthands' own default (`whim`)
  applies.

```python
hands = fasthamer.load(mode="video", fasthands_detector="mediapipe")
```

The two detector models frame hands differently — on the bundled test image
the boxes are `[408,507,583,921]` (whim) vs `[414,518,584,906]` (palm) — and
that box drives the crop fed to HaMeR, so the choice affects the reconstructed
geometry, not just detection recall. Worth trying both on unusual footage
(egocentric/wrist cameras, heavy foreshortening).

### Handedness stability (video)

Detectors decide Right/Left from a single frame, so on hard footage —
foreshortened, self-occluding hands, e.g. a wrist/egocentric camera — the label
can flicker frame to frame. **This is not cosmetic:** HaMeR is a right-hand
model, so "left" hands are mirror-flipped before the network and un-mirrored
after. A flickered label produces mirror-wrong *geometry* on that frame (we
measure ~200 mm of vertex jump on an otherwise identical frame), which is why
relabeling the output afterwards is not enough.

`stabilize_handedness=True` fixes it upstream of the crop: each hand gets a
persistent IoU-matched track, and a spatially continuous hand keeps the label it
first appeared with. Tracks expire after `handedness_ttl` unseen frames so a
genuinely new hand can re-acquire.

The lock is *hysteretic*, not permanent: a tracked hand only switches its label
once the detector disagrees for `handedness_switch_frames` **consecutive**
frames (default 5). A single agreeing frame resets the streak, so alternating
flicker never accumulates into a switch — but a sustained correction still
wins, so a genuinely misidentified hand recovers instead of staying wrong for
the life of the track. Set `handedness_switch_frames=0` for a permanent lock.

```python
hands = fasthamer.load(mode="video", stabilize_handedness=True)
...
hands.reset()      # clear tracks when the video sequence restarts
```

For rigs where handedness is known and fixed (single-hand egocentric mounts),
`force_handedness="right"` pins every detection outright — a stronger
constraint that takes precedence over the tracker. Both are orthogonal to
`swap_handedness`, which is a global flip applied by the detector.

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
