"""`fasthamer-webcam` — live webcam demo: realtime 3D hand mesh overlay.

    fasthamer-webcam                      # mesh overlay from the default camera
    fasthamer-webcam --mirror             # selfie view (flip horizontally)
    fasthamer-webcam --skeleton           # 2D joints instead of the mesh
    fasthamer-webcam --camera 0           # pick a camera (auto-detected by default)
    fasthamer-webcam --camera clip.mp4    # run on a video file instead
    fasthamer-webcam --record out.mp4     # save the annotated output
    fasthamer-webcam --no-display --max-frames 120   # headless benchmark

Keys while running: q / ESC quit, m toggle mesh, s toggle skeleton.

Latency notes: a background thread keeps draining the capture buffer so
inference always sees the freshest frame (cv2.VideoCapture.read() otherwise
hands you stale queued frames), and a worker thread does inference + overlay
while the main thread only shows the latest finished frame — the window stays
responsive and the mesh is drawn on the exact frame it was computed on.
"""
import argparse
import os
import sys
import threading
import time

import cv2

from .assets import cache_dir
from .rendering import draw_landmarks
from .tracker import load

# Camera auto-detection: on Macs with Continuity Camera or extra USB cameras,
# the low cv2 indices are often virtual/inactive devices that open fine but
# stream black frames (or none at all); the built-in camera may be 1 or 2.
_PROBE_INDICES = (0, 1, 2)
_BLACK_MEAN = 20.0           # mean pixel value below which a frame counts as black
_FIRST_FRAME_TIMEOUT = 1.5   # s to wait for a device to deliver any frame at all
_EXPOSURE_WINDOW = 1.0       # s of extra frames to let auto-exposure ramp up


def _open_capture(index, width, height):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _probe(cap):
    """Classify a freshly opened capture: "video" (delivers non-black frames),
    "black" (delivers frames but they are all black, e.g. an inactive
    Continuity Camera or a capped lens) or "none" (opens but never delivers
    a frame — a device that failed to initialize)."""
    got_frame = False
    deadline = time.monotonic() + _FIRST_FRAME_TIMEOUT
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            if not got_frame:
                got_frame = True
                deadline = time.monotonic() + _EXPOSURE_WINDOW
            if float(frame.mean()) > _BLACK_MEAN:
                return "video"
    return "black" if got_frame else "none"


def _remembered_index_file():
    return os.path.join(cache_dir(), "webcam_index")


def _load_remembered_index():
    try:
        with open(_remembered_index_file()) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _save_remembered_index(index):
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        with open(_remembered_index_file(), "w") as f:
            f.write(f"{index}\n")
    except OSError:
        pass


def _find_camera(width, height):
    """Pick a camera automatically: the index that worked last time first,
    then 0, 1, 2 — the first one that streams video wins."""
    remembered = _load_remembered_index()
    order = ([remembered] if remembered is not None else []) + \
        [i for i in _PROBE_INDICES if i != remembered]
    best = None   # (rank, index): rank 0 = black frames, 1 = no frames at all
    for i in order:
        cap = _open_capture(i, width, height)
        if cap is None:
            continue
        status = _probe(cap)
        if status == "video":
            _save_remembered_index(i)
            return cap, i
        cap.release()
        rank = 0 if status == "black" else 1
        if best is None or rank < best[0]:
            best = (rank, i)
    if best is None:
        raise SystemExit(
            "could not open a webcam — check camera permissions "
            "(macOS: System Settings > Privacy & Security > Camera; Linux: "
            "/dev/video* permissions) or pass --camera N")
    rank, i = best
    print(f"[fasthamer] no camera streamed video; using camera {i} anyway "
          f"({'black frames' if rank == 0 else 'no frames yet'}) — "
          "pass --camera N to pick another", file=sys.stderr)
    cap = _open_capture(i, width, height)
    if cap is None:
        raise SystemExit(f"could not reopen camera {i}")
    return cap, i


def open_source(source, width=0, height=0):
    """Open a video source: None auto-detects a camera (see _find_camera), an
    int is a cv2 camera index, a str is a video/image file path.
    Returns (capture, source)."""
    if source is None:
        return _find_camera(width, height)
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            raise SystemExit(f"could not open video file: {source}")
        return cap, source
    cap = _open_capture(source, width, height)
    if cap is None:
        raise SystemExit(
            f"could not open camera {source} — check camera permissions "
            "(System Settings > Privacy & Security > Camera) or try "
            "another --camera index")
    return cap, source


class SourceOpener(threading.Thread):
    """Opens/probes the source on a background thread so the (slow) camera
    probe overlaps the model load; get() joins and returns (capture, source)."""

    def __init__(self, source, width, height):
        super().__init__(daemon=True)
        self._args = (source, width, height)
        self.result = None
        self.error = None
        self.start()

    def run(self):
        try:
            self.result = open_source(*self._args)
        except BaseException as e:      # SystemExit included: re-raised in get()
            self.error = e

    def get(self):
        self.join()
        if self.error is not None:
            raise self.error
        return self.result


class ThreadedCamera:
    """Background-thread frame reader.

    live=True (camera): keeps draining the capture buffer and serves only the
    freshest frame; frames that arrive while the consumer is busy are dropped.
    live=False (video file): runs in lockstep so every frame is served once.
    """

    def __init__(self, cap, live=True):
        self.cap = cap
        self.live = live
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0            # bumps on every new frame
        self.consumed = 0       # seq of the last frame handed out
        self.stopped = False
        self.ended = False      # source stopped delivering (EOF / camera lost)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            if not self.live:
                with self.cond:
                    while self.consumed < self.seq and not self.stopped:
                        self.cond.wait(0.05)
                if self.stopped:
                    break
            ok, frame = self.cap.read()
            with self.cond:
                if ok and frame is not None:
                    self.frame = frame
                    self.seq += 1
                else:
                    self.ended = True
                self.cond.notify_all()
            if self.ended:
                break

    def read(self, timeout=0.1):
        """Wait up to `timeout` s for a frame newer than the last one served.
        Returns a copy of it, or None (timeout / end of stream)."""
        with self.cond:
            if self.seq <= self.consumed and not (self.ended or self.stopped):
                self.cond.wait(timeout)
            if self.seq <= self.consumed:
                return None
            self.consumed = self.seq
            frame = self.frame.copy()
            self.cond.notify_all()
            return frame

    def release(self):
        self.stopped = True
        with self.cond:
            self.cond.notify_all()
        self.thread.join(timeout=0.5)
        self.cap.release()


def _put_label(frame, text, y=28):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)


def _parse_source(s):
    if s is None:
        return None
    if s.isdigit():
        return int(s)
    if not os.path.exists(s):
        raise SystemExit(f"--camera: not a camera index and no such file: {s}")
    return s


def _make_writer(path, fps, size):
    ext = os.path.splitext(path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if ext == ".avi" else "mp4v"))
    w = cv2.VideoWriter(path, fourcc, fps, size)
    if not w.isOpened():
        raise SystemExit(f"could not open --record file for writing: {path}")
    return w


def build_parser():
    ap = argparse.ArgumentParser(
        prog="fasthamer-webcam",
        description="Live webcam demo: realtime 3D hand mesh overlay "
                    "(HaMeR on the Apple Neural Engine, or PyTorch/CUDA elsewhere). "
                    "Keys: q/ESC quit, m toggle mesh, s toggle skeleton.")
    ap.add_argument("--camera", default=None, metavar="N|PATH",
                    help="cv2 camera index, or a video/image file to run on "
                         "instead of a camera. Default: probe cameras 0, 1, 2 "
                         "and use the first that streams video (on Macs with "
                         "Continuity Camera, 0 is often a black virtual device); "
                         "the working index is remembered for next time")
    ap.add_argument("--mirror", action="store_true",
                    help="selfie view: flip the image horizontally")
    ap.add_argument("--skeleton", action="store_true",
                    help="draw the 2D joint skeleton instead of the mesh")
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--width", type=int, default=640, help="capture width (default 640)")
    ap.add_argument("--height", type=int, default=480, help="capture height (default 480)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="mesh opacity 0-1 (default 1.0)")
    ap.add_argument("--stabilize", action="store_true",
                    help="lock each hand's Right/Left label across frames "
                         "(stops handedness flicker mirroring the mesh)")
    ap.add_argument("--force-handedness", default=None, choices=["right", "left"],
                    help="pin handedness outright (e.g. single-hand egocentric rigs)")
    ap.add_argument("--backend", default="auto", choices=["auto", "coreml", "torch"],
                    help="inference backend: auto (CoreML on macOS, PyTorch elsewhere), "
                         "coreml (Apple Neural Engine) or torch (CUDA/CPU)")
    ap.add_argument("--device", default="auto",
                    help="torch backend device: auto, cuda, cuda:1, cpu, mps (default auto)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                    help="torch backend precision (default auto: float16 on tensor-core GPUs)")
    ap.add_argument("--detector", default="auto", choices=["auto", "fasthands", "mediapipe"],
                    help="detection stack: auto (fasthands on macOS, mediapipe elsewhere), "
                         "fasthands (CoreML/ANE) or mediapipe (Google MediaPipe Tasks; "
                         "needs fasthamer[mediapipe])")
    ap.add_argument("--fasthands-detector", default=None, choices=["whim", "mediapipe"],
                    help="fasthands detector model: whim (full-hand box, default) "
                         "or mediapipe (palm detector); needs fasthands>=0.4")
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    choices=["CPU_AND_NE", "ALL", "CPU_AND_GPU", "CPU_ONLY"])
    ap.add_argument("--model-dir", default=None,
                    help="local model bundle (default: the fasthamer cache); for the torch "
                         "backend this may also be a hamer.ckpt")
    ap.add_argument("--record", default=None, metavar="PATH",
                    help="also write the annotated frames to a video file (.mp4/.avi)")
    ap.add_argument("--no-display", action="store_true",
                    help="headless: don't open a window (benchmarking / --record)")
    ap.add_argument("--max-frames", type=int, default=0, metavar="N",
                    help="stop after N processed frames (0 = run until quit)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = _parse_source(args.camera)

    opener = SourceOpener(source, args.width, args.height)   # overlaps the load
    print("[fasthamer] loading model (on macOS the first run compiles it for "
          "your device, which can take ~30 s)...", flush=True)
    t = time.time()
    hands = load(mode="video", max_hands=args.max_hands,
                 backend=args.backend,
                 detector=args.detector,
                 fasthands_detector=args.fasthands_detector,
                 model_dir=args.model_dir,
                 stabilize_handedness=args.stabilize,
                 force_handedness=args.force_handedness,
                 compute_units=args.compute_units,
                 device=args.device, dtype=args.dtype)
    print(f"[fasthamer] model ready in {time.time() - t:.1f} s", flush=True)

    cap, source = opener.get()
    live = not isinstance(source, str)
    if not live:
        print(f"[fasthamer] reading {source}", flush=True)
    elif args.camera is None:
        print(f"[fasthamer] using camera {source} (auto-detected; pass "
              "--camera N to override)", flush=True)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cam = ThreadedCamera(cap, live=live)

    # Runtime toggles (read by the worker each frame, flipped by the key loop).
    opts = {"mesh": not args.skeleton, "skeleton": args.skeleton}
    # The worker owns inference + overlay and publishes finished frames here;
    # the main thread only reads `state["frame"]` to display it.
    state = {"frame": None, "stop": False, "count": 0, "writer": None}
    lock = threading.Lock()

    def worker():
        fps = None
        t_prev = None
        while not state["stop"]:
            frame = cam.read()
            if frame is None:
                if cam.ended:
                    break
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands(rgb)
            if opts["mesh"]:
                out = cv2.cvtColor(hands.render(rgb, result, alpha=args.alpha),
                                   cv2.COLOR_RGB2BGR)
            else:
                out = frame          # already our own copy (cam.read())
            if opts["skeleton"]:
                out = draw_landmarks(out, result, bgr=True)
            now = time.time()
            if t_prev is not None:
                inst = 1.0 / max(now - t_prev, 1e-6)
                fps = inst if fps is None else 0.3 * inst + 0.7 * fps
            t_prev = now
            _put_label(out, f"{fps:4.1f} FPS | hands: {len(result)}" if fps
                       else f"hands: {len(result)}")
            if args.record:
                if state["writer"] is None:
                    state["writer"] = _make_writer(
                        args.record, src_fps if src_fps > 1 else 30.0,
                        (out.shape[1], out.shape[0]))
                state["writer"].write(out)
            with lock:
                state["frame"] = out
                state["count"] += 1

    wt = threading.Thread(target=worker, daemon=True)
    wt.start()

    win = "fasthamer webcam  (q quit | m mesh | s skeleton)"
    t_start = time.time()
    code = 0
    try:
        while True:
            with lock:
                shown = state["frame"]
                count = state["count"]
            if args.max_frames and count >= args.max_frames:
                break
            if not wt.is_alive():
                if live and cam.ended:
                    print("[fasthamer] camera stopped delivering frames",
                          file=sys.stderr)
                    code = 1
                break
            if args.no_display or shown is None:
                time.sleep(0.005)
                continue
            cv2.imshow(win, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                opts["mesh"] = not opts["mesh"]
            elif key == ord("s"):
                opts["skeleton"] = not opts["skeleton"]
    except KeyboardInterrupt:
        pass
    finally:
        state["stop"] = True
        wt.join(timeout=1.0)
        cam.release()
        if state["writer"] is not None:
            state["writer"].release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    n = state["count"]
    if args.record and n:
        print(f"[fasthamer] wrote {n} frames to {args.record}")
    if args.no_display:
        print(f"[fasthamer] processed {n} frames in {elapsed:.1f} s = "
              f"{n / max(elapsed, 1e-6):.1f} FPS")
    return code


if __name__ == "__main__":
    sys.exit(main())
