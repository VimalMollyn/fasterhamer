"""Low-latency realtime webcam demo for fasthamer.

Two things keep this smoother than the simple `webcam_demo.py`:

  * ThreadedCamera — a background thread continuously drains the capture buffer
    and always serves the *freshest* frame, so you never process stale buffered
    frames (cv2.VideoCapture.read() otherwise returns the next queued frame,
    which lags further behind the longer inference takes).
  * A worker/display split — one worker thread does capture + inference +
    overlay on its own frame and publishes the finished annotated frame; the
    main thread just displays the latest finished frame and handles the
    keyboard. The window stays responsive, and because the mesh is drawn on the
    exact frame it was computed for, it never drifts out of alignment.

    python webcam_threaded_demo.py                 # mesh overlay, mirrored view
    python webcam_threaded_demo.py --skeleton      # 2D joints instead of mesh
    python webcam_threaded_demo.py --camera-id 0
    python webcam_threaded_demo.py --no-display --max-frames 120   # benchmark

Press 'q' or ESC to quit.
"""
import argparse
import threading
import time

import cv2

import fasthamer


class ThreadedCamera:
    """Background-thread webcam reader that always serves the freshest frame."""

    def __init__(self, cam_id, width, height):
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.ok = self.cap.isOpened()
        if self.ok:
            self.ok, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def isOpened(self):
        return self.ok

    def release(self):
        self.stopped = True
        try:
            self.thread.join(timeout=0.5)
        except Exception:
            pass
        self.cap.release()


def put_label(frame, text):
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=1,
                    help="cv2 camera index (built-in FaceTime cam is often 1)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--skeleton", action="store_true", help="draw 2D joints, no mesh")
    ap.add_argument("--no-flip", action="store_true", help="do not mirror the view")
    ap.add_argument("--stabilize", action="store_true",
                    help="lock each hand's Right/Left label across frames "
                         "(stops handedness flicker mirroring the mesh)")
    ap.add_argument("--force-handedness", default=None, choices=["right", "left"],
                    help="pin handedness outright (e.g. single-hand egocentric rigs)")
    ap.add_argument("--fasthands-detector", default=None, choices=["whim", "mediapipe"],
                    help="fasthands detector model: whim (full-hand box, default) "
                         "or mediapipe (original palm detector); needs fasthands>=0.4")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--no-display", action="store_true",
                    help="headless: don't open a window (for benchmarking)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N processed frames (0 = run until quit)")
    args = ap.parse_args()

    hands = fasthamer.load(mode="video", max_hands=args.max_hands,
                           model_dir=args.model_dir,
                           stabilize_handedness=args.stabilize,
                           force_handedness=args.force_handedness,
                           fasthands_detector=args.fasthands_detector)

    cap = ThreadedCamera(args.camera_id, args.width, args.height)
    if not cap.isOpened():
        raise SystemExit("could not open webcam — check camera permissions "
                         "(System Settings > Privacy & Security > Camera) "
                         "or try another --camera-id")

    # The worker thread owns all inference/rendering and publishes finished
    # frames here; the main thread only reads `state["frame"]` to display it.
    state = {"frame": None, "stop": False, "count": 0}
    lock = threading.Lock()

    def worker():
        fps = None
        while not state["stop"]:
            ok, frame = cap.read()
            if not ok:
                continue
            if not args.no_flip:
                frame = cv2.flip(frame, 1)
            t = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands(rgb)
            if args.skeleton:
                out = fasthamer.draw_landmarks(frame, result, bgr=True)
            else:
                out = cv2.cvtColor(hands.render(rgb, result), cv2.COLOR_RGB2BGR)
            dt = time.time() - t
            inst = 1.0 / max(dt, 1e-6)
            fps = inst if fps is None else 0.4 * inst + 0.6 * fps
            put_label(out, f"{fps:4.1f} infer-FPS | hands: {len(result)}")
            with lock:
                state["frame"] = out
                state["count"] += 1

    wt = threading.Thread(target=worker, daemon=True)
    wt.start()

    win = "fasthamer realtime (q/ESC to quit)"
    t_start = time.time()
    try:
        while True:
            with lock:
                shown = state["frame"]
                count = state["count"]
            if shown is not None and not args.no_display:
                cv2.imshow(win, shown)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            elif args.no_display:
                time.sleep(0.005)
            if args.max_frames and count >= args.max_frames:
                break
    finally:
        state["stop"] = True
        wt.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()

    if args.no_display:
        elapsed = time.time() - t_start
        n = state["count"]
        print(f"processed {n} frames in {elapsed:.1f}s = {n / max(elapsed, 1e-6):.1f} FPS")


if __name__ == "__main__":
    main()
