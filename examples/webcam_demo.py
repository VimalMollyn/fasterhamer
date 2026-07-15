"""Realtime webcam demo: 3D hand mesh overlay via fasthamer.

    python webcam_demo.py                 # mesh overlay, mirrored view
    python webcam_demo.py --skeleton      # 2D joints instead of mesh
    python webcam_demo.py --camera-id 0

Press 'q' or ESC to quit.
"""
import argparse
import time

import cv2

import fasthamer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", type=int, default=1,
                    help="cv2 camera index (built-in FaceTime cam is often 1)")
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--skeleton", action="store_true", help="draw 2D joints, no mesh")
    ap.add_argument("--no-flip", action="store_true", help="do not mirror the view")
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    hands = fasthamer.load(mode="video", max_hands=args.max_hands,
                           model_dir=args.model_dir)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise SystemExit("could not open webcam — check camera permissions "
                         "(System Settings > Privacy & Security > Camera) "
                         "or try another --camera-id")

    fps = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_flip:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        result = hands(rgb)
        if args.skeleton:
            out = fasthamer.draw_landmarks(frame, result, bgr=True)
        else:
            out = cv2.cvtColor(hands.render(rgb, result), cv2.COLOR_RGB2BGR)
        dt = time.time() - t0

        inst = 1.0 / max(dt, 1e-6)
        fps = inst if fps is None else 0.4 * inst + 0.6 * fps
        label = f"{fps:4.1f} FPS | hands: {len(result)}"
        cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("fasthamer (q/ESC to quit)", out)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
