"""Run fasthamer on a single image and save mesh + skeleton overlays.

    python image_demo.py photo.jpg
    python image_demo.py photo.jpg --out out.jpg
"""
import argparse

import cv2
import numpy as np

import fasthamer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="fasthamer_out.jpg")
    ap.add_argument("--model-dir", default=None,
                    help="local model bundle (default: auto-download to cache)")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    hands = fasthamer.load(mode="image", model_dir=args.model_dir)
    result = hands(rgb)

    print(f"found {len(result)} hand(s); camera: focal={result.focal:.1f}px "
          f"cx={result.cx:.1f} cy={result.cy:.1f}")
    for i, hand in enumerate(result):
        wrist = hand.keypoints_camera[0]
        print(f"  hand {i}: {'right' if hand.is_right else 'left'}"
              f" | wrist at ({wrist[0]:+.3f}, {wrist[1]:+.3f}, {wrist[2]:.3f}) m"
              f" | betas[:3]={np.round(hand.betas[:3], 3)}"
              f" | global_orient_aa={np.round(hand.global_orient_aa, 3)}")

    overlay = hands.render(rgb, result)
    overlay = fasthamer.draw_landmarks(overlay, result)
    cv2.imwrite(args.out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
