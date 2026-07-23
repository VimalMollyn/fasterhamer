"""Integration test: handedness flicker corrupts the reconstructed geometry,
and the stabilizer (running before preprocess_hands) fixes it.

Drives a real HandMesh with a stub detector that reports a fixed box and a
flickering Right/Left label, so the only thing changing between frames is the
handedness. Needs the model bundle:

    FASTHAMER_MODEL_DIR=_DATA/fasthamer_bundle \\
        python fasthamer/tests/test_handedness_geometry.py
"""
import sys

import cv2
import numpy as np

import fasthamer

IMAGE = "example_data/test5.jpg"
FAILURES = []


class FlickerDetector:
    """Stub detector: fixed box, handedness cycling through `labels`."""

    def __init__(self, box, labels):
        self.box = np.asarray(box, dtype=np.float32)
        self.labels = labels
        self.i = 0

    def detect(self, rgb, timestamp_ms=0, swap_handedness=False):
        lab = self.labels[self.i % len(self.labels)]
        self.i += 1
        return [self.box.copy()], [int(lab)]


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def run_frames(hm, rgb, n=4):
    """Returns (is_right per frame, vertices per frame)."""
    rights, verts = [], []
    for _ in range(n):
        res = hm(rgb)
        rights.append(res.hands[0].is_right)
        verts.append(res.hands[0].vertices.copy())
    return rights, verts


def main():
    rgb = cv2.cvtColor(cv2.imread(IMAGE), cv2.COLOR_BGR2RGB)

    # Get a real hand box to make the stub realistic.
    probe = fasthamer.load(mode="image")
    box = probe(rgb).hands[0].bbox
    del probe

    flicker = [1, 0, 1, 0]  # detector flip-flops on a stationary hand

    # --- stabilizer OFF: flicker reaches the crop -> geometry mirrors ---
    off = fasthamer.load(mode="video", stabilize_handedness=False)
    off.detector = FlickerDetector(box, flicker)
    r_off, v_off = run_frames(off, rgb)

    # --- stabilizer ON: label locked on first appearance ---
    on = fasthamer.load(mode="video", stabilize_handedness=True)
    on.detector = FlickerDetector(box, flicker)
    r_on, v_on = run_frames(on, rgb)

    check("without stabilizer, handedness flickers", r_off == [True, False, True, False],
          str(r_off))
    check("with stabilizer, handedness is locked", r_on == [True] * 4, str(r_on))

    # Geometry: consecutive-frame vertex change. The input image and box are
    # identical every frame, so any movement is purely the handedness bug.
    d_off = max(float(np.abs(v_off[i + 1] - v_off[i]).max()) for i in range(3))
    d_on = max(float(np.abs(v_on[i + 1] - v_on[i]).max()) for i in range(3))
    check("flicker moves the mesh (stabilizer off)", d_off > 0.01,
          f"max consecutive vertex delta = {d_off * 1000:.1f} mm")
    check("stabilized mesh is frame-to-frame identical", d_on == 0.0,
          f"max consecutive vertex delta = {d_on * 1000:.3f} mm")

    # The label-only workaround is insufficient: on a flicker frame the mesh is
    # a genuinely different (mirrored-crop) reconstruction, not the same mesh
    # with a different name. Compare the L-frame mesh to the locked R mesh.
    mirror_gap = float(np.abs(v_off[1] - v_on[1]).max())
    check("flicker frame reconstructs different geometry", mirror_gap > 0.01,
          f"delta vs stabilized = {mirror_gap * 1000:.1f} mm")

    # --- force_handedness overrides everything ---
    forced = fasthamer.load(mode="video", force_handedness="left")
    forced.detector = FlickerDetector(box, flicker)
    r_forced, _ = run_frames(forced, rgb)
    check("force_handedness pins the label", r_forced == [False] * 4, str(r_forced))

    # --- image mode ignores the stabilizer (no temporal continuity) ---
    img_mode = fasthamer.load(mode="image", stabilize_handedness=True)
    check("stabilize_handedness is a no-op in image mode",
          img_mode._stabilizer is None)

    print("\nALL PASS" if not FAILURES else f"\nFAILURES: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
