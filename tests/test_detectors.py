"""Detector-selection tests: every supported stack/model combination loads and
produces hands, and invalid combinations fail loudly.

    FASTHAMER_MODEL_DIR=_DATA/fasthamer_bundle \\
        python fasthamer/tests/test_detectors.py

The `mediapipe` stack needs the optional extra (pip install "fasthamer[mediapipe]");
`fasthands_detector` needs fasthands >= 0.4.0. Both are skipped if unavailable.
"""
import sys

import cv2

import fasthamer

IMAGE = "example_data/test5.jpg"
FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def skip(name, why):
    print(f"SKIP {name}: {why}")


def main():
    rgb = cv2.cvtColor(cv2.imread(IMAGE), cv2.COLOR_BGR2RGB)
    boxes = {}

    combos = [
        ("fasthands default", {}, True),
        ("fasthands whim", {"fasthands_detector": "whim"}, True),
        ("fasthands mediapipe", {"fasthands_detector": "mediapipe"}, True),
        ("mediapipe tasks", {"detector": "mediapipe"}, False),
    ]
    for name, kw, required in combos:
        try:
            res = fasthamer.load(mode="image", **kw)(rgb)
            boxes[name] = res.hands[0].bbox
            check(name, len(res) == 2, f"{len(res)} hands")
        except (ImportError, ValueError) as e:
            if required:
                check(name, False, f"{type(e).__name__}: {e}")
            else:
                skip(name, str(e)[:60])

    # whim is fasthands' default, so those two must agree exactly...
    if "fasthands default" in boxes and "fasthands whim" in boxes:
        same = (boxes["fasthands default"] == boxes["fasthands whim"]).all()
        check("whim is the fasthands default", same)

    # ...and the palm detector must give a genuinely different crop, which is
    # why the choice matters for the reconstructed geometry.
    if "fasthands whim" in boxes and "fasthands mediapipe" in boxes:
        differs = (boxes["fasthands whim"] != boxes["fasthands mediapipe"]).any()
        check("whim and palm detector differ", differs,
              f"whim={boxes['fasthands whim'].round().tolist()} "
              f"palm={boxes['fasthands mediapipe'].round().tolist()}")

    # fasthands_detector is meaningless for the Tasks stack -> must raise.
    try:
        fasthamer.load(mode="image", detector="mediapipe",
                       fasthands_detector="whim")
        check("rejects fasthands_detector on the wrong stack", False, "no error")
    except ValueError:
        check("rejects fasthands_detector on the wrong stack", True)
    except ImportError:
        skip("rejects fasthands_detector on the wrong stack", "mediapipe extra missing")

    # unknown stack name -> must raise
    try:
        fasthamer.load(mode="image", detector="nope")
        check("rejects unknown detector", False, "no error")
    except ValueError:
        check("rejects unknown detector", True)

    print("\nALL PASS" if not FAILURES else f"\nFAILURES: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
