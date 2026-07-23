"""Unit tests for the temporal handedness stabilizer.

Pure logic — no model or camera needed:

    python fasthamer/tests/test_stabilize.py
"""
import sys

from fasthamer import HandednessStabilizer

R, L = 1, 0
FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {got}, want {want}")
    if not ok:
        FAILURES.append(name)


def test_flicker_is_suppressed():
    """A stationary hand keeps the label it first appeared with, even when the
    detector flips its mind on later frames."""
    s = HandednessStabilizer()
    box = [100, 100, 200, 200]
    got = [s([box], [flip])[0] for flip in (R, L, R, L, L)]
    check("flicker suppressed", got, [R, R, R, R, R])


def test_new_location_trusts_detector():
    """A hand appearing somewhere with no overlapping track adopts the
    detector's current call."""
    s = HandednessStabilizer()
    s([[0, 0, 100, 100]], [R])
    got = s([[500, 500, 600, 600]], [L])
    check("new location trusts detector", got, [L])


def test_two_hands_one_to_one():
    """Overlapping boxes must not both claim the same track: each keeps its
    own locked label."""
    s = HandednessStabilizer()
    a, b = [0, 0, 100, 100], [50, 0, 150, 100]     # 33% IoU with each other
    s([a, b], [R, L])
    got = s([a, b], [L, R])                         # detector swaps both
    check("two hands stay one-to-one", got, [R, L])


def test_track_expires_after_ttl():
    """Once a track passes its TTL, a hand at that location is a new hand and
    the detector is trusted again."""
    s = HandednessStabilizer(ttl=3)
    box = [100, 100, 200, 200]
    s([box], [R])
    for _ in range(4):        # 4 empty frames > ttl=3
        s([], [])
    got = s([box], [L])
    check("track expires after ttl", got, [L])


def test_track_survives_within_ttl():
    s = HandednessStabilizer(ttl=5)
    box = [100, 100, 200, 200]
    s([box], [R])
    for _ in range(3):        # 3 empty frames < ttl=5
        s([], [])
    got = s([box], [L])
    check("track survives within ttl", got, [R])


def test_low_iou_is_a_new_hand():
    """Below the IoU threshold the box is not the same hand."""
    s = HandednessStabilizer(iou_match=0.5)
    s([[0, 0, 100, 100]], [R])
    got = s([[80, 80, 180, 180]], [L])   # ~2% IoU
    check("low IoU is a new hand", got, [L])


def test_moving_hand_keeps_label():
    """A hand drifting across the frame stays locked as long as consecutive
    boxes overlap enough."""
    s = HandednessStabilizer(iou_match=0.3)
    got = []
    for i in range(6):        # slide 20px/frame; consecutive IoU ~0.66
        x = i * 20
        got.append(s([[x, 0, x + 100, 100]], [R if i == 0 else L])[0])
    check("moving hand keeps label", got, [R] * 6)


def test_reset_clears_tracks():
    s = HandednessStabilizer()
    box = [100, 100, 200, 200]
    s([box], [R])
    s.reset()
    got = s([box], [L])
    check("reset clears tracks", got, [L])


def test_empty_frame_is_safe():
    s = HandednessStabilizer()
    check("empty frame returns empty", s([], []), [])


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("\nALL PASS" if not FAILURES else f"\nFAILURES: {FAILURES}")
    sys.exit(1 if FAILURES else 0)
