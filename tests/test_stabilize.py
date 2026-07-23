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
    """A stationary hand keeps the label it first appeared with when the
    detector only flickers briefly."""
    s = HandednessStabilizer(switch_frames=5)
    box = [100, 100, 200, 200]
    got = [s([box], [flip])[0] for flip in (R, L, R, L, L)]
    check("flicker suppressed", got, [R, R, R, R, R])


def test_sustained_disagreement_switches():
    """N consecutive disagreeing frames flip the locked label."""
    s = HandednessStabilizer(switch_frames=3)
    box = [100, 100, 200, 200]
    got = [s([box], [lab])[0] for lab in (R, L, L, L, L)]
    #                        frame:       0  1  2  3(switch) 4
    check("switches after N consecutive", got, [R, R, R, L, L])


def test_disagreement_streak_must_be_consecutive():
    """An agreeing frame resets the streak, so alternating flicker never
    accumulates into a switch."""
    s = HandednessStabilizer(switch_frames=3)
    box = [100, 100, 200, 200]
    got = [s([box], [lab])[0] for lab in (R, L, L, R, L, L, R, L, L)]
    check("interrupted streak never switches", got, [R] * 9)


def test_long_alternating_flicker_never_switches():
    """The pathological case: 40 frames of R/L flicker must stay locked."""
    s = HandednessStabilizer(switch_frames=3)
    box = [100, 100, 200, 200]
    got = [s([box], [R if i % 2 == 0 else L])[0] for i in range(40)]
    check("alternating flicker never switches", set(got), {R})


def test_switch_frames_zero_is_a_hard_lock():
    """switch_frames=0 restores the permanent lock."""
    s = HandednessStabilizer(switch_frames=0)
    box = [100, 100, 200, 200]
    got = [s([box], [lab])[0] for lab in (R, L, L, L, L, L, L)]
    check("switch_frames=0 never switches", got, [R] * 7)


def test_missed_frame_breaks_the_streak():
    """A frame where the track isn't detected resets its disagreement count."""
    s = HandednessStabilizer(switch_frames=3, ttl=10)
    box = [100, 100, 200, 200]
    s([box], [R])
    s([box], [L])          # disagree 1
    s([box], [L])          # disagree 2
    s([], [])              # missed -> streak reset
    got = [s([box], [L])[0], s([box], [L])[0]]   # only 2 in a row again
    check("missed frame breaks streak", got, [R, R])


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


def test_moving_hand_stays_on_its_track():
    """A hand drifting across the frame stays matched to its track as long as
    consecutive boxes overlap enough. switch_frames=0 isolates the spatial
    tracking from the hysteresis."""
    s = HandednessStabilizer(iou_match=0.3, switch_frames=0)
    got = []
    for i in range(6):        # slide 20px/frame; consecutive IoU ~0.66
        x = i * 20
        got.append(s([[x, 0, x + 100, 100]], [R if i == 0 else L])[0])
    check("moving hand stays on its track", got, [R] * 6)


def test_moving_hand_still_switches_on_sustained_disagreement():
    """Motion and hysteresis compose: a drifting, continuously-tracked hand
    still adopts a sustained new label."""
    s = HandednessStabilizer(iou_match=0.3, switch_frames=3)
    got = []
    for i in range(6):
        x = i * 20
        got.append(s([[x, 0, x + 100, 100]], [R if i == 0 else L])[0])
    #     frame:  0  1  2  3(switch) 4  5
    check("moving hand switches on sustained disagreement", got,
          [R, R, R, L, L, L])


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
