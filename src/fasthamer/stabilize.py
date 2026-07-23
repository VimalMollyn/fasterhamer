"""Temporal handedness stabilization for video mode.

Per-frame detectors decide Right/Left from a single frame's landmarks, and on
hard footage (foreshortened, self-occluding hands — e.g. wrist/egocentric
cameras) that label flickers frame to frame. The label is not cosmetic in
HaMeR: "left" hands are mirror-flipped before the network and un-mirrored
after, so a flickered label produces mirror-wrong *geometry* on that frame.
The stabilizer therefore runs before preprocessing, not on the output.
"""
from typing import Dict, List


class HandednessStabilizer:
    """Lock each hand's Right/Left label across video frames by IoU tracking.

    Feed it the detector's per-frame (boxes, is_right); it returns is_right
    with flicker removed: a box that stays spatially continuous keeps the label
    it first appeared with (the detector is trusted only when a hand first
    appears). Tracks unseen for `ttl` frames are dropped so a genuinely new
    hand at an old location can re-acquire.
    """

    def __init__(self, iou_match: float = 0.3, ttl: int = 10):
        self.iou_match = float(iou_match)
        self.ttl = int(ttl)
        self.tracks: List[Dict] = []   # each: {"box", "label", "seen"}
        self._frame = -1

    def reset(self) -> None:
        """Forget all tracks (call when the video sequence restarts)."""
        self.tracks.clear()
        self._frame = -1

    @staticmethod
    def _iou(a, b) -> float:
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / union if union > 0 else 0.0

    def __call__(self, boxes, is_right) -> List[int]:
        """boxes: list of (4,) xyxy; is_right: list of 0/1.
        Returns the stabilized is_right list."""
        self._frame += 1
        # One-to-one greedy matching, best IoU first: with overlapping hands,
        # two boxes must not both claim the same track.
        cand = []
        for bi, box in enumerate(boxes):
            for ti, t in enumerate(self.tracks):
                iou = self._iou(box, t["box"])
                if iou >= self.iou_match:
                    cand.append((iou, bi, ti))
        cand.sort(reverse=True)
        match, used = {}, set()
        for _iou, bi, ti in cand:
            if bi not in match and ti not in used:
                match[bi] = ti
                used.add(ti)

        out = []
        for bi, box in enumerate(boxes):
            if bi in match:  # spatially continuous -> keep the locked label
                t = self.tracks[match[bi]]
                t["box"], t["seen"] = box, self._frame
                out.append(t["label"])
            else:            # new location -> trust the detector once
                self.tracks.append({"box": box, "label": int(is_right[bi]),
                                    "seen": self._frame})
                out.append(int(is_right[bi]))
        self.tracks = [t for t in self.tracks if self._frame - t["seen"] <= self.ttl]
        return out
