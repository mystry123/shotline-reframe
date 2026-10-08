"""Face tracks per shot, and which of them are real people.

Tracking: ByteTrack (supervision, MIT) over face boxes, reset at every cut,
so identities never leak across shots.

Real-person filter. A face detector also fires on paintings, photos,
ornaments, people on screens and hands. Score alone can't separate them (a
real profile can score lower than a painted face), so each track gets:
- presence: share of the shot it's visible (one-frame hands drop out);
- liveness: real heads drift and their landmarks change (talking, blinking,
  turning); a painting or object is pixel-still for the whole shot;
- prominence: size relative to the largest live face in the shot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warnings

import supervision as sv

# sv.ByteTrack is deprecated in favour of Roboflow's `trackers` package; pinned
# below 0.31 until we evaluate that package.
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

from .faces import FaceDet
from .shots import Shot


@dataclass
class Track:
    shot: int
    track_id: int
    dets: list[FaceDet] = field(default_factory=list)
    liveness: float = 0.0
    presence: float = 0.0
    is_person: bool = False
    reason: str = ""

    @property
    def times(self) -> np.ndarray:
        return np.array([d.time for d in self.dets])

    def median(self, attr: str) -> float:
        return float(np.median([getattr(d, attr) for d in self.dets]))


def _liveness(dets: list[FaceDet]) -> float:
    """How much a face moves and changes shape, in % of face size.
    ~0 for paintings/objects (only compression noise); real people are higher."""
    if len(dets) < 3:
        return 0.0
    size = np.median([d.h for d in dets])
    centers = np.array([[d.cx, d.cy] for d in dets])
    # Landmarks relative to the box: changes with expression and head turns.
    shape = np.array([((d.landmarks - [d.x, d.y]) / [d.w, d.h]).ravel() for d in dets])
    drift = np.linalg.norm(np.diff(centers, axis=0), axis=1).mean() / max(size, 1e-6)
    deform = np.abs(np.diff(shape, axis=0)).mean()
    # Box size is left out on purpose: detections of non-faces flicker in size,
    # which would read as "alive". Real people score ~1.7+ on drift + deform,
    # paintings and ornaments under ~0.3 (test clip 001).
    return float(100 * (drift + deform))


def track_shot(shot: Shot, faces_per_frame: list[list[FaceDet]], fps: float) -> list[Track]:
    tracker = sv.ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=int(fps * 2),       # survive 2 s of occlusion
        minimum_matching_threshold=0.8,
        frame_rate=int(round(fps)),
        minimum_consecutive_frames=2,
    )
    tracks: dict[int, Track] = {}
    first = int(np.ceil(shot.start * fps))
    last = min(len(faces_per_frame), int(np.ceil(shot.end * fps)))
    frames_in_shot = max(1, last - first)
    for i in range(first, last):
        dets = faces_per_frame[i]
        if not dets:
            tracker.update_with_detections(sv.Detections.empty())
            continue
        xyxy = np.array([[d.x, d.y, d.x + d.w, d.y + d.h] for d in dets], dtype=np.float32) * 1000
        result = tracker.update_with_detections(
            sv.Detections(xyxy=xyxy, confidence=np.array([d.score for d in dets], dtype=np.float32),
                          class_id=np.zeros(len(dets), dtype=int))
        )
        if len(result) == 0:
            continue
        # The tracker returns its own copy of each box: map it back to our
        # detection by best overlap.
        overlap = sv.box_iou_batch(result.xyxy, xyxy)
        for row, tid in enumerate(result.tracker_id):
            idx = int(np.argmax(overlap[row]))
            if overlap[row, idx] > 0.5:
                tracks.setdefault(int(tid), Track(shot.index, int(tid))).dets.append(dets[idx])

    out = list(tracks.values())
    for t in out:
        t.presence = len(t.dets) / frames_in_shot
        t.liveness = _liveness(t.dets)
    _classify(out)
    return out


# Thresholds tuned on the test set; see tests/.
MIN_PRESENCE = 0.25
MIN_LIVENESS = 0.8
MIN_RELATIVE_SIZE = 0.15


def _classify(tracks: list[Track]) -> None:
    live = [t for t in tracks if t.liveness >= MIN_LIVENESS and t.presence >= MIN_PRESENCE]
    biggest = max((t.median("h") for t in live), default=0.0)
    for t in tracks:
        if t.presence < MIN_PRESENCE:
            t.reason = f"brief ({t.presence:.0%} of shot)"
        elif t.liveness < MIN_LIVENESS:
            t.reason = f"still ({t.liveness:.2f}): painting, photo or object"
        elif biggest and t.median("h") < MIN_RELATIVE_SIZE * biggest:
            t.reason = "tiny next to the main faces"
        else:
            t.is_person = True
            t.reason = "person"
