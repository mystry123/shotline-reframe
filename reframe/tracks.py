"""Face tracks per shot, and which of them are real people.

Tracking: ByteTrack (supervision, MIT) over face boxes, reset at every cut,
so identities never leak across shots.

Real-person test. A face detector also fires on paintings, photos,
ornaments, lamps and hands, and no single signal separates them at every
analysis quality (detector boxes wander on non-faces, so box motion looks
"alive"; a real profile can score lower than a painted face). Combined:
- presence:  share of the shot it's visible (a one-frame hand drops out);
- body:      a person detector (YOLOX) finds a body the face belongs to
             (ornaments, lamps and patterns have none);
- change:    the image inside the face changes from second to second
             (talking, blinking, turning); a painting or photo doesn't,
             even when it has a painted body.
Very large faces (extreme close-ups, body out of frame) count as people
on change alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warnings

import supervision as sv

# sv.ByteTrack is deprecated in favour of Roboflow's `trackers` package; pinned
# below 0.31 until we evaluate that package.
warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

import cv2

from .bodies import BodyDet
from .decode import Frames
from .faces import FaceDet
from .shots import Shot


@dataclass
class Track:
    shot: int
    track_id: int
    dets: list[FaceDet] = field(default_factory=list)
    liveness: float = 0.0      # image change inside the face, 1 s apart
    body: float = 0.0          # share of checked frames where a body holds this face
    # Body boxes (x0, y0, x1, y1, picture-normalized) that claimed this face as
    # their head: the person's whole extent (torso, arms, hands), for framing.
    body_boxes: list = field(default_factory=list)
    presence: float = 0.0
    is_person: bool = False
    reason: str = ""

    @property
    def times(self) -> np.ndarray:
        return np.array([d.time for d in self.dets])

    def median(self, attr: str) -> float:
        return float(np.median([getattr(d, attr) for d in self.dets]))

    def body_extent(self) -> tuple[float, float, float, float]:
        """Where the person is over the shot (x0, y0, x1, y1): the measured
        body boxes (10th/90th percentiles, so a stray box doesn't stretch it)
        joined with the face; estimated from the face when no body was seen."""
        fx0, fy0 = self.median("x"), self.median("y")
        fw, fh = self.median("w"), self.median("h")
        face = (fx0, fy0, fx0 + fw, fy0 + fh)
        boxes = np.array(getattr(self, "body_boxes", []) or [], dtype=float)
        if len(boxes):
            body = (np.percentile(boxes[:, 0], 10), np.percentile(boxes[:, 1], 10),
                    np.percentile(boxes[:, 2], 90), np.percentile(boxes[:, 3], 90))
        else:  # head, shoulders and upper body, as seen in most talking shots
            cx = fx0 + fw / 2
            body = (cx - 1.6 * fw, fy0 - 0.3 * fh, cx + 1.6 * fw, fy0 + 4.5 * fh)
        return (float(max(0.0, min(face[0], body[0]))), float(max(0.0, min(face[1], body[1]))),
                float(min(1.0, max(face[2], body[2]))), float(min(1.0, max(face[3], body[3]))))


def _face_patch(frames: Frames, d: FaceDet, size: int = 24) -> np.ndarray | None:
    img = frames.data[d.frame]
    H, W = img.shape[:2]
    x0, y0 = max(0, int(d.x * W)), max(0, int(d.y * H))
    x1, y1 = min(W, int((d.x + d.w) * W)), min(H, int((d.y + d.h) * H))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    g = cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-3)


def _image_change(t: "Track", frames: Frames) -> float | None:
    """Mean change of the normalized face image over about 1 second
    (rate-independent). None when the track is too short to tell."""
    lag = max(1, int(round(frames.fps)))
    patches = [_face_patch(frames, d) for d in t.dets]
    pairs = [(patches[i], patches[i + lag]) for i in range(len(patches) - lag)
             if patches[i] is not None and patches[i + lag] is not None
             and abs(t.dets[i + lag].time - t.dets[i].time - 1.0) < 0.5]
    if len(pairs) < 2:
        return None
    return float(np.mean([np.abs(a - b).mean() for a, b in pairs]))


def _head_score(d: FaceDet, b: BodyDet) -> float | None:
    """How well face `d` fits as the head of body `b` (None if it can't be):
    centred horizontally and in the top part of the box, at a plausible
    face-to-body width."""
    rel_x = (d.cx - b.x) / max(b.w, 1e-6)
    rel_y = (d.cy - b.y) / max(b.h, 1e-6)
    ratio = d.w / max(b.w, 1e-6)
    if not (0.1 <= rel_x <= 0.9 and -0.1 <= rel_y <= 0.45 and 0.08 <= ratio <= 0.9):
        return None
    return d.h  # the biggest face that fits is the head; ties are rare


def _body_shares(tracks: list["Track"], bodies: list[list[BodyDet]]) -> dict[int, float]:
    """Per track: share of body-checked frames where a body claims this face
    as its head. Each body claims at most one face, so an object next to a
    person (inside their box) doesn't borrow their body."""
    by_frame: dict[int, list[tuple["Track", FaceDet]]] = {}
    for t in tracks:
        for d in t.dets:
            by_frame.setdefault(d.frame, []).append((t, d))
    checked = {id(t): 0 for t in tracks}
    held = {id(t): 0 for t in tracks}
    for frame, items in by_frame.items():
        if frame >= len(bodies) or not _was_checked(bodies, frame):
            continue
        for t, _ in items:
            checked[id(t)] += 1
        for b in bodies[frame]:
            scored = [(sc, t) for t, d in items if (sc := _head_score(d, b)) is not None]
            if scored:
                owner = max(scored, key=lambda x: x[0])[1]
                held[id(owner)] += 1
                owner.body_boxes.append((b.x, b.y, b.x + b.w, b.y + b.h))
    return {id(t): (held[id(t)] / checked[id(t)] if checked[id(t)] else 0.0) for t in tracks}


_CHECKED: dict[int, set[int]] = {}


def _was_checked(bodies: list[list[BodyDet]], frame: int) -> bool:
    return frame in _CHECKED.get(id(bodies), set())


def mark_checked(bodies: list[list[BodyDet]], frames: set[int]) -> None:
    """Record which frames the body detector ran on (empty lists there mean
    "no body", elsewhere they mean "not checked")."""
    _CHECKED[id(bodies)] = frames


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
    return out


# Thresholds from test clip 001 (all three analysis qualities); see tests/.
MIN_PRESENCE = 0.25
MIN_CHANGE = 0.10            # paintings/photos 0.00-0.07; real faces 0.15+
MIN_BODY = 0.5
CLOSE_UP = 0.35              # face height (share of picture) where the body may be out of frame
MIN_RELATIVE_SIZE = 0.15


def classify(tracks: list[Track], frames: Frames, bodies: list[list[BodyDet]]) -> None:
    shares = _body_shares(tracks, bodies)
    changes: dict[int, float | None] = {}
    for t in tracks:
        changes[id(t)] = _image_change(t, frames)
        t.liveness = changes[id(t)] if changes[id(t)] is not None else -1.0
        t.body = shares[id(t)]

    def changes_enough(t: Track) -> bool:
        c = changes[id(t)]
        return c is None or c >= MIN_CHANGE  # too short to tell: don't reject on this

    candidates = [t for t in tracks if t.presence >= MIN_PRESENCE and changes_enough(t)
                  and (t.body >= MIN_BODY or t.median("h") >= CLOSE_UP)]
    biggest = max((t.median("h") for t in candidates), default=0.0)
    for t in tracks:
        if t.presence < MIN_PRESENCE:
            t.reason = f"brief ({t.presence:.0%} of shot)"
        elif t.body < MIN_BODY and t.median("h") < CLOSE_UP:
            t.reason = f"no body ({t.body:.0%}): an object, not a person"
        elif not changes_enough(t):
            t.reason = f"image doesn't change ({t.liveness:.2f}): painting or photo"
        elif biggest and t.median("h") < MIN_RELATIVE_SIZE * biggest:
            t.reason = "tiny next to the main faces"
        else:
            t.is_person = True
            t.reason = "person"
