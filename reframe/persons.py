"""People from bodies as well as faces.

Faces are the best signal for framing, but a face detector loses a person
who looks down, turns away, is backlit, blurred by motion or simply small.
Bodies (YOLOX) survive all of those. Per shot:

1. Body tracks: ByteTrack over the body boxes.
2. Links: a face track belongs to the body track that holds its faces as
   the head (_head_score) most often.
3. One person per body track: its face tracks' detections joined, and the
   gaps (no face for more than GAP_S) filled with a head estimated from the
   body box, at the face's usual place on that body (learnt from the frames
   where both were seen). Estimated heads are marked `estimated`.
4. A body track with no face at all is still a person when it is there for
   most of the shot, big enough, and its image changes over time (a poster
   or a mannequin doesn't).

Face tracks with no body (close-ups) are kept as they were.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import supervision as sv

from .bodies import BodyDet
from .decode import Frames
from .faces import FaceDet
from .shots import Shot
from .tracks import MIN_CHANGE, Track, _head_score

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

GAP_S = 0.6                 # no face for longer than this: estimate the head from the body
MIN_LINKS = 2               # face/body co-occurrences needed to join them
BODY_PRESENCE = 0.5         # a faceless body must be in this share of the shot's checked frames...
BODY_MIN_H = 0.12           # ...this tall (share of the picture)...
BODY_MIN_CHANGE = MIN_CHANGE  # ...and its image must change over time
# Head size on a body box when no face was ever seen with it: about a third
# of the box width (shoulders are roughly three heads wide).
HEAD_W_ON_BODY = 0.33
HEAD_TOP_ON_BODY = 0.02


def _body_tracks(shot: Shot, bodies: list[list[BodyDet]], checked: list[int], fps: float) -> dict[int, list[BodyDet]]:
    frames = [i for i in checked if shot.start * fps <= i < shot.end * fps]
    if not frames:
        return {}
    rate = fps / max(1, (frames[1] - frames[0]) if len(frames) > 1 else 1)
    tracker = sv.ByteTrack(track_activation_threshold=0.4, lost_track_buffer=int(rate * 2) + 1,
                           minimum_matching_threshold=0.8, frame_rate=max(1, int(round(rate))),
                           minimum_consecutive_frames=2)
    out: dict[int, list[BodyDet]] = {}
    for i in frames:
        dets = bodies[i] if i < len(bodies) else []
        if not dets:
            tracker.update_with_detections(sv.Detections.empty())
            continue
        xyxy = np.array([[b.x, b.y, b.x + b.w, b.y + b.h] for b in dets], dtype=np.float32) * 1000
        res = tracker.update_with_detections(sv.Detections(
            xyxy=xyxy, confidence=np.array([b.score for b in dets], dtype=np.float32), class_id=np.zeros(len(dets), dtype=int)))
        if len(res) == 0:
            continue
        overlap = sv.box_iou_batch(res.xyxy, xyxy)
        for row, tid in enumerate(res.tracker_id):
            k = int(np.argmax(overlap[row]))
            if overlap[row, k] > 0.5:
                out.setdefault(int(tid), []).append(dets[k])
    return out


def _body_change(boxes: list[BodyDet], frames: Frames) -> float | None:
    """Mean change of the normalized body image about 1 s apart (None: too short)."""
    def patch(b: BodyDet):
        img = frames.data[b.frame]
        H, W = img.shape[:2]
        x0, y0, x1, y1 = int(b.x * W), int(b.y * H), int((b.x + b.w) * W), int((b.y + b.h) * H)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None
        g = cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (24, 48), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        return (g - g.mean()) / (g.std() + 1e-3)
    diffs = []
    for a in boxes:
        b = min(boxes, key=lambda o: abs(o.time - a.time - 1.0))
        if abs(b.time - a.time - 1.0) < 0.5:
            pa, pb = patch(a), patch(b)
            if pa is not None and pb is not None:
                diffs.append(float(np.abs(pa - pb).mean()))
    return float(np.mean(diffs)) if len(diffs) >= 2 else None


def _head_model(pairs: list[tuple[FaceDet, BodyDet]], pic_aspect: float) -> tuple[float, float, float, float]:
    """Face relative to its body box: (centre x offset / body w, top offset / body h,
    face w / body w, face h / body w)."""
    if pairs:
        return (float(np.median([(f.cx - (b.x + b.w / 2)) / b.w for f, b in pairs])),
                float(np.median([(f.y - b.y) / b.h for f, b in pairs])),
                float(np.median([f.w / b.w for f, b in pairs])),
                float(np.median([f.h / b.w for f, b in pairs])))
    # Square-ish head in pixels: normalized height = width * picture aspect * 1.2.
    return 0.0, HEAD_TOP_ON_BODY, HEAD_W_ON_BODY, HEAD_W_ON_BODY * pic_aspect * 1.2


def _estimated(b: BodyDet, model) -> FaceDet:
    dx, top, fw, fh = model
    w, h = fw * b.w, fh * b.w
    cx = b.x + b.w / 2 + dx * b.w
    y = b.y + top * b.h
    x = cx - w / 2
    # A frontal face's landmarks (eyes, nose, mouth corners), so gaze and facing read as neutral.
    lm = np.array([[x + 0.3 * w, y + 0.4 * h], [x + 0.7 * w, y + 0.4 * h], [cx, y + 0.6 * h],
                   [x + 0.35 * w, y + 0.8 * h], [x + 0.65 * w, y + 0.8 * h]], dtype=np.float32)
    return FaceDet(b.frame, b.time, float(x), float(y), float(w), float(h), 0.0, lm, estimated=True)


def join_bodies(shot: Shot, tracks: list[Track], bodies: list[list[BodyDet]], checked: list[int],
                frames: Frames) -> list[Track]:
    """The shot's tracks with faces joined to bodies, gaps filled and faceless
    people added (see the module docstring)."""
    fps = frames.fps
    pic_aspect = frames.width / frames.height
    btracks = _body_tracks(shot, bodies, checked, fps)
    if not btracks:
        return tracks
    face_at: dict[int, list[tuple[Track, FaceDet]]] = {}
    for t in tracks:
        for d in t.dets:
            face_at.setdefault(d.frame, []).append((t, d))
    # Votes: which face track is the head of which body track.
    votes: dict[tuple[int, int], int] = {}
    pairs: dict[int, list[tuple[FaceDet, BodyDet]]] = {}
    for bid, bs in btracks.items():
        for b in bs:
            scored = [(sc, t, d) for t, d in face_at.get(b.frame, []) if (sc := _head_score(d, b)) is not None]
            if scored:
                _, t, d = max(scored, key=lambda x: x[0])
                votes[(bid, id(t))] = votes.get((bid, id(t)), 0) + 1
                pairs.setdefault(bid, []).append((d, b))
    owner: dict[int, int] = {}
    strength: dict[int, int] = {}
    for t in tracks:
        cands = [(n, bid) for (bid, tid), n in votes.items() if tid == id(t) and n >= MIN_LINKS]
        if cands:
            strength[id(t)], owner[id(t)] = max(cands)
    # A body holds one face at a time: of face tracks that overlap in time and
    # claim the same body (people sitting close), only the best-linked joins it.
    by_id = {id(t): t for t in tracks}
    for tid in sorted(owner, key=lambda k: -strength[k]):
        if tid not in owner:
            continue
        a = by_id[tid]
        for other in list(owner):
            b = by_id[other]
            if other != tid and owner[other] == owner[tid] and strength[other] <= strength[tid] \
                    and min(a.dets[-1].time, b.dets[-1].time) - max(a.dets[0].time, b.dets[0].time) > FRAGMENT_OVERLAP_S:
                del owner[other]

    out: list[Track] = []
    joined: dict[int, Track] = {}                 # body track -> its person, placed where its first face track was
    next_id = max([t.track_id for t in tracks], default=0) + 1000
    shot_frames = max(1, int(np.ceil(shot.end * fps)) - int(np.ceil(shot.start * fps)))
    checked_in_shot = max(1, sum(1 for i in checked if shot.start * fps <= i < shot.end * fps))
    for bid, bs in sorted(btracks.items()):
        faces = [t for t in tracks if owner.get(id(t)) == bid]
        if not faces:
            presence = len(bs) / checked_in_shot
            tall = float(np.median([b.h for b in bs]))
            if presence < BODY_PRESENCE or tall < BODY_MIN_H:
                continue
            change = _body_change(bs, frames)
            if change is not None and change < BODY_MIN_CHANGE:
                continue
        if any("painting" in t.reason for t in faces):
            for t in faces:                       # a painted face with a painted body: leave it rejected
                owner.pop(id(t), None)
            continue
        model = _head_model(pairs.get(bid, []), pic_aspect)
        real = sorted({d.frame: d for t in faces for d in t.dets}.values(), key=lambda d: d.frame)
        times = np.array([d.time for d in real])
        filled = [_estimated(b, model) for b in bs
                  if not len(times) or np.min(np.abs(times - b.time)) > GAP_S / 2]
        dets = sorted(real + filled, key=lambda d: d.time)
        person = Track(shot.index, next_id + bid, dets)
        person.body_boxes = [bx for t in faces for bx in t.body_boxes] + [(b.x, b.y, b.x + b.w, b.y + b.h) for b in bs]
        person.liveness = max((t.liveness for t in faces), default=-1.0)
        person.body = max((t.body for t in faces), default=1.0)
        every = max(1, round(shot_frames / checked_in_shot))
        person.presence = min(1.0, (len(real) + len(filled) * every) / shot_frames)
        person.is_person = True
        person.reason = ("person (face and body)" if faces and all(not d.estimated for d in dets)
                         else "person (face, body when the face is lost)" if faces else "person (body only: face not visible)")
        joined[bid] = person
    placed: set[int] = set()
    for t in tracks:
        bid = owner.get(id(t))
        if bid is None or bid not in joined:
            out.append(t)                         # not joined to a body (or the body was dropped)
        elif bid not in placed:
            out.append(joined[bid])
            placed.add(bid)
    people = [t for t in out if t.is_person]
    for bid, p in joined.items():                 # faceless people
        if bid in placed:
            continue
        # A face too briefly linked to join is still that person: no second person for its body.
        body = _median_body(btracks[bid])
        if any(_head_score(_median_face(t), body) is not None for t in people):
            continue
        out.append(p)
    return _join_fragments(out, fps)


def _median_face(t: Track) -> FaceDet:
    d = t.dets[0]
    return FaceDet(d.frame, d.time, t.median("x"), t.median("y"), t.median("w"), t.median("h"), 1.0, d.landmarks)


def _median_body(bs: list[BodyDet]) -> BodyDet:
    return BodyDet(bs[0].frame, bs[0].time, *(float(np.median([getattr(b, a) for b in bs])) for a in ("x", "y", "w", "h")), 1.0)


FRAGMENT_OVERLAP_S = 0.3   # person tracks overlapping less than this in time...
FRAGMENT_GAP_FACES = 2.0   # ...whose end and start are within this many face widths are one person


def _join_fragments(tracks: list[Track], fps: float) -> list[Track]:
    """One person split in two by a tracker ID switch: join tracks that follow
    on from each other in time and place."""
    people = sorted((t for t in tracks if t.is_person), key=lambda t: t.dets[0].time)
    gone: set[int] = set()
    for i, a in enumerate(people):
        if id(a) in gone:
            continue
        for b in people[i + 1:]:
            if id(b) in gone:
                continue
            a_end, b_start = a.dets[-1], b.dets[0]
            overlap = a_end.time - b_start.time
            near = abs(a_end.cx - b_start.cx) <= FRAGMENT_GAP_FACES * max(a.median("w"), b.median("w"))
            if overlap <= FRAGMENT_OVERLAP_S and near:
                a.dets = sorted({d.frame: d for d in a.dets + b.dets}.values(), key=lambda d: d.time)
                a.body_boxes = a.body_boxes + b.body_boxes
                a.presence = min(1.0, a.presence + b.presence)
                gone.add(id(b))
    return [t for t in tracks if id(t) not in gone]
