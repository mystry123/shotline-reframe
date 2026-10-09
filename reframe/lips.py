"""Who is talking when several people are on screen: mouth motion.

For shots with two or more people, the shot is decoded again at LIP_FPS
(speech moves the lips faster than the analysis frame rate can see) and
faces are found in every frame. Each face is aligned to a standard 112x112
pose from its five landmarks (eyes, nose, mouth corners: the ArcFace
template, same point order as YuNet), so head turns and moves are taken
out. Mouth activity is the frame-to-frame change of the mouth region minus
that of the eye region (lighting changes and leftover head motion move both;
speech moves the mouth).

During each speech turn in the shot, the person whose mouth is clearly more
active than at quiet moments is the speaker. No transcript, or nobody
clearly ahead: no answer (the caller falls back to the voice-to-person
mapping, or to framing everyone).

No extra model or dependency: YuNet's landmarks are enough to align faces.

Calibrated on clip008 (720p interview, faces ~86 px in two-shots): 8 of 9
turns right, 1 no answer, none wrong. On small faces (clip001, ~54 px in
360p) the scores are close and it answers nothing, which is the intent: no
answer is safe, a wrong one moves the camera to the wrong person.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .decode import Frames, decode
from .faces import FaceDet, FaceDetector
from .probe import Rect, VideoInfo
from .shots import Shot
from .speakers import Turn
from .tracks import Track

LIP_FPS = 15.0
LIP_WIDTH = 960      # decode width for mouth motion (clip008: 640 px 7/9 right, 960 px 9/9, 1280 px 8/9; never wrong)
TEMPLATE = np.array([[38.29, 51.70], [73.53, 51.50], [56.03, 71.74], [41.55, 92.37], [70.73, 92.20]], np.float32)
MOUTH = (slice(80, 108), slice(32, 80))      # rows, cols in the aligned 112x112 face
EYES = (slice(38, 64), slice(26, 86))
MIN_TURN_OVERLAP_S = 0.8
MIN_SCORE = 0.6     # activity above the quiet baseline (aligned-face grey levels per frame)...
MIN_LEAD = 1.5      # ...and this many times the next person's


@dataclass
class LipTurn:
    shot: int
    start: float
    end: float
    speaker: str
    track_id: int | None          # who it is (None: not clear)
    scores: dict[int, float]      # track id -> activity above baseline


def _aligned(frame: np.ndarray, d: FaceDet) -> np.ndarray | None:
    H, W = frame.shape[:2]
    pts = d.landmarks * np.array([W, H], np.float32)
    M, _ = cv2.estimateAffinePartial2D(pts, TEMPLATE)
    if M is None:
        return None
    face = cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32) if face.ndim == 3 else face.astype(np.float32)


def _where(track: Track, t: float) -> FaceDet | None:
    """The track's real face nearest time t (within 0.5 s)."""
    real = [d for d in track.dets if not d.estimated]
    if not real:
        return None
    d = min(real, key=lambda d: abs(d.time - t))
    return d if abs(d.time - t) <= 0.5 else None


def activity(frames: Frames, start: float, people: list[Track]) -> dict[int, np.ndarray]:
    """Mouth activity per frame for each person (NaN where their face isn't seen)."""
    faces = FaceDetector().detect(frames)
    n = len(frames.data)
    out = {p.track_id: np.full(n, np.nan) for p in people}
    prev: dict[int, np.ndarray] = {}
    for i in range(n):
        t = start + i / frames.fps
        frame = frames.data[i]
        free = list(faces[i])
        for p in people:
            ref = _where(p, t)
            if ref is None or not free:
                prev.pop(p.track_id, None)
                continue
            d = min(free, key=lambda f: np.hypot(f.cx - ref.cx, f.cy - ref.cy))
            if np.hypot(d.cx - ref.cx, d.cy - ref.cy) > 0.8 * max(ref.w, d.w):
                prev.pop(p.track_id, None)
                continue
            free.remove(d)
            face = _aligned(frame, d)
            if face is None:
                prev.pop(p.track_id, None)
                continue
            last = prev.get(p.track_id)
            if last is not None:
                scale = face.std() + 1e-3
                mouth = np.abs(face[MOUTH] - last[MOUTH]).mean() / scale
                eyes = np.abs(face[EYES] - last[EYES]).mean() / scale
                out[p.track_id][i] = max(0.0, mouth - eyes) * 100
            prev[p.track_id] = face
    return out


def speakers_in_shot(info: VideoInfo, area: Rect, shot: Shot, people: list[Track], turns: list[Turn],
                     width: int) -> list[LipTurn]:
    turns_here = [t for t in turns if min(t.end, shot.end) - max(t.start, shot.start) >= MIN_TURN_OVERLAP_S]
    if len(people) < 2 or not turns_here:
        return []
    frames = decode(info, area, fps=LIP_FPS, max_width=width, start=shot.start, duration=shot.end - shot.start)
    act = activity(frames, shot.start, people)
    times = shot.start + np.arange(len(frames.data)) / frames.fps
    speech = np.zeros(len(times), bool)
    for t in turns:
        speech |= (times >= t.start) & (times <= t.end)
    out = []
    for turn in turns_here:
        sel = (times >= max(turn.start, shot.start)) & (times <= min(turn.end, shot.end))
        scores = {}
        for p in people:
            a = act[p.track_id]
            inside = a[sel & ~np.isnan(a)]
            if len(inside) < LIP_FPS * 0.5:
                continue
            quiet = a[~speech & ~np.isnan(a)]
            baseline = float(np.median(quiet)) if len(quiet) >= LIP_FPS else float(np.nanpercentile(a, 20))
            scores[p.track_id] = round(float(np.mean(inside)) - baseline, 3)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        who = None
        if ranked and ranked[0][1] >= MIN_SCORE:
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            if ranked[0][1] >= MIN_LEAD * max(second, MIN_SCORE / MIN_LEAD):
                who = ranked[0][0]
        out.append(LipTurn(shot.index, max(turn.start, shot.start), min(turn.end, shot.end), turn.speaker, who, scores))
    return out
