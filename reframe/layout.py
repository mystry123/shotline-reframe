"""Layout per shot and the camera path inside it.

Coordinates: everything here is normalized to the *picture area* (0..1); the
engine converts to full-frame coordinates at the end.

Layouts
- full:   the target shape matches the picture (e.g. a 16:9 interview inside a
          9:16 export, going to 16:9): use the whole picture.
- single: one crop following one subject.
- split:  two stacked zones, one person each (left person on top, fixed for
          the shot).
- fit:    nothing to follow (no people, or too many): the whole picture scaled
          into the frame over a blurred copy.

Visual-only rules (speaker-aware rules plug in later):
- 0 people  → fit
- 1 person  → single on them
- 2 people  → both fit in one crop → single wide on both
              one faces the camera, the other is turned away (over-the-shoulder)
                → single on the one facing the camera
              otherwise → split
- 3+ people → fit
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .shots import Shot
from .tracks import Track

# Framing
FACE_FRACTION_SINGLE = 0.22   # face height / crop height for a single (medium close-up)
FACE_FRACTION_SPLIT = 0.40    # face height / zone height in a split
EYE_LINE = 1 / 3              # eyes a third from the top of the crop
LOOK_ROOM = 0.08              # shift towards where the face looks, share of crop width
# Camera
STATIONARY_SLACK = 0.6        # hold still if the subject's range fits in 60% of the free space
DEAD_ZONE = 0.15              # subject may wander ±15% of crop width before the camera moves
SMOOTH_SIGMA_S = 0.6
MAX_SPEED = 0.25              # picture widths per second
TURNED_AWAY = 0.2             # eye distance / face width below this = profile or back of head


@dataclass
class Keyframe:
    t: float
    x: float
    y: float
    w: float
    h: float


@dataclass
class Zone:
    canvas: tuple[float, float, float, float]   # left, top, width, height of the output (0..1)
    keyframes: list[Keyframe] = field(default_factory=list)
    subject: str = ""


@dataclass
class Segment:
    start: float
    end: float
    layout: str
    zones: list[Zone]
    reason: str


def _facing(t: Track) -> float:
    return float(np.median([np.linalg.norm(d.landmarks[0] - d.landmarks[1]) / max(d.w, 1e-6) for d in t.dets]))


def _gaze(t: Track) -> float:
    """-1 looking left … +1 looking right (nose offset from the eye midpoint)."""
    vals = []
    for d in t.dets:
        eye_mid = d.landmarks[:2, 0].mean()
        vals.append((d.landmarks[2, 0] - eye_mid) / max(d.w, 1e-6))
    return float(np.clip(np.median(vals) * 6, -1, 1))


def _crop_size(target_aspect: float, pic_aspect: float, face_h: float, face_fraction: float,
               min_crop_h: float) -> tuple[float, float]:
    """Crop (w, h) in picture-normalized units for an output of `target_aspect`
    (w/h in pixels). Height from the face size, never smaller than `min_crop_h`
    (resolution limit) and never bigger than the picture."""
    h = np.clip(face_h / face_fraction, min_crop_h, 1.0)
    w = h * target_aspect / pic_aspect
    if w > 1.0:  # wider than the picture allows: full width, shorter height
        w = 1.0
        h = pic_aspect / target_aspect
    return float(w), float(h)


def _smooth_path(times: np.ndarray, xs: np.ndarray, crop_w: float, fps: float) -> np.ndarray:
    """Camera centre per sample: dead zone, zero-phase Gaussian smoothing, speed limit."""
    cam = np.empty_like(xs)
    cam[0] = xs[0]
    half = DEAD_ZONE * crop_w
    for i in range(1, len(xs)):
        off = xs[i] - cam[i - 1]
        cam[i] = cam[i - 1] + (off - np.sign(off) * half if abs(off) > half else 0.0)
    sigma = max(1.0, SMOOTH_SIGMA_S * fps)
    k = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(cam, len(k) // 2, mode="edge")
    cam = np.convolve(padded, k, mode="valid")[: len(xs)]
    max_step = MAX_SPEED / fps
    for i in range(1, len(cam)):
        cam[i] = cam[i - 1] + np.clip(cam[i] - cam[i - 1], -max_step, max_step)
    return cam


def _rdp(points: list[Keyframe], tol: float) -> list[Keyframe]:
    if len(points) <= 2:
        return points
    a, b = points[0], points[-1]
    span = max(b.t - a.t, 1e-9)
    worst, idx = 0.0, 0
    for i, p in enumerate(points[1:-1], 1):
        r = (p.t - a.t) / span
        err = max(abs(p.x - (a.x + r * (b.x - a.x))), abs(p.y - (a.y + r * (b.y - a.y))))
        if err > worst:
            worst, idx = err, i
    if worst <= tol:
        return [a, b]
    return _rdp(points[: idx + 1], tol)[:-1] + _rdp(points[idx:], tol)


def _follow(shot: Shot, tracks: list[Track], crop_w: float, crop_h: float, fps: float) -> list[Keyframe]:
    """Keyframes for a crop following one or more tracks (their combined centre)."""
    times = np.arange(shot.start, shot.end, 1 / fps)
    cx, eye = [], []
    for t in times:
        pts = []
        for tr in tracks:
            near = min(tr.dets, key=lambda d: abs(d.time - t))
            pts.append((near.cx, near.eye_y, _gaze(tr)))
        cx.append(np.mean([p[0] for p in pts]) + (LOOK_ROOM * crop_w * pts[0][2] if len(pts) == 1 else 0))
        eye.append(np.mean([p[1] for p in pts]))
    cx, eye = np.array(cx), np.array(eye)
    free = 1.0 - crop_w
    stationary = (cx.max() - cx.min()) <= STATIONARY_SLACK * max(free, 1e-6) or free <= 1e-6
    if stationary:
        x = float(np.clip(np.median(cx) - crop_w / 2, 0, free))
        y = float(np.clip(np.median(eye) - EYE_LINE * crop_h, 0, 1 - crop_h))
        return [Keyframe(shot.start, x, y, crop_w, crop_h), Keyframe(shot.end, x, y, crop_w, crop_h)]
    centre = _smooth_path(times, cx, crop_w, fps)
    y = float(np.clip(np.median(eye) - EYE_LINE * crop_h, 0, 1 - crop_h))
    kfs = [Keyframe(float(t), float(np.clip(c - crop_w / 2, 0, free)), y, crop_w, crop_h) for t, c in zip(times, centre)]
    kfs.append(Keyframe(shot.end, kfs[-1].x, y, crop_w, crop_h))
    return _rdp(kfs, 0.003)


SWITCH_DELAY = 0.25          # cut to a new speaker this long after they start
MIN_LAYOUT_S = 2.5           # no framing shorter than this


def _speaker_runs(shot: Shot, a: Track, b: Track, speaking: "callable") -> list[tuple[float, float, Track]] | None:
    """Follow-the-speaker runs inside a shot, or None when the speaker can't
    be told (then the caller splits instead of guessing)."""
    step = 0.2
    times = np.arange(shot.start, shot.end, step)
    who: list[Track | None] = []
    for t in times:
        tr = speaking(t - SWITCH_DELAY)
        who.append(tr if tr in (a, b) else None)
    if all(w is None for w in who):
        return None
    # Hold the last speaker through pauses; before anyone speaks, use the first speaker.
    first = next(w for w in who if w is not None)
    current = first
    filled = []
    for w in who:
        current = w or current
        filled.append(current)
    runs: list[list] = []
    for t, w in zip(times, filled):
        if runs and runs[-1][2] is w:
            runs[-1][1] = t + step
        else:
            runs.append([t, t + step, w])
    # Too-short framings merge into the previous one (no flicker).
    merged: list[list] = []
    for r in runs:
        if merged and (r[1] - r[0] < MIN_LAYOUT_S or merged[-1][2] is r[2]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < MIN_LAYOUT_S:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    merged[0][0], merged[-1][1] = shot.start, shot.end
    return [(r[0], r[1], r[2]) for r in merged]


def plan_shot(shot: Shot, tracks: list[Track], target_aspect: float, pic_aspect: float,
              fps: float, min_crop_h: float, speaking: "callable | None" = None) -> list[Segment]:
    """Segments for one shot. `speaking(t)` returns the track speaking at time t
    (from the transcript), or None."""
    seg = _plan_shot(shot, tracks, target_aspect, pic_aspect, fps, min_crop_h, speaking)
    return seg if isinstance(seg, list) else [seg]


def _plan_shot(shot: Shot, tracks: list[Track], target_aspect: float, pic_aspect: float,
               fps: float, min_crop_h: float, speaking: "callable | None") -> "Segment | list[Segment]":
    people = [t for t in tracks if t.is_person]
    if abs(target_aspect / pic_aspect - 1) < 0.03:
        return Segment(shot.start, shot.end, "full",
                       [Zone((0, 0, 1, 1), [Keyframe(shot.start, 0, 0, 1, 1), Keyframe(shot.end, 0, 0, 1, 1)], "picture")],
                       "target shape matches the picture")
    if not people or len(people) >= 3:
        return Segment(shot.start, shot.end, "fit",
                       [Zone((0, 0, 1, 1), [Keyframe(shot.start, 0, 0, 1, 1), Keyframe(shot.end, 0, 0, 1, 1)], "picture")],
                       "no people" if not people else f"{len(people)} people")

    if len(people) == 1:
        p = people[0]
        w, h = _crop_size(target_aspect, pic_aspect, p.median("h"), FACE_FRACTION_SINGLE, min_crop_h)
        return Segment(shot.start, shot.end, "single", [Zone((0, 0, 1, 1), _follow(shot, [p], w, h, fps), f"track {p.track_id}")],
                       "one person")

    a, b = sorted(people, key=lambda t: t.median("cx"))
    # Both fit in one crop?
    w, h = _crop_size(target_aspect, pic_aspect, max(a.median("h"), b.median("h")), FACE_FRACTION_SINGLE, min_crop_h)
    span = (b.median("x") + b.median("w")) - a.median("x")
    if span * 1.3 <= w:
        return Segment(shot.start, shot.end, "single", [Zone((0, 0, 1, 1), _follow(shot, [a, b], w, h, fps), "both")],
                       "both people fit in one crop")
    fa, fb = _facing(a), _facing(b)
    if min(fa, fb) < TURNED_AWAY <= max(fa, fb):
        main = a if fa > fb else b
        w, h = _crop_size(target_aspect, pic_aspect, main.median("h"), FACE_FRACTION_SINGLE, min_crop_h)
        return Segment(shot.start, shot.end, "single", [Zone((0, 0, 1, 1), _follow(shot, [main], w, h, fps), f"track {main.track_id}")],
                       "over-the-shoulder: framing the person facing the camera")
    # Follow whoever is talking, when we know who that is.
    runs = _speaker_runs(shot, a, b, speaking) if speaking else None
    if runs:
        segments = []
        for start, end, who in runs:
            part = Shot(shot.index, start, end)
            w, h = _crop_size(target_aspect, pic_aspect, who.median("h"), FACE_FRACTION_SINGLE, min_crop_h)
            segments.append(Segment(start, end, "single", [Zone((0, 0, 1, 1), _follow(part, [who], w, h, fps), f"track {who.track_id}")],
                                    "following the speaker"))
        return segments
    zone_aspect = target_aspect * 2  # each zone is the full width and half the height
    zones = []
    for i, p in enumerate((a, b)):
        zw, zh = _crop_size(zone_aspect, pic_aspect, p.median("h"), FACE_FRACTION_SPLIT, min_crop_h / 2)
        zones.append(Zone((0, 0.5 * i, 1, 0.5), _follow(shot, [p], zw, zh, fps), f"track {p.track_id}"))
    return Segment(shot.start, shot.end, "split", zones, "two people apart")
