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
    info: dict = field(default_factory=dict)   # measurements (crop height, share of the person kept)


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
              fps: float, min_crop_h: float, speaking: "callable | None" = None,
              captions: "tuple[float, float, float, float] | None" = None) -> list[Segment]:
    """Segments for one shot. `speaking(t)` returns the track speaking at time t
    (from the transcript), or None. `captions`: box (x0, y0, x1, y1) of
    burned-in captions/titles that must stay in frame."""
    people = [t for t in tracks if t.is_person]
    if len(people) == 1 and abs(target_aspect / pic_aspect - 1) >= 0.03:
        return [_single_person(shot, people[0], target_aspect, pic_aspect, fps, min_crop_h, captions)]
    seg = _plan_shot(shot, tracks, target_aspect, pic_aspect, fps, min_crop_h, speaking)
    return seg if isinstance(seg, list) else [seg]


# ---------------------------------------------------------------------------
# One person: what must stay in frame, and how to fit it
# ---------------------------------------------------------------------------

KEEP_MARGIN = 0.03
FIT_WHEN_COVERS = 0.85   # must-keep area this big (share of the picture's height and width): show it all


# How much of the person to keep, in face widths/heights around the face:
# PREFERRED is head to chest with the shoulders; CORE is the least a crop may
# show (head, neck and the top of the shoulders). Captions are always kept.
PREFERRED = (1.3, 0.6, 3.0)   # each side, above, below
CORE = (0.8, 0.4, 2.0)


def _must_keep(person: Track, captions, spread=PREFERRED) -> tuple[float, float, float, float]:
    """The area around the person (plus burned-in captions) that must stay in frame."""
    cx, y, w, h = person.median("cx"), person.median("y"), person.median("w"), person.median("h")
    side, above, below = spread
    x0, x1 = cx - side * w, cx + side * w
    y0, y1 = y - above * h, y + below * h
    if captions:
        x0, y0 = min(x0, captions[0]), min(y0, captions[1])
        x1, y1 = max(x1, captions[2]), max(y1, captions[3])
    return max(0.0, x0 - KEEP_MARGIN), max(0.0, y0 - KEEP_MARGIN), min(1.0, x1 + KEEP_MARGIN), min(1.0, y1 + KEEP_MARGIN)


def _single_person(shot: Shot, p: Track, target_aspect: float, pic_aspect: float, fps: float,
                   min_crop_h: float, captions) -> Segment:
    kx0, ky0, kx1, ky1 = _must_keep(p, captions)
    max_w, max_h = _crop_for_target(target_aspect, pic_aspect)
    keep_w, keep_h = kx1 - kx0, ky1 - ky0
    if keep_w > max_w or keep_h > max_h:
        # The preferred area doesn't fit; a crop is still fine if the core does
        # (a little less shoulder beats shrinking the person over a blur).
        cx0, cy0, cx1, cy1 = _must_keep(p, captions, CORE)
        if cx1 - cx0 <= max_w and cy1 - cy0 <= max_h:
            kx0, ky0, kx1, ky1 = cx0, cy0, cx1, cy1
            keep_w, keep_h = kx1 - kx0, ky1 - ky0

    w, h = _crop_size(target_aspect, pic_aspect, p.median("h"), FACE_FRACTION_SINGLE, min_crop_h)
    if w < keep_w or h < keep_h:
        grow = max(keep_w / w, keep_h / h)
        w, h = min(max_w, w * grow), min(max_h, h * grow)
    if keep_w <= max_w and keep_h <= max_h and h >= min_crop_h - ZOOM_TOLERANCE:
        # Fits in a crop of the target shape within the zoom limit: follow the
        # person, never cutting the must-keep area.
        kfs = _follow(shot, [p], w, h, fps)
        for k in kfs:
            k.x = float(np.clip(k.x, max(0.0, kx1 - w), min(kx0, 1 - w)))
            k.y = float(np.clip(k.y, max(0.0, ky1 - h), min(ky0, 1 - h)))
        reason = "one person" + (" and their captions" if captions else "")
        return Segment(shot.start, shot.end, "single", [Zone((0, 0, 1, 1), kfs, f"track {p.track_id}")], reason)

    if keep_w >= FIT_WHEN_COVERS and keep_h >= FIT_WHEN_COVERS:
        return _fit_segment(shot, "the person and their captions fill the picture")

    # Too tall (or wide) for the target shape: show the must-keep area whole,
    # over a blurred copy of the picture (portrait video going to 16:9).
    return _framed_segment(shot, p, (kx0, ky0, kx1, ky1), target_aspect, pic_aspect,
                           "the person" + (" and their captions" if captions else "") + " don't fit the shape: shown whole over a blurred background")


def framed_canvas(region_w: float, region_h: float, target_aspect: float, pic_aspect: float):
    """Where a region (share of the picture) sits in the output when shown
    whole and as large as possible, centred: (left, top, width, height)."""
    region_aspect = (region_w * pic_aspect) / region_h          # pixels
    if region_aspect < target_aspect:
        cw, ch = region_aspect / target_aspect, 1.0
    else:
        cw, ch = 1.0, target_aspect / region_aspect
    return ((1 - cw) / 2, (1 - ch) / 2, cw, ch)


def _framed_segment(shot: Shot, p: Track, box, target_aspect: float, pic_aspect: float, reason: str) -> Segment:
    """`box` (x0, y0, x1, y1) shown whole, centred over a blurred copy of the picture."""
    kx0, ky0, kx1, ky1 = box
    keep_w, keep_h = kx1 - kx0, ky1 - ky0
    canvas = framed_canvas(keep_w, keep_h, target_aspect, pic_aspect)
    kfs = [Keyframe(shot.start, kx0, ky0, keep_w, keep_h), Keyframe(shot.end, kx0, ky0, keep_w, keep_h)]
    return Segment(shot.start, shot.end, "framed", [Zone(canvas, kfs, f"track {p.track_id}")], reason)


# ---------------------------------------------------------------------------
# Alternatives for the AI director: other reasonable framings of a shot
# ---------------------------------------------------------------------------

def _same_framing(a: list[Segment], b: list[Segment]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x.layout != y.layout or len(x.zones) != len(y.zones):
            return False
        for zx, zy in zip(x.zones, y.zones):
            kx, ky = zx.keyframes[0], zy.keyframes[0]
            if max(abs(kx.x - ky.x), abs(kx.y - ky.y), abs(kx.w - ky.w), abs(kx.h - ky.h)) > 0.05:
                return False
    return True


KEEP_TOLERANCE = 0.02   # share of the picture a crop may shave off a must-keep box
ZOOM_TOLERANCE = 0.01   # crop height may undershoot the zoom limit by this much
# Share of the height of the person's upper body (head down to the waist,
# where hands at work usually are) a framing must keep, unless the user chose
# tight framing. Height only: trimming the sides of a body (arms, chair) is a
# normal crop for a tall target; losing the hands and torso below is not.
SUBJECT_KEEP = 0.75
NEAR_FULL_CANVAS = 0.8   # a framed view covering more of the output than this (without filling its height) isn't offered
UPPER_BODY_FACES = 4.5  # the "upper body" ends this many face heights below the top of the face (about the waist)


def _inner(person: Track, captions):
    """What a crop must keep: the face (forehead to chin and a little neck; a
    tight close-up may trim the hair) and the captions whole when there are any."""
    x, y, w, h = person.median("x"), person.median("y"), person.median("w"), person.median("h")
    box = (x - 0.1 * w, y - 0.05 * h, x + 1.1 * w, y + 1.3 * h)
    if captions:
        box = (min(box[0], captions[0]), min(box[1], captions[1]), max(box[2], captions[2]), max(box[3], captions[3]))
    return box


def _keeps(seg: Segment, box) -> bool:
    """Every keyframe of every zone contains `box` (within KEEP_TOLERANCE)."""
    x0, y0, x1, y1 = box
    for z in seg.zones:
        for k in z.keyframes:
            if (k.x > x0 + KEEP_TOLERANCE or k.y > y0 + KEEP_TOLERANCE
                    or k.x + k.w < x1 - KEEP_TOLERANCE or k.y + k.h < y1 - KEEP_TOLERANCE):
                return False
    return True


def upper_body(person: Track) -> tuple[float, float, float, float]:
    """The part of the person a framing must not cut away: measured body
    (arms and hands included), down to UPPER_BODY_FACES face heights below
    the face (legs and floor don't count)."""
    x0, y0, x1, y1 = person.body_extent()
    limit = person.median("y") + UPPER_BODY_FACES * person.median("h")
    return x0, y0, x1, min(y1, max(limit, person.median("y") + person.median("h")))


def share_kept(seg: Segment, box, axes: str = "y") -> float:
    """Smallest share (over keyframes) of `box` the segment's crops keep,
    along `axes` ("y" height, "x" width, "xy" both)."""
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    worst = 1.0
    for z in seg.zones:
        for k in z.keyframes:
            if "x" in axes:
                worst = min(worst, max(0.0, min(x1, k.x + k.w) - max(x0, k.x)) / bw)
            if "y" in axes:
                worst = min(worst, max(0.0, min(y1, k.y + k.h) - max(y0, k.y)) / bh)
    return worst


def crop_height(seg: Segment) -> float:
    """Smallest crop height (share of the picture) per output height: what
    the zoom limit applies to (a framed region on half the canvas counts double)."""
    return min((k.h / max(z.canvas[3], 1e-6)) for z in seg.zones for k in z.keyframes)


def shot_alternatives(shot: Shot, tracks: list[Track], rules: list[Segment], target_aspect: float,
                      pic_aspect: float, fps: float, min_crop_h: float, captions=None,
                      strict: bool = True) -> dict[str, list[Segment]]:
    """Named framings of a shot, starting with the rules' choice ("rules").

    Every alternative is checked before it is offered: it keeps the face and
    any captions whole, stays within the zoom limit (`min_crop_h`, from the
    output resolution and the user's framing choice) and, unless `strict` is
    off (the user chose tight framing), keeps SUBJECT_KEEP of the person's
    upper body. Near-duplicates are left out, so a shot may have only one."""
    out: dict[str, list[Segment]] = {"rules": rules}
    people = [t for t in tracks if t.is_person]
    person = people[0] if len(people) == 1 else None

    def measure(segs: list[Segment]) -> None:
        for seg in segs:
            seg.info["crop_h"] = round(crop_height(seg), 3)
            if person is not None:
                seg.info["person_kept"] = round(share_kept(seg, upper_body(person)), 3)

    def add(name: str, segs: list[Segment]) -> None:
        measure(segs)
        for seg in segs:
            if seg.layout == "framed":
                _, _, cw, ch = seg.zones[0].canvas
                # Blurred bands are worth it beside a full-height picture, or
                # around a clearly smaller one; thin bands above and below a
                # near-full crop are just a worse version of that crop.
                if ch < 0.98 and cw * ch > NEAR_FULL_CANVAS:
                    return
        if person is not None:
            if not all(_keeps(seg, _inner(person, captions)) for seg in segs):
                return                                   # cuts the face or captions
            if any(seg.info["crop_h"] < min_crop_h - ZOOM_TOLERANCE for seg in segs):
                return                                   # zooms in further than allowed
            if strict and any(seg.info["person_kept"] < SUBJECT_KEEP for seg in segs):
                return                                   # cuts away too much of the person
        if not any(_same_framing(segs, v) for v in out.values()):
            out[name] = segs

    measure(rules)
    if abs(target_aspect / pic_aspect - 1) < 0.03:
        return out
    if person is not None:
        p = person
        tag = f"track {p.track_id}"
        max_w, max_h = _crop_for_target(target_aspect, pic_aspect)
        # Close: a medium close-up following the face.
        w, h = _crop_size(target_aspect, pic_aspect, p.median("h"), FACE_FRACTION_SINGLE, min_crop_h)
        add("close", [Segment(shot.start, shot.end, "single", [Zone((0, 0, 1, 1), _follow(shot, [p], w, h, fps), tag)],
                              "close-up on the person")])
        # Medium: the biggest crop of the target shape, on the person, keeping captions when they fit.
        kfs = _follow(shot, [p], max_w, max_h, fps)
        kx0, ky0, kx1, ky1 = _must_keep(p, captions, CORE)
        if kx1 - kx0 <= max_w and ky1 - ky0 <= max_h:
            for k in kfs:
                k.x = float(np.clip(k.x, max(0.0, kx1 - max_w), min(kx0, 1 - max_w)))
                k.y = float(np.clip(k.y, max(0.0, ky1 - max_h), min(ky0, 1 - max_h)))
        add("medium", [Segment(shot.start, shot.end, "single", [Zone((0, 0, 1, 1), kfs, tag)], "as close as the shape allows")])
        # Framed over a blurred background, from tightest to widest: head and
        # shoulders, head to chest, the whole upper body (hands and what they hold).
        body = upper_body(p)
        cap = captions or body
        body_box = (max(0.0, min(body[0], cap[0]) - KEEP_MARGIN), max(0.0, min(body[1], cap[1]) - KEEP_MARGIN),
                    min(1.0, max(body[2], cap[2]) + KEEP_MARGIN), min(1.0, max(body[3], cap[3]) + KEEP_MARGIN))
        # Upper: the closest framed view that still passes the person check:
        # the upper body trimmed from the bottom to SUBJECT_KEEP of its height
        # (never above the face or the captions).
        inner = _inner(p, captions)
        upper_bottom = max(body[1] + SUBJECT_KEEP * (body[3] - body[1]) + KEEP_TOLERANCE, inner[3])
        upper_box = (body_box[0], body_box[1], body_box[2], min(1.0, upper_bottom + KEEP_MARGIN))
        for name, box in (("whole", _must_keep(p, captions, CORE)), ("whole-wide", _must_keep(p, captions, PREFERRED)),
                          ("upper", upper_box), ("body", body_box)):
            add(name, [_framed_segment(shot, p, box, target_aspect, pic_aspect,
                                       "the person shown whole over a blurred background")])
    add("wide", [_fit_segment(shot, "the whole picture")])

    # The rules' framing is also the fallback when the director can't be
    # reached, so it must pass the same checks: if it cuts too much of the
    # person or zooms too far, the closest framing that passes takes its place.
    if person is not None and len(out) > 1:
        def passes(segs: list[Segment]) -> bool:
            return all(seg.info["crop_h"] >= min_crop_h - ZOOM_TOLERANCE
                       and (not strict or seg.info["person_kept"] >= SUBJECT_KEEP) for seg in segs)
        if not passes(out["rules"]):
            for name in ("upper", "whole", "whole-wide", "body", "medium", "close", "wide"):
                if name in out and passes(out[name]):
                    out["rules"] = out.pop(name)
                    break
    return out


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


# ---------------------------------------------------------------------------
# Shots without people: follow the content (text, UI, products)
# ---------------------------------------------------------------------------

CONTENT_MARGIN = 0.04        # keep this much space around content inside the crop
CONTENT_JUMP = 0.12          # content centre moving this far between samples = a scene change
MIN_CONTENT_RUN_S = 1.0      # graphics change quickly: a framing may be this short
MIN_FIT_SWITCH_S = 2.0       # but crop <-> fit switches are rarer


def _crop_for_target(target_aspect: float, pic_aspect: float) -> tuple[float, float]:
    """Largest crop of the picture with the target shape."""
    if target_aspect <= pic_aspect:
        return target_aspect / pic_aspect, 1.0
    return 1.0, pic_aspect / target_aspect


def _content_position(box, crop_w: float, crop_h: float) -> tuple[float, float]:
    """Crop top-left that centres the content, then nudges to keep all of it inside."""
    x = box.cx - crop_w / 2
    y = box.cy - crop_h / 2
    x = min(max(x, box.x1 + CONTENT_MARGIN - crop_w), box.x0 - CONTENT_MARGIN)
    y = min(max(y, box.y1 + CONTENT_MARGIN - crop_h), box.y0 - CONTENT_MARGIN)
    return float(np.clip(x, 0, 1 - crop_w)), float(np.clip(y, 0, 1 - crop_h))


def plan_content(shot: Shot, boxes: list, fps: float, target_aspect: float, pic_aspect: float) -> list[Segment]:
    """Segments for a shot with no people, from per-frame content boxes."""
    crop_w, crop_h = _crop_for_target(target_aspect, pic_aspect)
    first, last = int(np.ceil(shot.start * fps)), min(len(boxes), int(np.ceil(shot.end * fps)))
    samples = []   # (time, position or None for "fit")
    previous = None
    for i in range(first, last):
        b = boxes[i]
        if b is None:  # blank frame: keep what we had
            samples.append((i / fps, previous))
            continue
        fits = (b.x1 - b.x0) + 2 * CONTENT_MARGIN <= crop_w and (b.y1 - b.y0) + 2 * CONTENT_MARGIN <= crop_h
        previous = _content_position(b, crop_w, crop_h) if fits else None
        samples.append((i / fps, previous))
    if not samples:
        return [_fit_segment(shot, "nothing to frame")]

    # Runs of the same mode, split where the content jumps (an animated scene change).
    runs: list[list] = []
    for t, pos in samples:
        if runs:
            prev = runs[-1][2][-1]
            same_mode = (pos is None) == (prev is None)
            jumped = pos is not None and prev is not None and np.hypot(pos[0] - prev[0], pos[1] - prev[1]) > CONTENT_JUMP
            if same_mode and not jumped:
                runs[-1][1] = t + 1 / fps
                runs[-1][2].append(pos)
                continue
        runs.append([t, t + 1 / fps, [pos]])

    # No flicker: merge short runs into their neighbour (fit switches need longer).
    merged: list[list] = []
    for r in runs:
        is_fit = r[2][0] is None
        min_len = MIN_FIT_SWITCH_S if (is_fit or (merged and merged[-1][2][0] is None)) else MIN_CONTENT_RUN_S
        if merged and r[1] - r[0] < min_len:
            prev_fit = merged[-1][2][0] is None
            # A short crop inside fit (or the other way) becomes fit: safer to show everything.
            if prev_fit or is_fit:
                merged[-1][2] = [None] * len(merged[-1][2])
            merged[-1][1] = r[1]
            merged[-1][2].extend([None] * len(r[2]) if (prev_fit or is_fit) else r[2])
        else:
            merged.append(r)
    merged[0][0], merged[-1][1] = shot.start, shot.end

    segments = []
    for start, end, positions in merged:
        part = Shot(shot.index, start, end)
        if positions[0] is None or any(p is None for p in positions):
            segments.append(_fit_segment(part, "content too big for the frame"))
            continue
        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])
        slack_x, slack_y = 1 - crop_w, 1 - crop_h
        if np.ptp(xs) <= 0.02 + 0.3 * slack_x and np.ptp(ys) <= 0.02 + 0.3 * slack_y:
            x, y = float(np.median(xs)), float(np.median(ys))
            kfs = [Keyframe(start, x, y, crop_w, crop_h), Keyframe(end, x, y, crop_w, crop_h)]
        else:
            times = np.linspace(start, end, len(positions), endpoint=False)
            # _smooth_path works on centres; positions are top-left corners.
            sx = _smooth_path(times, xs + crop_w / 2, crop_w, fps) - crop_w / 2
            sy = _smooth_path(times, ys + crop_h / 2, crop_h, fps) - crop_h / 2
            kfs = [Keyframe(float(t), float(np.clip(a, 0, slack_x)), float(np.clip(b, 0, slack_y)), crop_w, crop_h)
                   for t, a, b in zip(times, sx, sy)]
            kfs.append(Keyframe(end, kfs[-1].x, kfs[-1].y, crop_w, crop_h))
            kfs = _rdp(kfs, 0.003)
        segments.append(Segment(start, end, "single", [Zone((0, 0, 1, 1), kfs, "content")], "following the content"))
    return segments


def _fit_segment(shot: Shot, reason: str) -> Segment:
    return Segment(shot.start, shot.end, "fit",
                   [Zone((0, 0, 1, 1), [Keyframe(shot.start, 0, 0, 1, 1), Keyframe(shot.end, 0, 0, 1, 1)], "picture")],
                   reason)
