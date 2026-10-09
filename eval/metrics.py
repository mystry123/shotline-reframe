"""Reframe quality metrics for one clip and one target shape.

The plan is replayed exactly as the renderer draws it (reframe.render's view
model, eased transitions included), sampled at SAMPLE_FPS. Coordinates are
shares of the full source frame.

People and captions (no labels needed; from the engine's own detections):
    face_cut_rate      share of samples in single-person shots whose face box (where it is then) isn't fully in the crop
    caption_cut_rate   share of samples in shots with burned-in captions where the crop slices the caption band
    person_kept        mean share of the person's upper body (height) kept, single-person shots
Motion (no labels):
    jumps              framing jumps over JUMP of the frame width in one sample with no motion around
                       it (a glide moves fast too, but never alone), away from cuts and transitions
    jerk_p95           95th percentile of |third derivative| of the crop centre (frame widths / s^3)
    reversals_per_min  direction changes of the crop centre per minute (moves slower than 1% / s ignored)
    zoom_p95           95th percentile of output pixels per source pixel at the plan's output height
Against labels (synthetic truth, or eval/labels/<clip>.json):
    transition_f1      boundary F1 (a cut within CUT_TOL s; a gradual transition anywhere inside it)
    picture_iou        IoU of the detected picture area with the true one
"""

from __future__ import annotations

import numpy as np

from reframe.engine import picture_captions
from reframe.layout import upper_body
from reframe.render import _blend, _ease, _view

SAMPLE_FPS = 10
JUMP = 0.05
CUT_TOL = 0.12
EDGE_TOL = 0.01


def _views(plan: dict) -> list[tuple[float, tuple, tuple]]:
    """(t, canvas, crop) per sample, as the renderer would draw it."""
    src = plan["source"]
    W, H, pic = src["width"], src["height"], src["picture"]
    a, b = (float(v) for v in plan["target"].split(":"))
    segs = plan["segments"]
    out = []
    for t in np.arange(0, src["duration"], 1 / SAMPLE_FPS):
        k = next((i for i, s in enumerate(segs) if s["start"] <= t < s["end"]), len(segs) - 1)
        seg = segs[k]
        view = _view(seg, t, pic, W, H, a / b)
        if view is None:                         # split screen: judge its first zone
            z = seg["zones"][0]
            from reframe.render import _interp
            c = z["canvas"]
            view = ((c["left"], c["top"], c["width"], c["height"]), _interp(z["keyframes"], t), 0.0)
        tr = seg.get("transition") or {}
        if tr.get("type") == "ease" and k > 0 and t < seg["start"] + tr.get("duration", 0):
            prev = segs[k - 1]
            before = _view(prev, min(t, prev["end"] - 1e-3), pic, W, H, a / b)
            if before is not None:
                view = _blend(before, view, _ease((t - seg["start"]) / tr["duration"]))
        out.append((float(t), view[0], tuple(view[1])))
    return out


def _to_frame(box, area, W, H):
    """Picture-normalized (x0, y0, x1, y1) → full-frame shares."""
    x0, y0, x1, y1 = box
    return ((area.x + x0 * area.w) / W, (area.y + y0 * area.h) / H, (area.x + x1 * area.w) / W, (area.y + y1 * area.h) / H)


def _inside(box, crop) -> bool:
    x, y, w, h = crop
    return box[0] >= x - EDGE_TOL and box[1] >= y - EDGE_TOL and box[2] <= x + w + EDGE_TOL and box[3] <= y + h + EDGE_TOL


def _overlaps(box, crop) -> bool:
    x, y, w, h = crop
    return box[0] < x + w and box[2] > x and box[1] < y + h and box[3] > y


def _boundaries(analysis) -> list[float]:
    """Where the engine starts a new shot (or a transition)."""
    trans = getattr(analysis, "transitions", None)
    if trans:
        return [((t.start + t.end) / 2) for t in trans]
    return [s.start for s in analysis.shots[1:]]


def transition_f1(analysis, truth: list[dict]) -> dict:
    found = _boundaries(analysis)
    matched_found, hits = set(), 0
    by_type: dict[str, list[int]] = {}
    for tr in truth:
        lo, hi = tr["start"] - CUT_TOL, tr["end"] + CUT_TOL
        idx = next((i for i, b in enumerate(found) if lo <= b <= hi and i not in matched_found), None)
        by_type.setdefault(tr["type"], [0, 0])[1] += 1
        if idx is not None:
            matched_found.add(idx)
            hits += 1
            by_type[tr["type"]][0] += 1
    precision = hits / len(found) if found else 1.0
    recall = hits / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"transition_f1": round(f1, 3), "transition_found": len(found), "transition_truth": len(truth),
            **{f"recall_{k}": round(v[0] / v[1], 3) for k, v in by_type.items()}}


def picture_iou(analysis, truth: dict) -> float:
    a = analysis.area
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.w, a.y + a.h
    bx0, by0, bx1, by1 = truth["x"], truth["y"], truth["x"] + truth["w"], truth["y"] + truth["h"]
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return round(inter / union, 4) if union else 0.0


def plan_metrics(analysis, plan: dict) -> dict:
    W, H = analysis.info.width, analysis.info.height
    area = analysis.area
    views = _views(plan)
    cuts = _boundaries(analysis)
    trans = [(t.start, t.end) for t in (getattr(analysis, "transitions", None) or [])]
    near_cut = lambda t: any(abs(t - c) <= 1.5 / SAMPLE_FPS for c in cuts) or any(a - 0.1 <= t <= b + 0.1 for a, b in trans)  # noqa: E731

    face_cut = face_n = cap_cut = cap_n = 0
    kept = []
    from reframe.engine import TEXT_PAD_S, shot_pieces
    starts = {round(s.start, 3) for s in analysis.shots}
    for shot in (piece for s in analysis.shots for piece in shot_pieces(analysis, s)):
        # A piece starting inside a shot eases in before its text appears (TEXT_PAD_S lead).
        lead = 0.0 if round(shot.start, 3) in starts else TEXT_PAD_S
        people = [tr for tr in analysis.tracks[shot.index] if tr.is_person]
        band = picture_captions(analysis, shot)   # per piece: text is protected only while it is on screen
        band_ff = _to_frame(band, area, W, H) if band else None
        samples = [v for v in views if shot.start <= v[0] < shot.end]
        if len(people) == 1:
            p = people[0]
            fx, fy, fw, fh = p.median("x"), p.median("y"), p.median("w"), p.median("h")
            for t, _, crop in samples:
                # Where the person is at this moment (a person who walks isn't at their median).
                dx, dy = p.shift_at(t)
                face = _to_frame((fx + dx, fy + dy, fx + dx + fw, fy + dy + fh), area, W, H)
                body = _to_frame(upper_body(p, t), area, W, H)
                face_n += 1
                face_cut += not _inside(face, crop)
                x, y, w, h = crop
                kept.append(max(0.0, min(body[3], y + h) - max(body[1], y)) / max(body[3] - body[1], 1e-6))
        if band_ff:
            for t, _, crop in samples:
                if t < shot.start + lead:
                    continue
                cap_n += 1
                cap_cut += _overlaps(band_ff, crop) and not _inside(band_ff, crop)

    centres = np.array([(c[0] + c[2] / 2, c[1] + c[3] / 2) for _, _, c in views])
    ts = np.array([t for t, _, _ in views])
    # A jump is an isolated step: (nearly) still before and after, one big move
    # between. A fast glide moves a lot per sample too, but never alone.
    steps = np.r_[0.0, np.hypot(*np.diff(centres, axis=0).T), 0.0]
    jumps = sum(1 for i in range(1, len(views))
                if steps[i] > JUMP and steps[i - 1] < JUMP / 4 and steps[i + 1] < JUMP / 4 and not near_cut(ts[i]))
    # Per-shot derivatives (cuts are allowed to jump).
    jerks, reversals = [], 0
    dt = 1 / SAMPLE_FPS
    edges = [0.0, *cuts, float(plan["source"]["duration"]) + 1]
    for a0, a1 in zip(edges, edges[1:]):
        sel = (ts >= a0 + 2 * dt) & (ts < a1 - 2 * dt)
        x = centres[sel, 0]
        if len(x) >= 4:
            jerks.extend(np.abs(np.diff(x, 3)) / dt ** 3)
        v = np.diff(x) / dt if len(x) >= 2 else np.array([])
        moving = v[np.abs(v) > 0.01]
        reversals += int(np.sum(np.diff(np.sign(moving)) != 0)) if len(moving) > 1 else 0
    out_h = plan.get("settings", {}).get("outputHeight", 1080)
    zooms = [(canvas[3] * out_h) / max(crop[3] * H, 1e-6) for _, canvas, crop in views]
    minutes = max(plan["source"]["duration"] / 60, 1 / 60)
    return {
        "face_cut_rate": round(face_cut / face_n, 4) if face_n else None,
        "caption_cut_rate": round(cap_cut / cap_n, 4) if cap_n else None,
        "person_kept": round(float(np.mean(kept)), 3) if kept else None,
        "jumps": jumps,
        "jerk_p95": round(float(np.percentile(jerks, 95)), 3) if jerks else 0.0,
        "reversals_per_min": round(reversals / minutes, 2),
        "zoom_p95": round(float(np.percentile(zooms, 95)), 2) if zooms else None,
    }


def camera_metrics(analysis, truth: dict) -> dict:
    """Measured source-camera motion against the synthetic truth: the pan speed
    (frame widths per second) and the total zoom over the clip."""
    steps = [m for m in getattr(analysis, "motion", []) if m is not None]
    if not steps or not truth:
        return {}
    out = {}
    W = analysis.area.w
    if "pan_px_per_s" in truth:
        measured = float(np.mean([m.dx for m in steps])) * analysis.fps
        expected = truth["pan_px_per_s"] / W
        out["camera_error"] = round(abs(measured - expected) / expected, 3)
    if "zoom" in truth:
        measured = float(np.prod([m.scale for m in steps]))
        frames = len(analysis.motion)
        lo, hi = truth["zoom"]
        expected = lo + (hi - lo) * min(1.0, (frames - 1) / analysis.fps / truth["over_s"])
        out["camera_error"] = round(abs(measured - expected) / (expected - lo), 3)
    return out


def speaker_metrics(analysis, speaker_x: dict) -> dict:
    """In shots with several people, during speech: does the engine name the
    person who is talking? `speaker_x`: where each voice sits (x range,
    share of the picture). wrong_speaker_rate: share of answered speech
    seconds given to the wrong person; speaker_answered: share answered."""
    from reframe.speakers import active_person
    right = wrong = total = 0.0
    step = 0.2
    for shot in analysis.shots:
        people = [t for t in analysis.tracks[shot.index] if t.is_person]
        if len(people) < 2:
            continue
        for turn in analysis.turns:
            if turn.speaker not in speaker_x:
                continue
            lo, hi = speaker_x[turn.speaker]
            for t in np.arange(max(turn.start, shot.start), min(turn.end, shot.end), step):
                total += step
                who = next((tr for lt in getattr(analysis, "lip_turns", []) if lt.shot == shot.index
                            and lt.track_id is not None and lt.start <= t <= lt.end
                            for tr in people if tr.track_id == lt.track_id), None)
                if who is None:
                    pid = active_person(analysis.turns, analysis.people, t)
                    who = next((tr for tr in people if analysis.people.person_of.get((shot.index, tr.track_id)) == pid), None)
                if who is None:
                    continue
                if lo <= who.median("cx") < hi:
                    right += step
                else:
                    wrong += step
    if total == 0:
        return {}
    return {"wrong_speaker_rate": round(wrong / max(right + wrong, 1e-9), 3), "speaker_answered": round((right + wrong) / total, 3)}
