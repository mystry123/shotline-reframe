"""Layout smoothing: no blink of a different layout between two cuts.

Each shot gets its layout on its own merits, so a short shot between two
cropped (full-frame) shots may come out as the picture over a blurred
background (two people, a wide shot, nobody to follow), or a short close-up
between two blurred ones as a crop. Each choice is right alone; together the
blurred bars pop in and out in under a second, which reads as a glitch: the
cut already changes the picture, and the frame changing shape twice on top
of it is the jolt.

So after the per-shot choices, a run of shots of one layout kind shorter
than ISLAND_S, between shots of another kind, takes its neighbours' kind
when it can do so without losing what matters:

- to a crop: the largest crop of the target shape (no zoom) on the people
  who matter - who is talking, else everyone if they fit, else the most
  prominent face - keeping their faces and any captions whole. Extra people,
  hands and edges may go: for a second, losing them costs less than two
  layout jumps. On-screen text being read (a slide, a screen) is never cut,
  so those shots keep the blurred background.
- to the blurred background: the shot's own largest picture over it.

When a short shot has to keep the blurred background, it gets the largest
picture that passes its checks (thin bars). Changes stay on the cuts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import (CONTENT_MARGIN, Segment, Zone, Keyframe, _crop_for_target, _follow, _inner, _keeps)

ISLAND_S = 1.5        # a layout shown for less than this between other layouts is a blink
BARS = ("fit", "framed")


def kind_of(segs: list[Segment]) -> str:
    """'fill' (the picture fills the frame), 'bars' (picture over a blurred
    background), 'split', or 'mixed' (a shot that changes layout inside)."""
    layouts = {s.layout for s in segs}
    if layouts <= set(BARS):
        return "bars"
    if layouts <= {"single", "full"}:
        return "fill"
    if layouts == {"split"}:
        return "split"
    return "mixed"


@dataclass
class Context:
    """What the engine knows about a shot that smoothing needs."""
    analysis: object
    target_aspect: float
    pic_aspect: float
    shot_text: "callable"        # shot -> (is a screen, captions box or None)
    captions: "callable"         # shot -> captions box in the lower picture, or None


def _speakers(ctx: Context, shot) -> list:
    """Tracks of this shot seen or heard talking during it, most talking first."""
    a = ctx.analysis
    tracks = [t for t in a.tracks[shot.index] if t.is_person]
    talk: dict[int, float] = {}
    for lt in getattr(a, "lip_turns", []) or []:
        if lt.shot == shot.index and lt.track_id is not None:
            overlap = min(lt.end, shot.end) - max(lt.start, shot.start)
            if overlap > 0:
                talk[lt.track_id] = talk.get(lt.track_id, 0.0) + overlap
    if not talk:
        for turn in getattr(a, "turns", []) or []:
            overlap = min(turn.end, shot.end) - max(turn.start, shot.start)
            pid = a.people.speaker_person.get(turn.speaker) if overlap > 0 else None
            if pid is None:
                continue
            for tr in tracks:
                if a.people.person_of.get((shot.index, tr.track_id)) == pid:
                    talk[tr.track_id] = talk.get(tr.track_id, 0.0) + overlap
    by_id = {t.track_id: t for t in tracks}
    return [by_id[i] for i, _ in sorted(talk.items(), key=lambda kv: -kv[1]) if i in by_id]


def _face_span(tracks) -> float:
    x0 = min(t.median("x") - 0.1 * t.median("w") for t in tracks)
    x1 = max(t.median("x") + 1.1 * t.median("w") for t in tracks)
    return x1 - x0


def fill_framing(ctx: Context, item: dict) -> list[Segment] | None:
    """A crop filling the frame for a short shot, or None when it would lose
    what matters (a face that must stay, captions, on-screen text)."""
    shot, opts = item["shot"], item["options"]
    # A checked crop the shot already has: the widest one.
    fills = [(n, segs) for n, segs in opts.items() if kind_of(segs) == "fill"]
    if fills:
        name, segs = max(fills, key=lambda ns: min(s.info.get("crop_h", 0) for s in ns[1]))
        return segs
    a = ctx.analysis
    screen, captions = ctx.shot_text(shot)
    if screen:
        return None                          # a slide or screen: its text is the point
    crop_w, crop_h = _crop_for_target(ctx.target_aspect, ctx.pic_aspect)
    people = [t for t in a.tracks[shot.index] if t.is_person]
    if not people:
        return _content_crop(ctx, shot, crop_w, crop_h, captions)
    # Who must stay: who is talking; else everyone if they fit; else the most prominent face.
    keep = _speakers(ctx, shot)[:2]
    if not keep:
        keep = people if _face_span(people) <= crop_w else [max(people, key=lambda t: t.median("h"))]
    if len(keep) == 2 and _face_span(keep) > crop_w:
        keep = keep[:1]                      # two talking far apart: the one talking most
    band = ctx.captions(shot)
    kfs = _follow(shot, keep, crop_w, crop_h, a.fps)
    seg = Segment(shot.start, shot.end, "single", [Zone((0, 0, 1, 1), kfs, "short shot")],
                  "short shot cropped like the shots around it (no blink of a blurred background)")
    if not all(_keeps(seg, lambda t, p=p: _inner(p, band, t)) for p in keep):
        return None
    seg.info["crop_h"] = round(crop_h, 3)
    seg.info["smoothed"] = True
    return [seg]


def _content_crop(ctx: Context, shot, crop_w: float, crop_h: float, captions) -> list[Segment] | None:
    """No people: a crop on the shot's content (all of it, with captions), or
    centred when there is no measured content (b-roll)."""
    a = ctx.analysis
    first, last = int(np.ceil(shot.start * a.fps)), min(len(a.content), int(np.ceil(shot.end * a.fps)))
    boxes = [b for b in a.content[first:last] if b is not None]
    if boxes:
        x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
        x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    else:
        x0, y0, x1, y1 = 0.5, 0.5, 0.5, 0.5
    if captions:
        x0, y0, x1, y1 = min(x0, captions[0]), min(y0, captions[1]), max(x1, captions[2]), max(y1, captions[3])
    if (x1 - x0) + 2 * CONTENT_MARGIN > crop_w or (y1 - y0) + 2 * CONTENT_MARGIN > crop_h:
        return None
    x = float(np.clip((x0 + x1) / 2 - crop_w / 2, 0, 1 - crop_w))
    y = float(np.clip((y0 + y1) / 2 - crop_h / 2, 0, 1 - crop_h))
    seg = Segment(shot.start, shot.end, "single",
                  [Zone((0, 0, 1, 1), [Keyframe(shot.start, x, y, crop_w, crop_h), Keyframe(shot.end, x, y, crop_w, crop_h)], "short shot")],
                  "short shot cropped like the shots around it (no blink of a blurred background)")
    seg.info["crop_h"] = round(crop_h, 3)
    seg.info["smoothed"] = True
    return [seg]


def _picture_share(seg: Segment, target_aspect: float, pic_aspect: float) -> float:
    """How much of the output the picture covers (the rest is blurred background)."""
    if seg.layout == "fit":
        r = pic_aspect / target_aspect
        return min(r, 1 / r)
    _, _, cw, ch = seg.zones[0].canvas
    return cw * ch


def largest_bars(ctx: Context, item: dict) -> list[Segment] | None:
    """The shot's checked picture-over-blur option with the biggest picture."""
    bars = [segs for segs in item["options"].values() if kind_of(segs) == "bars"]
    if not bars:
        return None
    return max(bars, key=lambda segs: min(_picture_share(s, ctx.target_aspect, ctx.pic_aspect) for s in segs))


def smooth(ctx: Context, items: list[dict], picked: list[list[Segment]], notes: list[dict] | None = None) -> int:
    """Replaces blinks in `picked` (per item) in place. Returns how many shots changed.
    `notes` (per item) gets {"smoothed": reason} where a shot changed."""
    changed = 0
    for _ in range(len(items) + 1):          # one fix per pass (runs change); at most one per shot
        runs = _runs(items, picked)
        did = False
        for i, (kind, a, b, dur) in enumerate(runs):
            if dur >= ISLAND_S or kind == "mixed":
                continue
            prev = runs[i - 1] if i > 0 else None
            nxt = runs[i + 1] if i + 1 < len(runs) else None
            neighbours = [r for r in (prev, nxt) if r is not None and r[0] != "mixed"]
            if not neighbours:
                continue
            want = max(neighbours, key=lambda r: r[3])[0] if len({r[0] for r in neighbours}) > 1 else neighbours[0][0]
            if want == kind or want == "split":
                continue
            new = []
            for pos in range(a, b):
                segs = fill_framing(ctx, items[pos]) if want == "fill" else largest_bars(ctx, items[pos])
                if segs is None:
                    new = None
                    break
                new.append(segs)
            if new is None:
                # Can't take the neighbours' layout: at least the biggest picture (thin bars).
                if kind == "bars":
                    for pos in range(a, b):
                        best = largest_bars(ctx, items[pos])
                        if best is not None and best is not picked[pos]:
                            picked[pos] = best
                            if notes is not None:
                                notes[pos]["smoothed"] = "short shot kept over a blurred background, as large as it fits"
                continue
            for pos, segs in zip(range(a, b), new):
                picked[pos] = segs
                changed += 1
                if notes is not None:
                    notes[pos]["smoothed"] = f"short shot shown like the shots around it ({want})"
            did = True
            break                            # runs changed: recompute
        if not did:
            break
    return changed


def _runs(items: list[dict], picked: list[list[Segment]]) -> list[tuple[str, int, int, float]]:
    """(kind, first item, last item + 1, duration) of consecutive items of one kind."""
    runs: list[list] = []
    for pos, segs in enumerate(picked):
        k = kind_of(segs)
        shot = items[pos]["shot"]
        if runs and runs[-1][0] == k:
            runs[-1][2] = pos + 1
            runs[-1][3] += shot.end - shot.start
        else:
            runs.append([k, pos, pos + 1, shot.end - shot.start])
    return [tuple(r) for r in runs]


def blinks(segments: list[dict] | list[Segment], duration: float | None = None) -> int:
    """How many layout runs shorter than ISLAND_S sit between two other layouts
    (the metric smoothing drives to zero)."""
    seq: list[tuple[str, float]] = []
    for s in segments:
        layout = s["layout"] if isinstance(s, dict) else s.layout
        start = s["start"] if isinstance(s, dict) else s.start
        end = s["end"] if isinstance(s, dict) else s.end
        k = "bars" if layout in BARS else "fill" if layout in ("single", "full") else layout
        if seq and seq[-1][0] == k:
            seq[-1] = (k, seq[-1][1] + end - start)
        else:
            seq.append((k, end - start))
    return sum(1 for i in range(1, len(seq) - 1) if seq[i][1] < ISLAND_S)
