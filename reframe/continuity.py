"""Continuity: the same framing across cuts of one setup, and smooth framing
changes inside a shot.

Setups. Talking-head videos are full of jump cuts from one camera position.
In the source the person stays the same size across those cuts; if each shot
were framed on its own, the person would resize at every cut. Shots showing
the same single person at about the same place and size form a setup; the
director picks once per setup and every shot in it gets one geometry (the
union of the shots' framings, so each shot's checks still pass).

Transitions. When the framing has to change inside a shot (content growing
or shrinking, a layout switch), the change eases over TRANSITION_S instead of
jumping. Changes at real cuts stay instant.
"""

from __future__ import annotations

import numpy as np

from .layout import Keyframe, Segment, framed_canvas

SETUP_POSITION = 0.08     # face centre within this share of the picture
SETUP_SIZE = 1.3          # face height within this ratio
TRANSITION_S = 0.5        # an eased change inside a shot takes this long (shorter for short segments)
# A sideways move bigger than this share of the crop width reads as a whip pan:
# cut instead (Watch to Edit, arXiv 1807.03125, uses 0.75).
PAN_LIMIT = 0.75
AT_CUT_S = 0.05
CUT_BELOW_IOU = 0.2       # framings overlapping this little (IoU) are different pictures: cut between them


def _single_person(analysis, shot_index: int):
    people = [t for t in analysis.tracks[shot_index] if t.is_person]
    return people[0] if len(people) == 1 else None


def setups(analysis, items: list[dict]) -> list[list[int]]:
    """Groups of positions in `items` (shot_options output) that show the same
    setup: the same single person, face at about the same place and size.
    Shots that match nothing (no person, several people) are their own group."""
    groups: list[list[int]] = []
    keys: list[tuple | None] = []
    for pos, item in enumerate(items):
        shot = item["shot"]
        p = _single_person(analysis, shot.index)
        key = None
        if p is not None:
            pid = analysis.people.person_of.get((shot.index, p.track_id))
            # Shots whose burned-in captions were replaced and shots that kept
            # them can't share a framing (the union would bring the band back).
            key = (pid, p.median("cx"), p.median("cy"), p.median("h"), item.get("captions") == "replaced")
        for g, k in zip(groups, keys):
            if key is None or k is None or key[4] != k[4]:
                continue
            same_person = key[0] is None or k[0] is None or key[0] == k[0]
            if (same_person and abs(key[1] - k[1]) <= SETUP_POSITION and abs(key[2] - k[2]) <= SETUP_POSITION
                    and max(key[3], k[3]) / max(min(key[3], k[3]), 1e-6) <= SETUP_SIZE):
                g.append(pos)
                break
        else:
            groups.append([pos])
            keys.append(key)
    return groups


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def unify(groups: list[list[int]], picked: list[list[Segment]], target_aspect: float, pic_aspect: float) -> None:
    """Give every shot of a setup the same framing geometry (in place).
    Only for one-zone single crops or framed views of the same layout."""
    for group in groups:
        if len(group) < 2:
            continue
        segs = [picked[pos] for pos in group]
        if any(len(s) != 1 or len(s[0].zones) != 1 for s in segs):
            continue
        layouts = {s[0].layout for s in segs}
        if len(layouts) != 1 or layouts & {"fit", "full", "split"}:
            continue
        layout = layouts.pop()
        if layout == "framed":
            region = _union([(k.x, k.y, k.x + k.w, k.y + k.h) for s in segs for k in s[0].zones[0].keyframes])
            w, h = region[2] - region[0], region[3] - region[1]
            canvas = framed_canvas(w, h, target_aspect, pic_aspect)
            for s in segs:
                z = s[0].zones[0]
                z.canvas = canvas
                z.keyframes = [Keyframe(k.t, region[0], region[1], w, h) for k in z.keyframes]
        else:  # single: one crop size; each shot keeps its own centre (clamped)
            w = max(k.w for s in segs for k in s[0].zones[0].keyframes)
            h = max(k.h for s in segs for k in s[0].zones[0].keyframes)
            for s in segs:
                z = s[0].zones[0]
                z.keyframes = [Keyframe(k.t, float(np.clip(k.x + k.w / 2 - w / 2, 0, 1 - w)),
                                        float(np.clip(k.y + k.h / 2 - h / 2, 0, 1 - h)), w, h) for k in z.keyframes]
        for s in segs:
            s[0].info["setup"] = group[0]


def _iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def transitions(segments: list[dict], cut_times: list[float]) -> None:
    """Mark how each segment starts (in place, on the output dicts):
    "cut" at real cuts, for split screens, for long sideways moves and between
    framings that barely overlap; "ease" for other changes inside a shot."""
    for prev, seg in zip(segments, segments[1:]):
        at_cut = any(abs(seg["start"] - c) <= AT_CUT_S for c in cut_times)
        if at_cut:
            continue
        if len(prev["zones"]) != 1 or len(seg["zones"]) != 1:
            seg["transition"] = {"type": "cut"}                    # split screens switch by cutting
            continue
        a, b = prev["zones"][0]["keyframes"][-1], seg["zones"][0]["keyframes"][0]
        same_size = abs(a["w"] - b["w"]) < 0.02 and abs(a["h"] - b["h"]) < 0.02 and prev["layout"] == seg["layout"]
        sideways = abs((a["x"] + a["w"] / 2) - (b["x"] + b["w"] / 2)) / max(min(a["w"], b["w"]), 1e-6)
        if same_size and sideways > PAN_LIMIT:
            seg["transition"] = {"type": "cut"}                    # a long move: cut, don't whip-pan
            continue
        if _iou(a, b) <= CUT_BELOW_IOU:
            seg["transition"] = {"type": "cut"}                    # a different picture altogether: cut, don't swoop
            continue
        duration = min(TRANSITION_S, 0.4 * (seg["end"] - seg["start"]), 0.4 * (prev["end"] - prev["start"]))
        seg["transition"] = {"type": "ease", "duration": round(max(duration, 0.1), 3)}
