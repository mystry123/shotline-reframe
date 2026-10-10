"""Layout smoothing without video (runs in CI): a short shot of another
layout between cuts takes its neighbours' layout when it can keep what
matters, and gets the biggest picture when it can't."""

from types import SimpleNamespace

import numpy as np

from reframe.faces import FaceDet
from reframe.layout import Keyframe, Segment, Zone, _inner, _keeps
from reframe.shots import Shot
from reframe.smoothing import ISLAND_S, Context, blinks, kind_of, smooth
from reframe.speakers import People
from reframe.tracks import Track

TARGET, PIC = 9 / 16, 16 / 9


def _track(shot: Shot, track_id: int, x: float, y: float, h: float) -> Track:
    w = h * 0.8
    lm = np.array([[x + 0.3 * w, y + 0.4 * h], [x + 0.7 * w, y + 0.4 * h], [x + 0.5 * w, y + 0.6 * h],
                   [x + 0.35 * w, y + 0.8 * h], [x + 0.65 * w, y + 0.8 * h]])
    t = Track(shot=shot.index, track_id=track_id, is_person=True)
    n = max(2, int((shot.end - shot.start) * 5))
    t.dets = [FaceDet(i, shot.start + i * (shot.end - shot.start) / n, x, y, w, h, 0.9, lm) for i in range(n + 1)]
    return t


def _fill(shot: Shot) -> list[Segment]:
    kf = [Keyframe(shot.start, 0.34, 0, 0.316, 1), Keyframe(shot.end, 0.34, 0, 0.316, 1)]
    return [Segment(shot.start, shot.end, "single", [Zone((0, 0, 1, 1), kf, "")], "crop")]


def _fit(shot: Shot) -> list[Segment]:
    kf = [Keyframe(shot.start, 0, 0, 1, 1), Keyframe(shot.end, 0, 0, 1, 1)]
    return [Segment(shot.start, shot.end, "fit", [Zone((0, 0, 1, 1), kf, "picture")], "fit")]


def _framed(shot: Shot, canvas_h: float) -> list[Segment]:
    kf = [Keyframe(shot.start, 0.3, 0.1, 0.4, 0.8), Keyframe(shot.end, 0.3, 0.1, 0.4, 0.8)]
    return [Segment(shot.start, shot.end, "framed", [Zone((0, (1 - canvas_h) / 2, 1, canvas_h), kf, "")], "framed")]


def _setup(durations, people_in=None, screen=(), turns=()):
    """Shots of the given durations; people_in: shot index -> [(x, y, h)]."""
    shots, t = [], 0.0
    for i, d in enumerate(durations):
        shots.append(Shot(i, t, t + d))
        t += d
    tracks = {s.index: [_track(s, k + 1, *f) for k, f in enumerate((people_in or {}).get(s.index, []))] for s in shots}
    person_of = {(s.index, tr.track_id): 10 * s.index + tr.track_id for s in shots for tr in tracks[s.index]}
    analysis = SimpleNamespace(tracks=tracks, fps=5, content=[None] * int(t * 5 + 5), lip_turns=[],
                               turns=[SimpleNamespace(**tu) for tu in turns],
                               people=People(person_of=person_of, speaker_person={"spk": 11} if turns else {}))
    ctx = Context(analysis, TARGET, PIC, lambda s: (s.index in screen, None), lambda s: None)
    return shots, ctx


def test_short_two_person_shot_between_crops_is_cropped_too():
    # Two people too far apart for one crop, nobody talking: the most prominent face stays.
    shots, ctx = _setup([3, 0.7, 3], {1: [(0.15, 0.2, 0.12), (0.7, 0.25, 0.18)]})
    items = [{"shot": s, "options": {"rules": _fit(s) if s.index == 1 else _fill(s), "wide": _fit(s)}} for s in shots]
    picked = [it["options"]["rules"] for it in items]
    notes = [{} for _ in items]
    assert blinks([seg for segs in picked for seg in segs]) == 1
    assert smooth(ctx, items, picked, notes) == 1
    assert kind_of(picked[1]) == "fill" and "smoothed" in notes[1]
    big = ctx.analysis.tracks[1][1]
    assert _keeps(picked[1][0], lambda t: _inner(big, None, t))       # the bigger face is whole
    assert blinks([seg for segs in picked for seg in segs]) == 0


def test_the_person_talking_is_the_one_kept():
    shots, ctx = _setup([3, 0.8, 3], {1: [(0.1, 0.2, 0.15), (0.75, 0.2, 0.15)]},
                        turns=[{"start": 3.0, "end": 3.8, "speaker": "spk"}])
    items = [{"shot": s, "options": {"rules": _fit(s) if s.index == 1 else _fill(s)}} for s in shots]
    picked = [it["options"]["rules"] for it in items]
    smooth(ctx, items, picked)
    talking = ctx.analysis.tracks[1][0]          # person id 11 = the speaker
    assert kind_of(picked[1]) == "fill"
    assert _keeps(picked[1][0], lambda t: _inner(talking, None, t))


def test_a_screen_keeps_the_blurred_background_but_as_big_as_it_fits():
    shots, ctx = _setup([3, 0.9, 3], screen={1})
    small, big = _framed(shots[1], 0.5), _framed(shots[1], 0.7)
    items = [{"shot": s, "options": {"rules": _fill(s)}} for s in shots]
    items[1]["options"] = {"rules": small, "bigger": big}
    picked = [it["options"]["rules"] for it in items]
    smooth(ctx, items, picked)
    assert picked[1] is big


def test_long_shots_keep_their_layout():
    shots, ctx = _setup([3, ISLAND_S + 0.5, 3], {1: [(0.1, 0.2, 0.15), (0.75, 0.2, 0.15)]})
    items = [{"shot": s, "options": {"rules": _fit(s) if s.index == 1 else _fill(s)}} for s in shots]
    picked = [it["options"]["rules"] for it in items]
    assert smooth(ctx, items, picked) == 0 and kind_of(picked[1]) == "bars"


def test_short_crop_between_blurred_shots_joins_them():
    shots, ctx = _setup([3, 0.6, 3])
    items = [{"shot": s, "options": {"rules": _fit(s)}} for s in shots]
    items[1]["options"] = {"rules": _fill(shots[1]), "wide": _fit(shots[1])}
    picked = [it["options"]["rules"] for it in items]
    smooth(ctx, items, picked)
    assert kind_of(picked[1]) == "bars"


def test_every_blink_in_a_long_video_is_fixed():
    durations = [2.5, 0.6] * 8 + [2.5]
    shots, ctx = _setup(durations, {i: [(0.1, 0.2, 0.15), (0.75, 0.2, 0.15)] for i in range(1, 17, 2)})
    items = [{"shot": s, "options": {"rules": _fit(s) if s.index % 2 else _fill(s)}} for s in shots]
    picked = [it["options"]["rules"] for it in items]
    assert smooth(ctx, items, picked) == 8
    assert blinks([seg for segs in picked for seg in segs]) == 0
