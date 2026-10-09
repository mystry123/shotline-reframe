"""Speaker-switch discipline, without video (runs in CI): follow the speaker,
never flicker, frame both when switching would be a jump cut, show both when
they talk over each other, and cut (not swoop) between unrelated framings."""

import numpy as np

from reframe.continuity import transitions
from reframe.faces import FaceDet
from reframe.layout import plan_shot
from reframe.shots import Shot
from reframe.tracks import Track

WIDE, PORTRAIT = 16 / 9, 9 / 16
SHOT = Shot(0, 0.0, 12.0)


def _person(track_id: int, cx: float, h: float = 0.16) -> Track:
    w = h * 9 / 16
    x, y = cx - w / 2, 0.2
    lm = np.array([[x + 0.3 * w, y + 0.4 * h], [x + 0.7 * w, y + 0.4 * h], [x + 0.5 * w, y + 0.6 * h],
                   [x + 0.35 * w, y + 0.8 * h], [x + 0.65 * w, y + 0.8 * h]])
    t = Track(shot=0, track_id=track_id, is_person=True)
    t.dets = [FaceDet(i, i / 5, x, y, w, h, 0.9, lm) for i in range(60)]
    return t


def _plan(a, b, speaking):
    return plan_shot(SHOT, [a, b], PORTRAIT, WIDE, 5.0, 0.3, speaking)


def test_follows_the_speaker_and_switches_on_a_new_turn():
    a, b = _person(1, 0.2), _person(2, 0.8)
    segs = _plan(a, b, lambda t: a if t < 6 else b)
    assert [s.zones[0].subject for s in segs] == ["track 1", "track 2"]
    assert abs(segs[1].start - 6.25) < 0.21          # a beat after the new speaker starts


def test_a_short_reply_does_not_move_the_camera():
    a, b = _person(1, 0.2), _person(2, 0.8)
    segs = _plan(a, b, lambda t: b if 5 <= t < 6 else a)   # a one-second "yeah"
    assert [s.zones[0].subject for s in segs] == ["track 1"]


def test_speakers_close_together_get_one_framing_not_jump_cuts():
    a, b = _person(1, 0.465, h=0.12), _person(2, 0.535, h=0.12)
    segs = _plan(a, b, lambda t: a if t < 6 else b)
    assert len(segs) == 1 and segs[0].zones[0].subject == "both"


def test_talking_over_each_other_shows_both():
    a, b = _person(1, 0.2), _person(2, 0.8)
    segs = _plan(a, b, lambda t: (a, b) if 4 <= t < 8 else a)
    assert [s.layout for s in segs] == ["single", "split", "single"]


def test_unrelated_framings_cut_instead_of_easing():
    def seg(start, end, x, w):
        k = {"t": start, "x": x, "y": 0.0, "w": w, "h": 1.0}
        return {"start": start, "end": end, "layout": "single", "zones": [{"keyframes": [k, dict(k, t=end)]}]}
    near = [seg(0, 5, 0.30, 0.32), seg(5, 10, 0.36, 0.40)]     # overlapping, different size: ease
    far = [seg(0, 5, 0.00, 0.32), seg(5, 10, 0.55, 0.40)]      # no overlap, different size: cut
    transitions(near, [])
    transitions(far, [])
    assert near[1]["transition"]["type"] == "ease"
    assert far[1]["transition"]["type"] == "cut"
