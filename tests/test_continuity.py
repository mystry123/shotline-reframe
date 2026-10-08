"""Continuity without video (runs in CI): setups, one geometry per setup,
and how framing changes inside a shot are marked."""

from types import SimpleNamespace

import numpy as np
import pytest

from reframe.continuity import setups, transitions, unify
from reframe.faces import FaceDet
from reframe.layout import Keyframe, Segment, Zone
from reframe.shots import Shot
from reframe.speakers import People
from reframe.tracks import Track


def _track(shot: int, track_id: int, x: float, y: float, h: float) -> Track:
    w = h * 0.8
    lm = np.array([[x + 0.3 * w, y + 0.4 * h], [x + 0.7 * w, y + 0.4 * h], [x + 0.5 * w, y + 0.6 * h],
                   [x + 0.35 * w, y + 0.8 * h], [x + 0.65 * w, y + 0.8 * h]])
    t = Track(shot=shot, track_id=track_id, is_person=True)
    t.dets = [FaceDet(i, i / 5, x, y, w, h, 0.9, lm) for i in range(5)]
    return t


def _analysis(faces: list[tuple], person_ids: list[int]):
    """One single-person shot per face (x, y, h); person ids per shot."""
    tracks = {i: [_track(i, 1, *f)] for i, f in enumerate(faces)}
    people = People(person_of={(i, 1): pid for i, pid in enumerate(person_ids)})
    items = [{"shot": Shot(i, float(i), float(i + 1)), "options": {}} for i in range(len(faces))]
    return SimpleNamespace(tracks=tracks, people=people), items


def test_jump_cuts_of_one_camera_are_one_setup_other_people_are_not():
    a, items = _analysis([(0.40, 0.15, 0.12), (0.42, 0.16, 0.13), (0.10, 0.30, 0.10), (0.41, 0.14, 0.12)],
                         [1, 1, 2, 1])
    assert setups(a, items) == [[0, 1, 3], [2]]


def test_same_person_from_another_camera_position_is_another_setup():
    a, items = _analysis([(0.40, 0.15, 0.12), (0.40, 0.15, 0.30)], [1, 1])   # much closer shot
    assert setups(a, items) == [[0], [1]]


def _framed(y1: float) -> list[Segment]:
    return [Segment(0, 1, "framed", [Zone((0.3, 0, 0.4, 1), [Keyframe(0, 0.2, 0.1, 0.6, y1 - 0.1)])], "")]


def test_a_setup_gets_one_framing_that_contains_every_shot_s_framing():
    picked = [_framed(0.6), _framed(0.8), _framed(0.7)]
    unify([[0, 1, 2]], picked, 16 / 9, 9 / 16)
    boxes = {(round(k.x, 6), round(k.y, 6), round(k.w, 6), round(k.h, 6)) for segs in picked for k in segs[0].zones[0].keyframes}
    assert boxes == {(0.2, 0.1, 0.6, 0.7)}                               # union: the tallest
    assert len({segs[0].zones[0].canvas for segs in picked}) == 1


def test_different_layouts_in_a_setup_are_left_alone():
    fit = [Segment(0, 1, "fit", [Zone((0, 0, 1, 1), [Keyframe(0, 0, 0, 1, 1)])], "")]
    picked = [_framed(0.6), fit]
    unify([[0, 1]], picked, 16 / 9, 9 / 16)
    assert picked[0][0].zones[0].keyframes[0].h == pytest.approx(0.5)


def _seg(start, end, layout="single", x=0.3, w=0.4, zones=1):
    kf = {"t": start, "x": x, "y": 0.0, "w": w, "h": 1.0}
    return {"start": start, "end": end, "layout": layout,
            "zones": [{"canvas": {}, "keyframes": [kf, {**kf, "t": end}]}] * zones}


def test_changes_inside_a_shot_ease_and_cuts_stay_cuts():
    segs = [_seg(0, 4, "fit", 0, 1), _seg(4, 8), _seg(8, 12, "fit", 0, 1)]
    transitions(segs, cut_times=[8.0])
    assert segs[1]["transition"]["type"] == "ease" and 0.1 <= segs[1]["transition"]["duration"] <= 0.5
    assert "transition" not in segs[2]                                   # at a real cut


def test_a_long_sideways_move_cuts_instead_of_whip_panning():
    segs = [_seg(0, 4, x=0.0, w=0.3), _seg(4, 8, x=0.6, w=0.3)]
    transitions(segs, cut_times=[])
    assert segs[1]["transition"] == {"type": "cut"}


def test_split_screens_switch_by_cutting():
    segs = [_seg(0, 4), _seg(4, 8, "split", zones=2)]
    transitions(segs, cut_times=[])
    assert segs[1]["transition"] == {"type": "cut"}
