"""Framing guards, without video (these run in CI): the zoom limit, how much
of the person a framing keeps, and which alternatives the director is offered.

The scene: a portrait (9:16) talking shot going to 16:9. The person's face is
near the top, their body (arms and hands on a desk) reaches far down the
frame, like the over-cropped shot that motivated these checks.
"""

import numpy as np
import pytest

from reframe.faces import FaceDet
from reframe.layout import (SUBJECT_KEEP, Keyframe, Segment, Zone, crop_height, share_kept,
                            shot_alternatives, upper_body)
from reframe.options import OUTPUT_HEIGHTS, Options
from reframe.shots import Shot
from reframe.tracks import Track

PORTRAIT, WIDE = 9 / 16, 16 / 9


def _person(face=(0.38, 0.14, 0.24, 0.13), body=(0.05, 0.10, 0.95, 0.92), with_body=True) -> Track:
    x, y, w, h = face
    lm = np.array([[x + 0.3 * w, y + 0.4 * h], [x + 0.7 * w, y + 0.4 * h], [x + 0.5 * w, y + 0.6 * h],
                   [x + 0.35 * w, y + 0.8 * h], [x + 0.65 * w, y + 0.8 * h]])
    t = Track(shot=0, track_id=1, is_person=True)
    t.dets = [FaceDet(i, i / 5, x, y, w, h, 0.9, lm) for i in range(20)]
    if with_body:
        t.body_boxes = [body] * 4
    return t


def _alternatives(zoom="balanced", person=None, pic_px_h=1280):
    shot = Shot(0, 0.0, 4.0)
    rules = [Segment(0, 4, "fit", [Zone((0, 0, 1, 1), [Keyframe(0, 0, 0, 1, 1), Keyframe(4, 0, 0, 1, 1)])], "rules")]
    options = Options(zoom=zoom)
    return shot_alternatives(shot, [person or _person()], rules, WIDE, PORTRAIT, 5.0,
                             options.min_crop_height(pic_px_h), None, strict=zoom != "tight")


def test_zoom_settings_are_ordered_and_tight_always_allows_2x():
    for h in (120, 202, 360, 720, 1080, 1280, 1920, 2160):
        for out in OUTPUT_HEIGHTS:
            sharp, balanced, tight = (Options(zoom=z, output_height=out).min_crop_height(h)
                                      for z in ("sharp", "balanced", "tight"))
            assert tight <= balanced <= sharp <= 1.0
            assert tight <= 0.5


def test_measurements():
    seg = Segment(0, 1, "single", [Zone((0, 0, 1, 1), [Keyframe(0, 0.0, 0.2, 1.0, 0.3)])], "")
    assert crop_height(seg) == pytest.approx(0.3)
    assert share_kept(seg, (0.0, 0.2, 1.0, 0.5)) == pytest.approx(1.0)
    assert share_kept(seg, (0.0, 0.2, 1.0, 0.8)) == pytest.approx(0.5)
    narrow = Segment(0, 1, "single", [Zone((0, 0, 1, 1), [Keyframe(0, 0.4, 0.0, 0.3, 1.0)])], "")
    assert share_kept(narrow, (0.2, 0.1, 0.8, 0.7)) == pytest.approx(1.0)          # sides trimmed: fine
    assert share_kept(narrow, (0.2, 0.1, 0.8, 0.7), "xy") == pytest.approx(0.5)
    framed = Segment(0, 1, "framed", [Zone((0.3, 0, 0.4, 0.5), [Keyframe(0, 0, 0, 0.5, 0.4)])], "")
    assert crop_height(framed) == pytest.approx(0.8)     # half the canvas: counts double


def test_upper_body_uses_the_measured_body_down_to_the_waist():
    person = _person()
    x0, y0, x1, y1 = upper_body(person)
    assert (x0, y0, x1) == pytest.approx((0.05, 0.10, 0.95))
    assert y1 == pytest.approx(0.14 + 4.5 * 0.13)       # waist, not the floor


def test_no_body_seen_falls_back_to_an_estimate_around_the_face():
    x0, y0, x1, y1 = upper_body(_person(with_body=False))
    assert x0 < 0.38 and x1 > 0.62 and y0 < 0.14 and y1 > 0.5


def test_balanced_never_offers_a_framing_that_cuts_the_person_or_zooms_too_far():
    opts = _alternatives("balanced")
    assert "close" not in opts                           # the full-width 16:9 close-up of a portrait
    limit = Options(zoom="balanced").min_crop_height(1280)
    for name, segs in opts.items():
        if name == "rules":
            continue
        for seg in segs:
            assert seg.info["person_kept"] >= SUBJECT_KEEP
            assert seg.info["crop_h"] >= limit - 0.011
    assert "body" in opts                                # the whole person, hands included
    assert any(segs[0].layout == "fit" for segs in opts.values())   # and the whole picture (here: the rules)


def test_tight_is_the_opt_in_to_close_ups():
    assert "close" in _alternatives("tight")


def test_every_alternative_is_measured_for_the_director():
    for segs in _alternatives("balanced").values():
        assert {"crop_h", "person_kept"} <= set(segs[0].info)


def test_low_resolution_sources_get_no_zoomed_crops():
    # 360 px tall: any crop would be upscaled; only views of the whole picture height remain.
    for name, segs in _alternatives("balanced", pic_px_h=360).items():
        if name != "rules":
            assert all(seg.info["crop_h"] >= 0.99 for seg in segs)
