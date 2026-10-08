"""Clip 002: a 9:16 motion-graphics app promo (no people): animated text,
phone mockups and a logo on a light background, changing through animated
transitions rather than cuts. Video not in git (testdata/clip002.mp4)."""

import os
from pathlib import Path

import pytest

from reframe.engine import analyze, plan

ROOT = Path(__file__).resolve().parent.parent
VIDEO = os.environ.get("CLIP002", str(ROOT / "testdata" / "clip002.mp4"))
pytestmark = pytest.mark.skipif(not Path(VIDEO).exists(), reason="clip002.mp4 not available")


@pytest.fixture(scope="module")
def analysis():
    return analyze(VIDEO)


def layout_at(result, t):
    return next(s for s in result["segments"] if s["start"] <= t < s["end"])


def test_no_people_and_no_bars(analysis):
    assert not any(t.is_person for ts in analysis.tracks.values() for t in ts)
    assert (analysis.area.w, analysis.area.h) == (analysis.info.width, analysis.info.height)
    assert analysis.text == []


def test_text_screens_are_cropped_and_phone_screens_fit_in_square(analysis):
    r = plan(analysis, "1:1")
    for t in (6.0, 9.0, 14.0):          # "When? Where?", "7,946", search bar
        assert layout_at(r, t)["layout"] == "single", t
    for t in (24.0, 30.0):               # tall phone mockups
        assert layout_at(r, t)["layout"] == "fit", t


def test_layouts_never_flicker(analysis):
    for ratio in ("16:9", "1:1", "2:3"):
        for s in plan(analysis, ratio)["segments"]:
            assert s["end"] - s["start"] >= 0.99, (ratio, s["start"])
