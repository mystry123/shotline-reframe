"""Clip 003: a 9:16 talking head with burned-in captions in the picture, slides
and a screen recording (jump cuts from one camera). Local only: the video
isn't in git; put it at testdata/clip003.mp4 or set CLIP003."""

import os
from pathlib import Path

import pytest

from reframe.engine import analyze, plan
from reframe.options import Options

ROOT = Path(__file__).resolve().parent.parent
VIDEO = os.environ.get("CLIP003", str(ROOT / "testdata" / "clip003.mp4"))
pytestmark = pytest.mark.skipif(not Path(VIDEO).exists(), reason="clip003.mp4 not available")


@pytest.fixture(scope="module")
def analysis():
    return analyze(VIDEO, None)


def test_jump_cuts_of_the_talking_head_get_one_framing(analysis):
    p = plan(analysis, "16:9", options=Options())
    by_setup = {}
    for shot, seg in zip(p["shots"], p["segments"]):
        k = seg["zones"][0]["keyframes"][0]
        by_setup.setdefault(shot["setup"], set()).add((seg["layout"], round(k["w"], 3), round(k["h"], 3)))
    biggest = max(by_setup.values(), key=len)
    assert max(len([s for s in p["shots"] if s["setup"] == k]) for k in by_setup) >= 8   # the talking head
    assert all(len(v) == 1 for v in by_setup.values()), biggest


def test_replace_leaves_the_captions_out_of_talking_head_shots(analysis):
    keep = plan(analysis, "16:9", options=Options())
    replace = plan(analysis, "16:9", options=Options(), captions_mode="replace")
    assert keep["pictureCaptions"] and replace["captions"]["replaced"]
    band_top = 0.6 * analysis.area.h / analysis.info.height     # captions start at 60% of the picture
    for shot, seg in zip(replace["shots"], replace["segments"]):
        if any(a <= shot["start"] < b for a, b in replace["captions"]["replaced"]):
            k = seg["zones"][0]["keyframes"][0]
            assert k["y"] + k["h"] <= band_top + 0.01
