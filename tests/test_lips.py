"""Who is talking in two-shots, from mouth motion (clip008: a 720p interview,
host left, guest right; transcript speakers labelled by hand). Local only:
the video stays out of git."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "testdata" / "clip008.mp4"
WORDS = ROOT / "testdata" / "clip008.transcript.json"

pytestmark = pytest.mark.skipif(not (VIDEO.exists() and WORDS.exists()), reason="needs testdata/clip008 (local only)")


@pytest.fixture(scope="module")
def analysis():
    from reframe.engine import analyze
    return analyze(str(VIDEO), json.loads(WORDS.read_text())["words"])


def test_mouth_motion_names_the_speaker_in_two_shots(analysis):
    decided = [lt for lt in analysis.lip_turns if lt.track_id is not None]
    assert len(decided) >= 8
    for lt in decided:
        track = next(t for t in analysis.tracks[lt.shot] if t.track_id == lt.track_id)
        side = "left" if track.median("cx") < 0.5 else "right"
        assert side == ("left" if lt.speaker == "speaker_1" else "right"), lt


def test_logo_is_a_watermark_not_captions(analysis):
    from reframe.engine import _watermarks, picture_captions
    marks = _watermarks(analysis)
    assert len(marks) == 1 and marks[0][0] > 0.7 and marks[0][1] > 0.85     # JOHNANDERSON, bottom right
    # Shots with no on-screen graphics have no captions to keep.
    assert picture_captions(analysis, analysis.shots[1]) is None
