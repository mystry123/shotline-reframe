"""Clip 001: a 9:16 export of a letterboxed 16:9 interview (multi-camera, two
people, painting and ornaments behind them, captions burned into the bars).

The video isn't in git (third-party footage); put it at testdata/clip001.mp4
or set CLIP001. The transcript is committed.
"""

import json
import os
from pathlib import Path

import pytest

from reframe.engine import analyze, plan

ROOT = Path(__file__).resolve().parent.parent
VIDEO = os.environ.get("CLIP001", str(ROOT / "testdata" / "clip001.mp4"))
WORDS = json.loads((ROOT / "testdata" / "clip001.transcript.json").read_text())["words"]

pytestmark = pytest.mark.skipif(not Path(VIDEO).exists(), reason="clip001.mp4 not available")


@pytest.fixture(scope="module")
def analysis():
    return analyze(VIDEO, WORDS)


def test_finds_the_real_picture_inside_the_bars(analysis):
    a = analysis.area
    assert (a.x, a.w) == (0, 360)
    assert 215 <= a.y <= 223 and 198 <= a.h <= 206
    assert abs(a.w / a.h - 16 / 9) < 0.03


def test_cuts_match_the_edit(analysis):
    cuts = [round(s.start, 1) for s in analysis.shots[1:]]
    assert cuts == pytest.approx([13.9, 15.7, 17.0, 21.0, 27.2, 33.9], abs=0.15)


def test_only_real_people_are_subjects(analysis):
    people = {i: sorted(t.median("cx") for t in ts if t.is_person) for i, ts in analysis.tracks.items()}
    # Painting figures, the shelf ornament and a hand are rejected.
    expected = {0: [0.49], 1: [0.53], 2: [0.49], 3: [0.56], 4: [0.36], 5: [0.25, 0.82], 6: [0.53]}
    assert people.keys() == expected.keys()
    for shot, xs in expected.items():
        assert people[shot] == pytest.approx(xs, abs=0.03), f"shot {shot}"


def test_two_people_across_all_shots_and_voices_mapped(analysis):
    assert len(set(analysis.people.person_of.values())) == 2
    guest = analysis.people.person_of[(0, analysis.tracks[0][0].track_id)]
    assert analysis.people.speaker_person["speaker_0"] == guest
    assert analysis.people.speaker_person["speaker_1"] != guest
    assert "speaker_2" not in analysis.people.speaker_person  # the lone "Yeah" is ignored


def test_16x9_uses_the_whole_picture(analysis):
    result = plan(analysis, "16:9")
    assert {s["layout"] for s in result["segments"]} == {"full"}


@pytest.mark.parametrize("ratio", ["1:1", "2:3"])
def test_portrait_targets_hold_still_and_step_at_cuts(analysis, ratio):
    result = plan(analysis, ratio)
    segs = result["segments"]
    assert [s["layout"] for s in segs] == ["single"] * 7
    assert "over-the-shoulder" in segs[5]["reason"]
    for s in segs:
        kfs = s["zones"][0]["keyframes"]
        assert len({(k["w"], k["h"]) for k in kfs}) == 1          # no zoom change inside a shot
        assert kfs[0]["t"] == s["start"] and kfs[-1]["t"] == s["end"]  # framing changes only at cuts


def test_finds_the_burned_in_title_and_captions(analysis):
    found = {t.id: t for t in analysis.text}
    assert set(found) == {"top-title", "bottom-captions"}
    assert found["top-title"].kind == "title" and found["top-title"].active < 0.15   # first ~3 s only
    assert found["bottom-captions"].kind == "captions" and found["bottom-captions"].active > 0.7
    assert all(t.keyable for t in analysis.text)


def test_text_is_dropped_unless_the_user_keeps_it(analysis):
    dropped = plan(analysis, "1:1")
    assert [t["id"] for t in dropped["text"]] == ["top-title", "bottom-captions"]
    assert dropped["overlays"] == []
    kept = plan(analysis, "1:1", keep_text=["bottom-captions"])
    assert [o["id"] for o in kept["overlays"]] == ["bottom-captions"]
    p = kept["overlays"][0]["placement"]
    assert 0 <= p["left"] and p["left"] + p["width"] <= 1 and p["top"] + p["height"] <= 0.9  # above platform UI
