"""Cuts, fades and dissolves on the synthetic clips (exact truth), and camera
motion on the synthetic pan and zoom."""

import numpy as np
import pytest

from eval import synthetic
from reframe.decode import decode
from reframe.motion import estimate
from reframe.probe import find_picture_area, probe
from reframe.shots import detect_transitions


def _frames(name: str, fps: float | None = None, width: int = 320):
    info = probe(str(synthetic.ensure(name)))
    area = find_picture_area(info)
    return info, decode(info, area, fps=fps or info.fps, max_width=width)


@pytest.mark.parametrize("name", ["cuts", "dissolve", "fade", "pan", "zoom"])
def test_transitions_match_truth(name):
    info, frames = _frames(name)
    _, found = detect_transitions(frames, info.duration)
    truth = synthetic.truth(name)["transitions"]
    assert [t.kind for t in found] == [t["type"] for t in truth]
    for f, t in zip(found, truth):
        tol = 0.12
        assert f.start >= t["start"] - tol and f.end <= t["end"] + tol


def test_shots_meet_in_the_middle_of_a_dissolve():
    info, frames = _frames("dissolve")
    shots, found = detect_transitions(frames, info.duration)
    assert len(shots) == 2
    assert shots[1].start == pytest.approx(found[0].middle)


def test_camera_pan_speed():
    info, frames = _frames("pan", fps=5.0, width=640)
    steps = [m for m in estimate(frames, [[] for _ in frames.data], [0]) if m]
    speed = np.mean([m.dx for m in steps]) * frames.fps
    assert speed == pytest.approx(160 / 1280, rel=0.05)
    assert all(m.reliable for m in steps)


def test_camera_zoom():
    info, frames = _frames("zoom", fps=5.0, width=640)
    steps = [m for m in estimate(frames, [[] for _ in frames.data], [0]) if m]
    assert np.prod([m.scale for m in steps]) == pytest.approx(1.5, abs=0.05)
