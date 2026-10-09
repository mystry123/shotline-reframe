"""L1-optimal crop paths: hold through jitter, move when the subject moves,
respect the speed limit and hard bounds; and a walking person followed by
their body when the face is lost."""

import numpy as np
import pytest

from eval import synthetic
from reframe.paths import l1_path

FPS = 10
T = np.arange(0, 20, 1 / FPS)
W, MARGIN = 0.4, 0.12


def _path(cx, bounds=(0.0, 0.6), speed=0.25):
    return l1_path(cx - W + MARGIN, cx - MARGIN, bounds, FPS, speed)


def test_still_subject_with_jitter_holds():
    cx = 0.5 + np.random.default_rng(0).normal(0, 0.01, len(T))
    assert np.ptp(_path(cx)) < 1e-3


def test_moves_when_the_subject_moves_and_holds_otherwise():
    cx = np.where(T < 5, 0.3, np.where(T < 8, 0.3 + 0.1 * (T - 5), 0.6))
    p = _path(cx)
    moving = np.abs(np.diff(p)) * FPS > 1e-3
    starts = np.flatnonzero(np.diff(moving.astype(int)) == 1) / FPS
    assert len(starts) == 1 and 3.5 <= starts[0] <= 5.5     # one move, around when the walk starts
    assert np.ptp(p[T < 3]) < 1e-3 and np.ptp(p[T > 10]) < 1e-3
    inside = (cx - p >= MARGIN - 1e-3) & (p + W - cx >= MARGIN - 1e-3)
    assert inside.all()


def test_speed_limit_and_bounds():
    cx = np.where(T < 5, 0.2, 0.8)
    p = _path(cx, bounds=(0.05, 0.5))
    assert np.abs(np.diff(p)).max() * FPS <= 0.25 + 1e-6
    assert p.min() >= 0.05 - 1e-9 and p.max() <= 0.5 + 1e-9


def test_per_sample_bounds():
    cx = np.full(len(T), 0.5)
    bounds = [(0.0, 0.1) if t < 10 else (0.3, 0.6) for t in T]
    p = l1_path(cx - W + MARGIN, cx - MARGIN, bounds, FPS, 1.0)
    assert all(lo - 1e-9 <= x <= hi + 1e-9 for x, (lo, hi) in zip(p, bounds))


needs_testdata = pytest.mark.skipif("real_moving_subject" not in synthetic.local_available(),
                                    reason="needs testdata/clip006.mp4 (local only)")


@needs_testdata
def test_walking_person_followed_when_face_is_lost():
    from eval.metrics import plan_metrics
    from reframe.engine import analyze, plan
    a = analyze(str(synthetic.ensure_local("real_moving_subject")), None)
    people = [t for t in a.tracks[0] if t.is_person]
    assert len(people) == 1 and people[0].moving
    assert any(d.estimated for d in people[0].dets)       # the looking-down stretch comes from the body
    for ratio in ("9:16", "1:1"):
        m = plan_metrics(a, plan(a, ratio))
        assert m["face_cut_rate"] == 0.0 and m["jumps"] == 0
