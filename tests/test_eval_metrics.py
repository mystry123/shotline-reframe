"""The evaluation's own metrics, on hand-made plans (no video; runs in CI)."""

from types import SimpleNamespace

from eval.metrics import _views, plan_metrics, transition_f1


def _plan(segments, duration=4.0):
    return {"source": {"width": 1000, "height": 1000, "duration": duration, "fps": 25,
                       "picture": {"x": 0, "y": 0, "w": 1000, "h": 1000}},
            "target": "1:1", "settings": {"outputHeight": 1000}, "segments": segments}


def _single(start, end, x, w=0.5):
    kf = {"x": x, "y": 0.0, "w": w, "h": 0.5}
    return {"start": start, "end": end, "layout": "single",
            "zones": [{"canvas": {"left": 0, "top": 0, "width": 1, "height": 1},
                       "keyframes": [{**kf, "t": start}, {**kf, "t": end}]}]}


def _analysis(cut_times=()):
    shots = [SimpleNamespace(index=0, start=0.0, end=4.0)]
    return SimpleNamespace(info=SimpleNamespace(width=1000, height=1000),
                           area=SimpleNamespace(x=0, y=0, w=1000, h=1000),
                           shots=shots + [SimpleNamespace(index=i + 1, start=t, end=4.0) for i, t in enumerate(cut_times)],
                           tracks={i: [] for i in range(len(cut_times) + 1)}, text_in_picture=[])


def test_a_hard_jump_inside_a_shot_is_counted():
    plan = _plan([_single(0, 2, 0.0), _single(2, 4, 0.5)])
    assert plan_metrics(_analysis(), plan)["jumps"] == 1


def test_the_same_jump_at_a_cut_is_not():
    plan = _plan([_single(0, 2, 0.0), _single(2, 4, 0.5)])
    assert plan_metrics(_analysis([2.0]), plan)["jumps"] == 0


def test_an_eased_change_is_not_a_jump():
    second = {**_single(2, 4, 0.5), "transition": {"type": "ease", "duration": 0.5}}
    assert plan_metrics(_analysis(), _plan([_single(0, 2, 0.0), second]))["jumps"] == 0
    xs = [crop[0] for t, _, crop in _views(_plan([_single(0, 2, 0.0), second])) if 2.0 <= t <= 2.5]
    assert xs == sorted(xs) and xs[0] < 0.1 and xs[-1] > 0.4                 # glides across


def test_transition_f1_counts_a_boundary_inside_a_gradual_transition():
    truth = [{"type": "cut", "start": 1.0, "end": 1.0}, {"type": "dissolve", "start": 2.0, "end": 3.0}]
    good = transition_f1(_analysis([1.05, 2.6]), truth)
    assert good["transition_f1"] == 1.0 and good["recall_dissolve"] == 1.0
    missed = transition_f1(_analysis([1.05]), truth)
    assert missed["recall_dissolve"] == 0.0 and missed["transition_f1"] < 1.0
