"""Picture-area detection on synthetic clips (made with ffmpeg; runs in CI)."""

import pytest

from eval import synthetic
from reframe.probe import find_picture_area, probe


@pytest.mark.parametrize("name", ["blurred_bars", "blurred_bars_tall", "black_bars"])
def test_the_real_picture_is_found_inside_bars(name):
    area = find_picture_area(probe(str(synthetic.ensure(name))))
    assert area.__dict__ == synthetic.truth(name)["picture"]


@pytest.mark.parametrize("name", ["cuts", "pan", "zoom", "dissolve"])
def test_pictures_without_bars_stay_whole(name):
    info = probe(str(synthetic.ensure(name)))
    area = find_picture_area(info)
    assert (area.x, area.y, area.w, area.h) == (0, 0, info.width, info.height)
