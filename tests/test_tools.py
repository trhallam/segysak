import pytest
from hypothesis import given
from hypothesis.strategies import integers, text, floats

import numpy as np

from segysak import tools

IN_CROP = [5, 10, 25, 35]
IN_LIMITS = [0, 20, 0, 28]


def test_tools_check_crop():
    assert [5, 10, 25, 28] == tools.check_crop(IN_CROP, IN_LIMITS)


def test_tools_check_crop_errors():
    with pytest.raises(ValueError):
        tools.check_crop([100, 0, 0, 100], [0, 100, 0, 100])
    with pytest.raises(ValueError):
        tools.check_crop([0, 100, 0, 100], [100, 0, 0, 100])
    with pytest.raises(ValueError):
        tools.check_crop(
            [0, 100, 100, 0],
            [0, 100, 0, 100],
        )

    with pytest.raises(ValueError):
        tools.check_crop([10, 100, 10, 100], [0, 9, 0, 9])


@given(floats(-10_000, 10_000), floats(-10_000, 10_000), integers(2, 1000))
def test_halfsample(a, b, n):
    test_array = np.linspace(a, b, n)
    truth_array = np.linspace(a, b, n * 2 - 1)
    assert np.allclose(tools.halfsample(test_array), truth_array)


def test_check_zcrop():
    assert [0, 1000] == tools.check_zcrop([0, 1000], [0, 2000])


def test_fix_bad_chars():
    text = "abcdefg�"
    assert tools.fix_bad_chars(text) == "abcdefg#"


def test_get_datetime():
    assert isinstance(tools._get_datetime(), tuple)


def test_get_userid():
    assert isinstance(tools._get_userid(), str)


if __name__ == "__main__":
    c = tools.check_crop(IN_CROP, IN_LIMITS)
    print(c)
