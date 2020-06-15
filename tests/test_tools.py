import pytest
from hypothesis import given
from hypothesis.strategies import integers, text, floats

import numpy as np

from segysak import tools

IN_CROP = [5, 10, 25, 35]
IN_LIMITS = [0, 20, 0, 28]


def test_tools_check_crop():
    assert [5, 10, 25, 28] == tools.check_crop(IN_CROP, IN_LIMITS)


@given(floats(-10_000, 10_000), floats(-10_000, 10_000), integers(2, 1000))
def test_halfsample(a, b, n):
    test_array = np.linspace(a, b, n)
    truth_array = np.linspace(a, b, n * 2 - 1)
    assert np.allclose(tools.halfsample(test_array), truth_array)


if __name__ == "__main__":
    c = tools.check_crop(IN_CROP, IN_LIMITS)
    print(c)
