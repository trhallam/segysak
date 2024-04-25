import pytest
import segyio
import numpy as np

from segysak.segy._segy_core import check_tracefield, sample_range


def test_check_tracefield():
    # check empty list
    assert check_tracefield()

    # check full list
    valid_keys = list(segyio.tracefield.keys.values())
    assert check_tracefield(valid_keys)

    # check full list with bad values
    valid_keys += [2, 182, 400]
    with pytest.raises(ValueError):
        check_tracefield(valid_keys)


@pytest.mark.parametrize("ns0,rate,n", ((0, 1, 100),))
def test_sample_range(ns0, rate, n):
    samples = sample_range(ns0, rate, n)
    assert samples[0] == ns0
    assert samples.size == n
    assert np.allclose(np.diff(samples), rate, atol=1e-5)
