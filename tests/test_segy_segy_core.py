import pytest
import segyio

from segysak.segy._segy_core import check_tracefield


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
