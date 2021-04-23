import pytest
import segyio

from segysak.segy._segy_core import check_tracefield


def test_check_tracefield():
    valid_keys = list(segyio.tracefield.keys.values())
    assert check_tracefield(valid_keys)

    valid_keys += [2, 182, 400]
    with pytest.raises(ValueError):
        check_tracefield(valid_keys)
