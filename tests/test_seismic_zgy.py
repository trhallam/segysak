from numpy.lib.arraysetops import isin
import pytest

import xarray as xr

from segysak.openzgy import zgy_loader


def test_zgy_loader(ozgy_files):
    file = ozgy_files
    ds = zgy_loader(file)
    assert isinstance(ds, xr.Dataset)
