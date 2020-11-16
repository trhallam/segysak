import pytest

import xarray as xr

from segysak.openzgy import zgy_loader, zgy_writer


def test_zgy_loader(ozgy_files):
    file = ozgy_files
    ds = zgy_loader(file)
    assert isinstance(ds, xr.Dataset)


def test_zgy_writer(temp_dir, zeros3d):
    temp_file = temp_dir / "tempzgy.zgy"
    zgy_writer(zeros3d, temp_file)
    assert temp_file.exists()
    ds = zgy_loader(temp_file)
    assert isinstance(ds, xr.Dataset)
