import xarray as xr
import pytest
import pathlib

from segysak.segy._segy_xarray import SegyBackendArray, SegyBackendEntrypoint
from segysak.segy import segy_header_scrape

TESTS_PATH = pathlib.Path(__file__).parent.absolute()
TEST_DATA_SEGYIO = TESTS_PATH / "test-data-segyio"
TEST_DATA_SEGYSAK = TESTS_PATH / "test-data-segysak"
TEST_DATA_VOLVE = TESTS_PATH / "test-data-volve"

def test_ZgyBackendEntrypoint_3D(ozgy_files):
    ds = xr.open_dataset(ozgy_files)
    assert isinstance(ds, xr.Dataset)
    ds.compute()

def test_ZgyBackendEntrypoint_3D_no_extra(ozgy_files):
    ds = xr.open_dataset(ozgy_files)
    assert isinstance(ds, xr.Dataset)
    ds.compute()

def test_ZgyBackendEntrypoint_3D_no_dims(ozgy_files):
    ds = xr.open_dataset(ozgy_files)
    assert isinstance(ds, xr.Dataset)
    ds.compute()

def test_ZgyBackendEntrypoint_3D_selection(ozgy_files):
    ds = xr.open_dataset(ozgy_files)
    assert isinstance(ds, xr.Dataset)

    # select iline
    sel = ds.isel(iline=0).compute()
    assert sel.iline.size == 1

    # select ilines
    sel = ds.isel(iline=[0, 2]).compute()
    assert sel.iline.size == 2

    # select iline slice
    sel = ds.isel(iline=slice(0, 3)).compute()
    assert sel.iline.size == 3

    # select xline
    sel = ds.isel(xline=0).compute()
    assert sel.xline.size == 1

    # select xlines
    sel = ds.isel(xline=[0, 2]).compute()
    assert sel.xline.size == 2

    # select xline slice
    sel = ds.isel(xline=slice(0, 3)).compute()
    assert sel.xline.size == 3

    # select time slice
    sel = ds.isel(twt=0).compute()
    assert sel.twt.size == 1


if __name__ == "__main__":
    test_file = "tests/test-data-ozgy/test_3d_float.zgy"
    ds = xr.open_dataset(test_file)

    print(ds)
    sub = ds.isel(xline=0, iline=slice(1, 10, 2), twt=slice(1, 10, 2))
    print(sub.data.values[0, 0])

    sub = ds.isel(twt=[5])
    print(sub.data.values)