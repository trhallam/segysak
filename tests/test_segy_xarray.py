import xarray as xr
import pytest
import pathlib

from segysak.segy._xarray import SgyBackendArray, SgyBackendEntrypoint

TESTS_PATH = pathlib.Path(__file__).parent.absolute()
TEST_DATA_SEGYIO = TESTS_PATH / "test-data-segyio"


def split_segyio_kwargs(segyio_kwargs):
    valid_dims = ("iline", "xline", "offset", "cdp")
    valid_kwargs = ("endian",)
    dims = dict()
    other = dict()
    for kwarg, value in segyio_kwargs.items():
        if kwarg in valid_dims:
            dims[kwarg] = value
        elif kwarg in valid_kwargs:
            other[kwarg] = value
    return dims, other


def test_SegyBackendEntrypoint_3d(segyio3d_test_files):
    path, segyio_kwargs = segyio3d_test_files
    dims, segyio_kwargs = split_segyio_kwargs(segyio_kwargs)
    ds = xr.open_dataset(
        path, dim_byte_fields=dims, silent=True, segyio_kwargs=segyio_kwargs
    )
    assert isinstance(ds, xr.Dataset)
    for dim in dims:
        assert dim in ds.sizes

    data = ds.data.compute()
    pass


def test_SegyBackendEntrypoint_3dps(segyio3dps_test_files):
    path, segyio_kwargs = segyio3dps_test_files
    dims, segyio_kwargs = split_segyio_kwargs(segyio_kwargs)
    ds = xr.open_dataset(
        path, dim_byte_fields=dims, silent=True, segyio_kwargs=segyio_kwargs
    )
    assert isinstance(ds, xr.Dataset)
    for dim in dims:
        assert dim in ds.sizes


def test_SegyBackendEntrypoint_3dps_subsel(segyio3dps_test_files):
    path, segyio_kwargs = segyio3dps_test_files
    dims, segyio_kwargs = split_segyio_kwargs(segyio_kwargs)
    ds = xr.open_dataset(
        path, dim_byte_fields=dims, silent=True, segyio_kwargs=segyio_kwargs
    )
    assert isinstance(ds, xr.Dataset)
    for dim in dims:
        assert dim in ds.sizes

    ds_sub = ds.isel(iline=0).data.values
    pass


def test_SegyBackendEntrypoint_variable_nbytes():
    ds = xr.open_dataset(
        TEST_DATA_SEGYIO / "f3.sgy",
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        silent=True,
    )
    for var, v in ds.variables.items():
        assert v.nbytes


def test_SegyBackendEntrypoint_isel():
    ds = xr.open_dataset(
        TEST_DATA_SEGYIO / "f3.sgy",
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        silent=True,
    )
    iline = ds.isel(iline=10).data
    assert iline.sizes["xline"] == 18
    assert iline.sizes["samples"] == 75
    # actually fetch data
    iline_v = iline.data
    assert iline_v.shape == (18, 75)
    # assert iline.transpose("xline", "samples").data.shape == (75, 18)

    xline = ds.isel(xline=10).data
    assert xline.sizes["iline"] == 23
    assert iline.sizes["samples"] == 75
    xline_v = xline.data
    assert xline_v.shape == (23, 75)

    zslc = ds.isel(samples=10).data
    assert zslc.sizes["xline"] == 18
    assert zslc.sizes["iline"] == 23
    zslc_v = zslc.data
    assert zslc_v.shape == (23, 18)

    sub_vol = ds.isel(iline=range(0, 4), xline=range(2, 8), samples=range(20, 50))
    assert sub_vol.sizes["iline"] == 4
    assert sub_vol.sizes["xline"] == 6
    assert sub_vol.sizes["samples"] == 30
    assert sub_vol.data.shape == (4, 6, 30)


def test_SegyBackendEntrypoint_transpose():
    ds = xr.open_dataset(
        TEST_DATA_SEGYIO / "f3.sgy",
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        silent=True,
    )
    iline = ds.isel(iline=10).data
    assert iline.sizes["xline"] == 18
    assert iline.sizes["samples"] == 75
    # actually fetch data
    iline_v = iline.T.data
    assert iline_v.shape == (75, 18)

    xline = ds.isel(xline=10).data
    assert xline.sizes["iline"] == 23
    assert iline.sizes["samples"] == 75
    xline_v = xline.T.data
    assert xline_v.shape == (75, 23)

    zslc = ds.isel(samples=10).data
    assert zslc.sizes["xline"] == 18
    assert zslc.sizes["iline"] == 23
    zslc_v = zslc.T.data
    assert zslc_v.shape == (18, 23)

    sub_vol = ds.isel(iline=range(0, 4), xline=range(2, 8), samples=range(20, 50))
    assert sub_vol.sizes["iline"] == 4
    assert sub_vol.sizes["xline"] == 6
    assert sub_vol.sizes["samples"] == 30
    assert sub_vol.data.shape == (4, 6, 30)


def test_SegyBackendEntrypoint_extra_byte_fields():
    ds = xr.open_dataset(
        TEST_DATA_SEGYIO / "f3.sgy",
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        silent=True,
    )
    assert "cdp_x" in ds
    assert "cdp_y" in ds

    for dim in ("iline", "xline"):
        assert dim in ds["cdp_x"].coords
