import xarray as xr
import pytest
import pathlib

from segysak.segy._xarray import SgyBackendArray, SgyBackendEntrypoint

TESTS_PATH = pathlib.Path(__file__).parent.absolute()
TEST_DATA_SEGYIO = TESTS_PATH / "test-data-segyio"
TEST_DATA_SEGYSAK = TESTS_PATH / "test-data-segysak"


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


@pytest.mark.parametrize(
    "file,shape",
    (
        (TEST_DATA_SEGYIO / "f3.sgy", (23, 18, 75)),
        (TEST_DATA_SEGYSAK / "f3-withdead.sgy", (23, 18, 75)),
    ),
)
def test_SegyBackendEntrypoint_isel(file, shape):
    nil, nxl, ns = shape
    ds = xr.open_dataset(
        file,
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        silent=True,
    )
    iline = ds.isel(iline=10).data
    assert iline.sizes["xline"] == nxl
    assert iline.sizes["samples"] == ns
    # actually fetch data
    iline_v = iline.data
    assert iline_v.shape == (nxl, ns)

    xline = ds.isel(xline=10).data
    assert xline.sizes["iline"] == nil
    assert iline.sizes["samples"] == ns
    xline_v = xline.data
    assert xline_v.shape == (nil, ns)

    zslc = ds.isel(samples=10).data
    assert zslc.sizes["xline"] == nxl
    assert zslc.sizes["iline"] == nil
    zslc_v = zslc.data
    assert zslc_v.shape == (nil, nxl)

    sub_vol = ds.isel(iline=range(0, 4), xline=range(2, 8), samples=range(20, 50))
    assert sub_vol.sizes["iline"] == 4
    assert sub_vol.sizes["xline"] == 6
    assert sub_vol.sizes["samples"] == 30
    assert sub_vol.data.shape == (4, 6, 30)


@pytest.mark.parametrize(
    "file,shape",
    (
        (TEST_DATA_SEGYIO / "f3.sgy", (23, 18, 75)),
        (TEST_DATA_SEGYSAK / "f3-withdead.sgy", (23, 18, 75)),
    ),
)
def test_SegyBackendEntrypoint_transpose(file, shape):
    nil, nxl, ns = shape
    ds = xr.open_dataset(
        file,
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        silent=True,
    )
    iline = ds.isel(iline=10).data
    assert iline.sizes["xline"] == nxl
    assert iline.sizes["samples"] == ns
    # actually fetch data
    iline_v = iline.T.data
    assert iline_v.shape == (ns, nxl)

    xline = ds.isel(xline=10).data
    assert xline.sizes["iline"] == nil
    assert iline.sizes["samples"] == ns
    xline_v = xline.T.data
    assert xline_v.shape == (ns, nil)

    zslc = ds.isel(samples=10).data
    assert zslc.sizes["xline"] == nxl
    assert zslc.sizes["iline"] == nil
    zslc_v = zslc.T.data
    assert zslc_v.shape == (nxl, nil)

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
