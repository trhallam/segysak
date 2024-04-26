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


def test_SegyBackendEntrypoint_variable_nbytes():
    ds = xr.open_dataset(
        TEST_DATA_SEGYIO / "f3.sgy",
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        silent=True,
    )
    for var, v in ds.variables.items():
        assert v.nbytes
    data = ds.isel(iline=10).data.values


def test_SegyBackendEntrypoint_extra_byte_fields():
    ds = xr.open_dataset(
        TEST_DATA_SEGYIO / "f3.sgy",
        dim_byte_fields={"iline": 189, "xline": 193},
        extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        silent=True,
    )
    pass


# TESTS_PATH = pathlib.Path(__file__).parent.absolute()
# TEST_DATA_SEGYIO = TESTS_PATH / "test-data-segyio"
# TEST_DATA_SEGYSAK = TESTS_PATH / "test-data-segysak"
# TEST_DATA_VOLVE = TESTS_PATH / "test-data-volve"


# @pytest.fixture(scope="session")
# def f3_open_dataset():
#     return (
#         TEST_DATA_SEGYIO / "f3.sgy",
#         {"iline": 189, "xline": 193},
#         {"cdp_x": 181, "cdp_y": 185},
#     )


# def test_SegyBackendEntrypoint_3D(f3_open_dataset):
#     filepath, dims, extra = f3_open_dataset
#     ds = xr.open_dataset(
#         filepath,
#         dim_byte_fields=dims,
#         extra_byte_fields=extra,
#     )
#     assert isinstance(ds, xr.Dataset)
#     ds.compute()


# def test_SegyBackendEntrypoint_3D_no_extra(f3_open_dataset):
#     filepath, dims, extra = f3_open_dataset
#     ds = xr.open_dataset(
#         filepath,
#         dim_byte_fields=dims,
#     )
#     assert isinstance(ds, xr.Dataset)
#     ds.compute()


# def test_SegyBackendEntrypoint_3D_no_dims(f3_open_dataset):
#     filepath, dims, extra = f3_open_dataset
#     ds = xr.open_dataset(
#         filepath,
#     )
#     assert isinstance(ds, xr.Dataset)
#     ds.compute()


# def test_SegyBackendEntrypoint_3D_selection(f3_open_dataset):
#     filepath, dims, extra = f3_open_dataset
#     ds = xr.open_dataset(
#         filepath,
#         dim_byte_fields=dims,
#         extra_byte_fields=extra,
#     )
#     assert isinstance(ds, xr.Dataset)

#     # select iline
#     sel = ds.isel(iline=0).compute()
#     assert sel.iline.size == 1

#     # select ilines
#     sel = ds.isel(iline=[0, 2]).compute()
#     assert sel.iline.size == 2

#     # select iline slice
#     sel = ds.isel(iline=slice(0, 3)).compute()
#     assert sel.iline.size == 3

#     # select xline
#     sel = ds.isel(xline=0).compute()
#     assert sel.xline.size == 1

#     # select xlines
#     sel = ds.isel(xline=[0, 2]).compute()
#     assert sel.xline.size == 2

#     # select xline slice
#     sel = ds.isel(xline=slice(0, 3)).compute()
#     assert sel.xline.size == 3

#     # select time slice
#     sel = ds.isel(twt=0).compute()
#     assert sel.twt.size == 1


# if __name__ == "__main__":
#     test_file = "tests/test-data-segysak/f3-withdead.sgy"
#     test_file = "examples/data/volve10r12-full-twt-sub3d.sgy"
#     ds = xr.open_dataset(
#         test_file,
#         dim_byte_fields={"iline": 189, "xline": 193},
#         extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
#     )

#     # print(ds)
#     # sub = ds.isel(xline=0, iline=[0, 3, 7], twt=slice(1, 10, 2))
#     # print(sub.data.values[:, 0])

#     # sub = ds.isel(twt=[50])
#     # print(sub.data.values)

#     print(
#         ds.isel(
#             twt=[100], xline=slice(0, 10000, 1), iline=slice(0, 10000, 5)
#         ).data.values
#     )
