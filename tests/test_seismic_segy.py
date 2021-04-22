import pytest
from pytest_cases import fixture_ref, parametrize

from .fixtures_seisnc import *

import pathlib

from hypothesis import given, settings
from hypothesis.strategies import integers

import numpy as np
import pandas as pd
import xarray as xr

from segysak.segy import (
    segy_loader,
    segy_converter,
    ncdf2segy,
    create_default_texthead,
    put_segy_texthead,
    get_segy_texthead,
    segy_header_scan,
    segy_header_scrape,
    segy_bin_scrape,
    segy_writer,
    well_known_byte_locs,
    output_byte_locs,
)

from segysak import open_seisnc

# from test_fixtures import *
from .fixtures_segy import TEST_SEGY_SIZE

GOOD_HEADER = (
    "a" * 3200,
    {n: "a" * 65 for n in range(1, 41)},
    bytes("a" * 3200, "utf8"),
)
BAD_HEADER = (
    "b" * 4000,
    {n: "b" * 85 for n in range(1, 41)},
    bytes("b" * 4000, "utf8"),
    {"not_a_num": "b" * 65},
)


def test_segy_fixtures(temp_dir, temp_segy):
    assert temp_segy.exists()


@pytest.fixture()
def cropping_limits():
    min_il = TEST_SEGY_SIZE // 4
    max_il = TEST_SEGY_SIZE // 2 + min_il
    min_xl = TEST_SEGY_SIZE + TEST_SEGY_SIZE // 4
    max_xl = TEST_SEGY_SIZE // 2 + min_xl
    crop = [min_il, max_il, min_xl, max_xl]

    min_z = TEST_SEGY_SIZE // 4
    max_z = TEST_SEGY_SIZE // 2 + min_z
    cropz = [min_z, max_z]
    return crop, cropz


def test_well_known_byte_locs_fail():
    with pytest.raises(ValueError):
        well_known_byte_locs("gibberish")


@pytest.mark.parametrize("header", GOOD_HEADER, ids=["str", "dict", "bytes"])
def test_put_segy_texthead_ok(temp_dir, temp_segy, header):
    put_segy_texthead(temp_segy, header)


@pytest.mark.parametrize("header", BAD_HEADER, ids=["str", "adict", "bytes", "bdict"])
def test_put_segy_texthead_notok(temp_dir, temp_segy, header):
    with pytest.warns(UserWarning):
        put_segy_texthead(temp_segy, header)


def test_segy_header_scan(temp_dir, temp_segy):
    head = segy_header_scan(temp_segy)
    scanned = head.nscan
    assert scanned == TEST_SEGY_SIZE ** 2
    # assert isinstance(head, dict) now a dataframe
    head = segy_header_scan(temp_segy, max_traces_scan=TEST_SEGY_SIZE, silent=True)
    scanned = head.nscan
    assert scanned == TEST_SEGY_SIZE
    # assert isinstance(head, dict) now a dataframe

    with pytest.raises(ValueError):
        header = segy_header_scan("", max_traces_scan=23.9, silent=True)


def test_segy_header_scan_all(temp_dir, temp_segy):
    _ = segy_header_scan(temp_segy, max_traces_scan=0, silent=True)
    scanned = _.nscan
    assert scanned == TEST_SEGY_SIZE ** 2
    _ = segy_header_scan(temp_segy, max_traces_scan="all", silent=True)
    scanned = _.nscan
    assert scanned == TEST_SEGY_SIZE ** 2


def test_segy_bin_scrape(temp_dir, temp_segy):
    assert isinstance(segy_bin_scrape(temp_segy), dict)


def test_segy_header_scrape(temp_dir, temp_segy):
    header = segy_header_scrape(temp_segy, silent=True)
    assert isinstance(header, pd.DataFrame)
    assert header.shape == (TEST_SEGY_SIZE ** 2, 89)


# def test_segy2ncdf_3D(temp_dir, temp_segy):
#     temp_seisnc = temp_segy.with_suffix(".SEISNC")
#     segy_loader(temp_segy, temp_seisnc, silent=True)


# def test_segy2ncdf_3D_cropH(temp_dir, temp_segy, cropping_limits):
#     temp_seisnc = temp_segy.with_suffix(".SEISNC")

#     crop, _ = cropping_limits
#     min_il, max_il, min_xl, max_xl = crop

#     segy2ncdf(temp_segy, temp_seisnc, silent=True, crop=crop)

#     ds = xr.open_dataset(temp_seisnc)
#     assert tuple(ds.dims.values()) == (
#         max_il - min_il + 1,
#         max_xl - min_xl + 1,
#         TEST_SEGY_SIZE,
#     )


# def test_segy2ncdf_3D_cropZ(temp_dir, temp_segy, cropping_limits):
#     temp_seisnc = temp_segy.with_suffix(".SEISNC")

#     _, cropz = cropping_limits
#     min_z, max_z = cropz

#     segy2ncdf(temp_segy, temp_seisnc, silent=True, zcrop=[min_z, max_z])

#     ds = xr.open_dataset(temp_seisnc)
#     assert tuple(ds.dims.values())[2] == max_z - min_z + 1


# def test_segy2ncdf_3D_crop(temp_dir, temp_segy, cropping_limits):
#     temp_seisnc = temp_segy.with_suffix(".SEISNC")
#     crop, cropz = cropping_limits
#     min_il, max_il, min_xl, max_xl = crop
#     min_z, max_z = cropz

#     segy2ncdf(temp_segy, temp_seisnc, silent=True, crop=crop, zcrop=cropz)

#     ds = xr.open_dataset(temp_seisnc)
#     assert tuple(ds.dims.values()) == (
#         max_il - min_il + 1,
#         max_xl - min_xl + 1,
#         max_z - min_z + 1,
#     )


def test_segyiotests_2ncdf(temp_dir, segyio3d_test_files):
    file, segyio_kwargs = segyio3d_test_files
    seisnc = temp_dir / file.with_suffix(".siesnc").name
    segy_converter(str(file), ncfile=seisnc, silent=True, **segyio_kwargs)
    ds = open_seisnc(seisnc)
    assert isinstance(ds, xr.Dataset)


def test_segyiotests_2ds(segyio3d_test_files):
    file, segyio_kwargs = segyio3d_test_files
    ds = segy_loader(str(file), silent=True, **segyio_kwargs)
    assert isinstance(ds, xr.Dataset)


def test_segyiotests_2ds_wheaderscrap(segyio3d_test_files):
    file, segyio_kwargs = segyio3d_test_files
    scrape_args = dict()
    try:
        scrape_args["endian"] = segyio_kwargs["endian"]
    except KeyError:
        pass
    header = segy_header_scrape(str(file), silent=True, **scrape_args)
    ds = segy_loader(str(file), silent=True, **segyio_kwargs, head_df=header)
    assert isinstance(ds, xr.Dataset)


def test_segyiotests_ps_2ncdf(temp_dir, segyio3dps_test_files):
    file, segyio_kwargs = segyio3dps_test_files
    seisnc = temp_dir / file.with_suffix(".siesnc").name
    segy_converter(str(file), ncfile=seisnc, silent=True, **segyio_kwargs)
    ds = open_seisnc(seisnc)
    assert isinstance(ds, xr.Dataset)


def test_segyiotests_ps_2ds(segyio3dps_test_files):
    file, segyio_kwargs = segyio3dps_test_files
    ds = segy_loader(str(file), silent=True, **segyio_kwargs)
    assert isinstance(ds, xr.Dataset)


def test_segyiotests_nohead_2ncdf(temp_dir, segyio_nohead_test_files):
    file, segyio_kwargs = segyio_nohead_test_files
    seisnc = temp_dir / file.with_suffix(".siesnc").name
    segy_converter(str(file), ncfile=seisnc, silent=True, **segyio_kwargs)
    ds = open_seisnc(seisnc)
    assert isinstance(ds, xr.Dataset)


### segy_writer tests


def test_output_byte_locs_fail():
    with pytest.raises(ValueError):
        output_byte_locs("gibberish")


@given(
    il_chunks=integers(1, 10),
)
@settings(max_examples=1, deadline=None)
def test_segyiotests_writer_from_ds(temp_dir, segyio3d_test_files, il_chunks):
    file, segyio_kwargs = segyio3d_test_files
    ds = segy_loader(str(file), silent=True, **segyio_kwargs)
    outfile = temp_dir / file.name
    segy_writer(ds, outfile, il_chunks=il_chunks)
    del ds
    ds = segy_loader(str(outfile), silent=True, **well_known_byte_locs("standard_3d"))
    assert isinstance(ds, xr.Dataset)


def test_segyiotests_writer_from_seisnc(temp_dir, segyio3d_test_files):
    file, segyio_kwargs = segyio3d_test_files
    seisnc = temp_dir / file.with_suffix(".siesnc").name
    segy_converter(str(file), ncfile=seisnc, silent=True, **segyio_kwargs)
    outfile = temp_dir / file.name
    segy_writer(seisnc, outfile)
    ds = segy_loader(str(outfile), silent=True, **well_known_byte_locs("standard_3d"))
    assert isinstance(ds, xr.Dataset)


def test_segyiotests_ps_writer_from_ds(temp_dir, segyio3dps_test_files):
    file, segyio_kwargs = segyio3dps_test_files
    ds = segy_loader(str(file), silent=True, **segyio_kwargs)
    outfile = temp_dir / file.name
    segy_writer(ds, outfile, use_text=True)
    del ds
    ds = segy_loader(
        str(outfile), silent=True, **well_known_byte_locs("standard_3d_gath")
    )
    assert isinstance(ds, xr.Dataset)


def test_segyiotests_ps_writer_from_ds(temp_dir, segyio3dps_test_files):
    file, segyio_kwargs = segyio3dps_test_files
    seisnc = temp_dir / file.with_suffix(".siesnc").name
    segy_converter(str(file), ncfile=seisnc, silent=True, **segyio_kwargs)
    outfile = temp_dir / file.name
    segy_writer(seisnc, outfile, use_text=True)
    ds = segy_loader(
        str(outfile), silent=True, **well_known_byte_locs("standard_3d_gath")
    )
    assert isinstance(ds, xr.Dataset)


# def test_segyiotests_nohead_2ds(segyio_nohead_test_files):
#     file, segyio_kwargs = segyio_nohead_test_files
#     ds = segy_loader(str(file), silent=True, **segyio_kwargs)
#     assert isinstance(ds, xr.Dataset)


# seisnc to segy to seisnc tests - go full circle


@parametrize(
    "empty",
    (
        fixture_ref(empty2d),
        fixture_ref(empty2d_gath),
        fixture_ref(empty3d),
        fixture_ref(empty3d_depth),
        fixture_ref(empty3d_gath),
        fixture_ref(empty3d_twt),
    ),
    ids=["2d", "2dgath", "3d", "3d_depth", "3d_gath", "3d_twt"],
)
def test_seisnc_return(temp_dir, empty):
    dims = list(empty.dims)
    shp = [empty.dims[d] for d in dims]
    domain = empty.d3_domain
    empty["data"] = (dims, np.zeros(shp))

    if "offset" in empty:
        empty["offset"] = empty["offset"].astype(np.int32)

    segy_writer(empty, temp_dir / "temp_empty.segy")
    back = segy_loader(temp_dir / "temp_empty.segy", vert_domain=domain)
    assert empty == back


if __name__ == "__main__":
    iln, xln, n, off, dom = (10, 11, 12, 5, "TWT")
    seisnc = create3d_dataset(
        (iln, xln, n, off),
        sample_rate=1,
        first_iline=1,
        first_xline=1,
        vert_domain=dom,
        first_offset=2,
        offset_step=2,
    )
    segy_writer(seisnc, "temp.segy")
