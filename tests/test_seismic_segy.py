import pytest

from .fixtures_seisnc import *

import pathlib

from hypothesis import given, settings
from hypothesis.strategies import integers

import numpy as np
import pandas as pd
import xarray as xr

from segysak.segy import (
    create_default_texthead,
    put_segy_texthead,
    get_segy_texthead,
    segy_header_scan,
    segy_header_scrape,
    segy_bin_scrape,
    header_as_dimensions,
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


@pytest.mark.parametrize("header", GOOD_HEADER, ids=["str", "dict", "bytes"])
def test_put_segy_texthead_ok(temp_dir, temp_segy, header):
    put_segy_texthead(temp_segy, header, line_counter=False)


@pytest.mark.parametrize("header", BAD_HEADER, ids=["str", "adict", "bytes"])
def test_put_segy_texthead_notok(temp_dir, temp_segy, header):
    with pytest.warns(UserWarning):
        put_segy_texthead(temp_segy, header)


def test_segy_header_scan(temp_dir, temp_segy):
    head = segy_header_scan(temp_segy)
    scanned = head.nscan
    assert scanned == TEST_SEGY_SIZE**2
    # assert isinstance(head, dict) now a dataframe
    head = segy_header_scan(temp_segy, max_traces_scan=TEST_SEGY_SIZE)
    scanned = head.nscan
    assert scanned == TEST_SEGY_SIZE
    # assert isinstance(head, dict) now a dataframe

    with pytest.raises(ValueError):
        header = segy_header_scan("", max_traces_scan=23.9)


def test_segy_header_scan_all(temp_dir, temp_segy):
    _ = segy_header_scan(temp_segy, max_traces_scan=0)
    scanned = _.nscan
    assert scanned == TEST_SEGY_SIZE**2
    _ = segy_header_scan(temp_segy, max_traces_scan=-1)
    scanned = _.nscan
    assert scanned == TEST_SEGY_SIZE**2


def test_segy_bin_scrape(temp_dir, temp_segy):
    assert isinstance(segy_bin_scrape(temp_segy), dict)


def test_segy_header_scrape(temp_dir, temp_segy):
    header = segy_header_scrape(temp_segy)
    assert isinstance(header, pd.DataFrame)
    assert header.shape == (TEST_SEGY_SIZE**2, 91)


def test_header_as_dimensions(temp_dir, temp_segy):
    header = segy_header_scrape(temp_segy)
    assert isinstance(header, pd.DataFrame)
    dims = header_as_dimensions(header, ("INLINE_3D", "CROSSLINE_3D"))
    assert len(dims) == 2


def process_segy_fixture(fixture):
    file, dim_byte_fields, extra_byte_fields, segyio_kwargs = fixture
    if extra_byte_fields is None:
        extra_byte_fields = dict()
    return file, dict(**dim_byte_fields, **extra_byte_fields, **segyio_kwargs)


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
