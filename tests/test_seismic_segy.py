import pytest
import pathlib

import numpy as np
import pandas as pd
import xarray as xr


from segysak.segy import segy2ncdf, ncdf2segy, create_default_texthead, put_segy_texthead, get_segy_texthead, \
    segy_header_scan, segy_header_scrape, segy_bin_scrape

from test_fixtures import *

GOOD_HEADER = ('a'*3200, {n:'a'*65 for n in range(1, 41)}, bytes('a'*3200, 'utf8'))
BAD_HEADER = ('b'*4000, {n:'b'*85 for n in range(1, 41)}, bytes('b'*4000, 'utf8'), {'not_a_num':'b'*65})

# @pytest.fixture(scope="session")
# def temp_dir(tmpdir_factory):
#     tdir = tmpdir_factory.mktemp(TEMP_TEST_DATA_DIR)
#     return pathlib.Path(str(tdir))

def test_segy_fixtures(temp_dir, temp_segy):
    assert temp_segy.exists()

@pytest.fixture()
def cropping_limits():
    min_il = TEST_SEGY_SIZE//4
    max_il = TEST_SEGY_SIZE//2 + min_il
    min_xl = TEST_SEGY_SIZE + TEST_SEGY_SIZE//4
    max_xl = TEST_SEGY_SIZE//2 + min_xl
    crop = [min_il, max_il, min_xl, max_xl]

    min_z = TEST_SEGY_SIZE//4
    max_z = TEST_SEGY_SIZE//2 + min_z
    cropz = [min_z, max_z]
    return crop, cropz

def test_get_segy_texthead(temp_dir, temp_segy):
    ebcidc = get_segy_texthead(temp_segy)
    assert isinstance(ebcidc, str)
    assert len(ebcidc) < 3200
    assert ebcidc[:3] == 'C01'

@pytest.mark.parametrize('header', GOOD_HEADER, ids=['str', 'dict', 'bytes'])
def test_put_segy_texthead_ok(temp_dir, temp_segy, header):
    put_segy_texthead(temp_segy, header)

@pytest.mark.parametrize('header', BAD_HEADER, ids=['str', 'adict', 'bytes', 'bdict'])
def test_put_segy_texthead_notok(temp_dir, temp_segy, header):
    with pytest.warns(UserWarning):
        put_segy_texthead(temp_segy, header)

def test_segy_header_scan(temp_dir, temp_segy):
    head, scanned = segy_header_scan(temp_segy)
    assert scanned == TEST_SEGY_SIZE**2
    assert isinstance(head, dict)
    head, scanned = segy_header_scan(temp_segy, max_traces_scan=TEST_SEGY_SIZE, silent=True)
    assert scanned == TEST_SEGY_SIZE
    assert isinstance(head, dict)

def test_segy_header_scan_all(temp_dir, temp_segy):
    _, scanned = segy_header_scan(temp_segy, max_traces_scan=0, silent=True)
    assert scanned == TEST_SEGY_SIZE**2
    _, scanned = segy_header_scan(temp_segy, max_traces_scan='all', silent=True)
    assert scanned == TEST_SEGY_SIZE**2

def test_segy_bin_scrape(temp_dir, temp_segy):
    assert isinstance(segy_bin_scrape(temp_segy), dict)

def test_segy_header_scrape(temp_dir, temp_segy):
    header = segy_header_scrape(temp_segy, silent=True)
    assert isinstance(header, pd.DataFrame)
    assert header.shape == (TEST_SEGY_SIZE**2, 89)

def test_segy2ncdf_3D(temp_dir, temp_segy):
    temp_seisnc = temp_segy.with_suffix('.SEISNC')
    segy2ncdf(temp_segy, temp_seisnc, silent=True)

def test_segy2ncdf_3D_cropH(temp_dir, temp_segy, cropping_limits):
    temp_seisnc = temp_segy.with_suffix('.SEISNC')

    crop, _ = cropping_limits
    min_il, max_il, min_xl, max_xl = crop

    segy2ncdf(temp_segy, temp_seisnc, silent=True, crop=crop)

    ds = xr.open_dataset(temp_seisnc)
    assert tuple(ds.dims.values()) == (max_il - min_il + 1, max_xl - min_xl + 1, TEST_SEGY_SIZE)

def test_segy2ncdf_3D_cropZ(temp_dir, temp_segy, cropping_limits):
    temp_seisnc = temp_segy.with_suffix('.SEISNC')

    _, cropz = cropping_limits
    min_z, max_z = cropz

    segy2ncdf(temp_segy, temp_seisnc, silent=True, zcrop=[min_z, max_z])

    ds = xr.open_dataset(temp_seisnc)
    assert tuple(ds.dims.values())[2] == max_z - min_z + 1

def test_segy2ncdf_3D_crop(temp_dir, temp_segy, cropping_limits):
    temp_seisnc = temp_segy.with_suffix('.SEISNC')
    crop, cropz = cropping_limits
    min_il, max_il, min_xl, max_xl = crop
    min_z, max_z = cropz

    segy2ncdf(temp_segy, temp_seisnc, silent=True, crop=crop, zcrop=cropz)

    ds = xr.open_dataset(temp_seisnc)
    assert tuple(ds.dims.values()) == (max_il - min_il + 1, max_xl - min_xl + 1, max_z - min_z + 1)

if __name__ == '__main__':
    pass
