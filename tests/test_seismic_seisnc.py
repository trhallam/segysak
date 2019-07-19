import pytest
import pathlib

from etlpy.seismic.seisnc import create_empty_seisnc, set_seisnc_dims, copy_seisnc_structure

TEST_DATA_DIR = pathlib.Path('working_test_data')
TEST_SEISNC = TEST_DATA_DIR / 'test.seisnc'
TESTC_SEISNC = TEST_DATA_DIR / 'test_copy.seisnc'

if not TEST_DATA_DIR.exists():
    TEST_DATA_DIR.mkdir()

def test_create_empty_seisnc():
    create_empty_seisnc(TEST_SEISNC, (10,15,20))
    # assert test.seisnc created by opening and checking dims

def test_set_seisnc_dims():
    set_seisnc_dims(TEST_SEISNC, 3, 1, 5, 2, 10, 4)
    # assert changed dims

def test_copy_seisnc_structure():
    copy_seisnc_structure(TEST_SEISNC, TESTC_SEISNC)

if __name__ == '__main__':
    test_create_empty_seisnc()
    test_set_seisnc_dims()
    test_copy_seisnc_structure()
    import xarray as xr
    d = xr.open_dataset(TEST_SEISNC)
    dc = xr.open_dataset(TESTC_SEISNC)