import pytest
import pathlib

import xarray as xr
import numpy as np

from segysak.seisnc import create_empty_seisnc, set_seisnc_dims

from test_fixtures import temp_dir, TEMP_TEST_DATA_DIR

TEST_SEISNC = 'test.SEISNC'
TEST_DIMS = (10, 15, 20)
TEST_IND = (3, 1, 5, 2, 10, 4)

def test_create_empty_seisnc(temp_dir):
    f = temp_dir/TEST_SEISNC
    create_empty_seisnc(f, TEST_DIMS)
    assert f.exists()

def test_set_seisnc_dims(temp_dir):
    set_seisnc_dims(temp_dir/TEST_SEISNC, *TEST_IND)

    ds = xr.open_dataset(temp_dir/TEST_SEISNC)
    v_test = np.arange(TEST_IND[0], TEST_DIMS[2]*TEST_IND[1] + TEST_IND[0], TEST_IND[1])
    i_test = np.arange(TEST_IND[2], TEST_DIMS[0]*TEST_IND[3] + TEST_IND[2], TEST_IND[3])
    x_test = np.arange(TEST_IND[4], TEST_DIMS[1]*TEST_IND[5] + TEST_IND[4], TEST_IND[5])

    assert np.array_equal(ds.vert.values, v_test)
    assert np.array_equal(ds.iline.values, i_test)
    assert np.array_equal(ds.xline.values, x_test)

if __name__ == '__main__':
    pass