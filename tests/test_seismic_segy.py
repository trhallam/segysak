import pytest
import pathlib

from etlpy.seismic.segy import segy2ncdf, ncdf2segy

TEST_DATA_DIR = pathlib.Path('test_data')
TEMP_TEST_DATA_DIR = 'test_data_temp'
TEST_SEGY = TEST_DATA_DIR / 'med.sgy'
TEST_NCOUT = 'test_ncout.SEISNC'
TEST_SEGYOUT = 'test_segyout.segy'

@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory):
    tdir = tmpdir_factory.mktemp(TEMP_TEST_DATA_DIR)
    return pathlib.Path(str(tdir))

def test_segy2ncdf(temp_dir):
    segy2ncdf(TEST_SEGY, temp_dir/TEST_NCOUT)

def test_ncdf2segy(temp_dir):
    ncdf2segy(temp_dir/TEST_NCOUT, temp_dir/TEST_SEGYOUT)

if __name__ == '__main__':
    test_segy2ncdf()
    import xarray as xr
    d = xr.open_dataset(TEST_NCOUT)