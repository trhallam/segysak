import pytest
import pathlib

from .fixtures_segy import *
from .fixtures_zgy import *
from .fixtures_seisnc import *
from .fixtures_data import *

collect_ignore = ["test_seismic_zgy.py"]


TEMP_TEST_DATA_DIR = "test_data_temp"


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory):
    tdir = tmpdir_factory.mktemp(TEMP_TEST_DATA_DIR)
    return pathlib.Path(str(tdir))
