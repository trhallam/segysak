import pytest
import pathlib

from .fixtures_segy import *
from .fixtures_seisnc import *
from .fixtures_data import *

from segysak.progress import Progress

collect_ignore = ["test_seismic_zgy.py"]


TEMP_TEST_DATA_DIR = "test_data_temp"


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory):
    tdir = tmpdir_factory.mktemp(TEMP_TEST_DATA_DIR)
    return pathlib.Path(str(tdir))


@pytest.fixture(autouse=True)
def tqdm():
    Progress.set_defaults(disable=True)
