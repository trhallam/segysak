import pytest
import pathlib

from .fixtures_segy import *
from .fixtures_seisnc import *
from .fixtures_data import *

from segysak.progress import Progress

collect_ignore = ["test_seismic_zgy.py"]


TEMP_TEST_DATA_DIR = "test_data_temp"
TESTS_PATH = pathlib.Path(__file__).parent.absolute()
TEST_DATA_SEGYIO = TESTS_PATH / "test-data-segyio"
TEST_DATA_SEGYSAK = TESTS_PATH / "test-data-segysak"


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory):
    tdir = tmpdir_factory.mktemp(TEMP_TEST_DATA_DIR)
    return pathlib.Path(str(tdir))


@pytest.fixture(scope="session")
def tests_path(autouse=True):
    return TESTS_PATH


@pytest.fixture(autouse=True)
def tqdm():
    Progress.set_defaults(disable=True)
