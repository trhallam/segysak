import pytest
import pathlib

# import requests
# import zipfile

import numpy as np

import segyio

from segysak.segy import put_segy_texthead, create_default_texthead
from segysak import create2d_dataset, create3d_dataset


TEST_DATA_OZGY = pathlib.Path(__file__).parent.absolute() / "test-data-ozgy"
TEST_DATA_OZGY = pathlib.Path(TEST_DATA_OZGY)

TEST_FILES_OZGY = [("test_3d_float.zgy",), ("test_3d_i16.zgy",)]


@pytest.fixture(
    scope="session",
    params=TEST_FILES_OZGY,
    ids=["float", "i16"],
)
def ozgy_files(request):
    file = TEST_DATA_OZGY / request.param[0]
    return file
