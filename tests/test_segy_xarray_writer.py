import xarray as xr
import pytest
import pathlib

from segysak.segy._xarray import SgyBackendArray, SgyBackendEntrypoint
from segysak.segy._xarray_writer import SegyWriter

TESTS_PATH = pathlib.Path(__file__).parent.absolute()
TEST_DATA_SEGYIO = TESTS_PATH / "test-data-segyio"
TEST_DATA_SEGYSAK = TESTS_PATH / "test-data-segysak"


def test_SegyWriter_default_text_header():
    with SegyWriter("") as writer:
        text = writer._default_text_header(
            {
                "iline": 189,
                "xline": 193,
                "cdp_x": 181,
                "cdp_y": 185,
            }
        )
    assert isinstance(text, str)
    for line in text.split("\n"):
        assert len(line) <= 80


def test_SegyWriter_write_text_header_no_file():
    with SegyWriter("nonexistant.file") as writer:
        with pytest.raises(FileNotFoundError):
            text = writer.write_text_header("some text")


# def test_SegyWriter(temp_dir, empty3d):

#     with SegyWriter(temp_dir/ 'empty3d.segy', use_text=True, silent=True) as writer:
