import pytest
import pathlib

import xarray as xr
import numpy as np

from segysak.segy._xarray import SgyBackendArray, SgyBackendEntrypoint
from segysak.segy._xarray_writer import (
    SegyWriter,
    chunk_iterator,
    chunked_index_iterator,
)

TESTS_PATH = pathlib.Path(__file__).parent.absolute()
TEST_DATA_SEGYIO = TESTS_PATH / "test-data-segyio"
TEST_DATA_SEGYSAK = TESTS_PATH / "test-data-segysak"


def test_chunked_index_iterator():
    chunks = (10, 10, 5, 10, 4)
    for i, chk in zip(
        chunked_index_iterator(chunks, np.arange(0, sum(chunks))), chunks
    ):
        assert i.size == chk


def test_chunk_iterator(f3_dataset):
    # no chunks -> auto chunking
    chunks = [chk for chk in chunk_iterator(f3_dataset["data"])]
    assert len(chunks) == 1

    # manual chunking
    ds_chunked = f3_dataset.chunk({"iline": 2, "xline": 2, "twt": -1})
    chunks = [chk for chk in chunk_iterator(ds_chunked["data"])]
    assert len(chunks) == 12 * 9


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


def test_SegyWriter_empty3d(temp_dir, empty3d):
    shape = tuple(size for size in empty3d.sizes.values())
    empty3d["data"] = xr.Variable(empty3d.dims, np.zeros(shape))
    with SegyWriter(temp_dir / "empty3d.segy", use_text=True, silent=True) as writer:
        writer.to_segy(
            empty3d,
            vert_dimension="twt",
        )


def test_SegyWriter_f3(temp_dir, f3_dataset):
    with SegyWriter(temp_dir / "f3_out.segy", use_text=True, silent=True) as writer:
        writer.to_segy(
            f3_dataset.chunk({"iline": -1, "xline": 2, "twt": -1}),
            vert_dimension="twt",
            trace_header_map={"cdp_x": 181, "cdp_y": 185},
            iline=189,
            xline=193,
        )
