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


# def test_SegyWriter_empty3d(temp_dir, empty3d):
#     shape = tuple(size for size in empty3d.sizes.values())
#     empty3d["data"] = xr.Variable(empty3d.dims, np.zeros(shape))
#     with SegyWriter(temp_dir / "empty3d.segy", use_text=True, silent=True) as writer:
#         writer.to_segy(
#             empty3d,
#             vert_dimension="twt",
#         )


@pytest.fixture(
    params=[
        {"iline": slice(0, 25), "xline": slice(0, 20), "twt": slice(0, 100)},
        {"iline": slice(10, 11), "xline": slice(0, 20), "twt": slice(0, 100)},
        {"iline": slice(0, 25), "xline": slice(10, 11), "twt": slice(0, 100)},
        {"iline": slice(0, 5), "xline": slice(10, 15), "twt": slice(0, 30)},
        {"iline": slice(0, 25), "xline": slice(0, 20), "twt": slice(10, 12)},
        {"iline": slice(0, 25), "xline": slice(0, 20), "twt": slice(20, 50)},
    ],
    scope="module",
    ids=[
        "full",
        "single-iline",
        "single-xline",
        "sub-volume",
        "single-slice",
        "trim-time",
    ],
)
def f3_dataset_slicing(request):
    return request.param


@pytest.fixture(
    params=[
        {"iline": -1, "xline": 2, "twt": -1},
        {"iline": 2, "xline": -1, "twt": -1},
        {"iline": 2, "xline": 2, "twt": -1},
        {"iline": 2, "xline": 2, "twt": 20},
        {},
    ],
    ids=["xline2-chunk", "iline2-chunk", "dual-trunk", "sub-chunk", "no-chunk"],
    scope="module",
)
def f3_dataset_chunks(request):
    return request.param


def test_SegyWriter_f3(
    request, temp_dir, f3_dataset, f3_dataset_slicing, f3_dataset_chunks
):
    slices = f3_dataset_slicing
    chunks = f3_dataset_chunks
    if chunks:
        ds = f3_dataset.isel(**slices).chunk(chunks)
    else:
        ds = f3_dataset.isel(**slices)
    id = request.node.callspec.id
    out_file = temp_dir / f"f3-out-{id}.segy"
    with SegyWriter(out_file, use_text=True, silent=True) as writer:
        writer.to_segy(
            ds,
            vert_dimension="twt",
            trace_header_map={"cdp_x": 181, "cdp_y": 185},
            iline=189,
            xline=193,
            silent=True,
        )

    ds_out = (
        xr.open_dataset(
            out_file,
            silent=True,
            dim_byte_fields={"iline": 189, "xline": 193},
            extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        )
        .rename_dims({"samples": "twt"})
        .rename_vars({"samples": "twt"})
    )

    assert ds.dims == ds_out.dims
    for dim in ds.dims:
        assert np.array_equal(ds[dim].values, ds_out[dim].values)

    assert np.allclose(ds["data"], ds_out["data"])
