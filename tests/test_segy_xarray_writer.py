import pytest
import pathlib

import xarray as xr
import numpy as np

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
    for (slc, i), chk in zip(
        chunked_index_iterator(chunks, np.arange(0, sum(chunks))), chunks
    ):
        assert isinstance(slc, slice)
        assert slc.stop - slc.start == i.size
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


@pytest.fixture(
    params=[
        {"iline": slice(0, 25), "xline": slice(0, 20), "twt": slice(0, 100)},
        {"iline": slice(10, 11), "xline": slice(0, 20), "twt": slice(0, 100)},
        {"iline": 10, "xline": slice(0, 20), "twt": slice(0, 100)},
        {"iline": slice(0, 25), "xline": slice(10, 11), "twt": slice(0, 100)},
        {"iline": slice(0, 25), "xline": 10, "twt": slice(0, 100)},
        {"iline": slice(0, 5), "xline": slice(10, 15), "twt": slice(0, 30)},
        {"iline": slice(0, 25), "xline": slice(0, 20), "twt": slice(50, 51)},
        {"iline": slice(0, 25), "xline": slice(0, 20), "twt": 50},
        {"iline": slice(0, 25), "xline": slice(0, 20), "twt": slice(20, 50)},
        {"iline": 1, "xline": 1, "twt": slice(0, 100)},
    ],
    scope="module",
    ids=[
        "full",
        "single-iline",
        "single-iline-int",
        "single-xline",
        "single-xline-int",
        "sub-volume",
        "single-slice",
        "single-slice-int",
        "trim-time",
        "single-trace",
    ],
)
def f3_dataset_slicing(request):
    return request.param


@pytest.fixture(
    params=[
        {"iline": -1, "xline": 10, "twt": -1},
        {"iline": 10, "xline": -1, "twt": -1},
        {"iline": 10, "xline": 10, "twt": -1},
        {"iline": 10, "xline": 10, "twt": 50},
        {},
    ],
    ids=["xline2-chunk", "iline2-chunk", "dual-chunk", "sub-chunk", "no-chunk"],
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
        ds = f3_dataset.chunk(chunks).isel(**slices)
    else:
        ds = f3_dataset.isel(**slices)

    id = request.node.callspec.id
    out_file = temp_dir / f"f3-out-{id}.segy"
    with SegyWriter(out_file, use_text=True) as writer:
        writer.to_segy(
            ds,
            vert_dimension="twt",
            trace_header_map={"cdp_x": 181, "cdp_y": 185},
            iline=189,
            xline=193,
        )

    ds_out = (
        xr.open_dataset(
            out_file,
            dim_byte_fields={"iline": 189, "xline": 193},
            extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        )
        .rename_dims({"samples": "twt"})
        .rename_vars({"samples": "twt"})
    )

    for dim in ds.dims:
        assert np.array_equal(ds[dim].values, ds_out[dim].values)

    if any(isinstance(slc, int) for slc in slices.values()):
        ds_out = ds_out.squeeze()

    assert ds.dims == ds_out.dims
    assert np.allclose(ds["data"], ds_out["data"])


def test_SegyWriter_f3_withdead(
    request, temp_dir, f3_withdead_dataset, f3_dataset_slicing, f3_dataset_chunks
):
    f3_dataset = f3_withdead_dataset
    slices = f3_dataset_slicing
    chunks = f3_dataset_chunks
    if chunks:
        ds = f3_dataset.chunk(chunks).isel(**slices)
    else:
        ds = f3_dataset.isel(**slices)

    # find dead traces - don't do this lazy
    if "twt" in ds.data.dims:
        ds["dead_traces"] = np.logical_not(
            np.abs(ds.data).sum(dim="twt").astype(bool)
        ).compute()
    else:
        # slices need to be handled without sum as no twt dim
        ds["dead_traces"] = np.abs(ds.data).compute() == 0.0
    dead_traces = "dead_traces"

    id = request.node.callspec.id
    out_file = temp_dir / f"f3dead-out-{id}.segy"
    with SegyWriter(out_file, use_text=True) as writer:
        writer.to_segy(
            ds,
            vert_dimension="twt",
            trace_header_map={"cdp_x": 181, "cdp_y": 185},
            dead_trace_var=dead_traces,
            iline=189,
            xline=193,
        )

    ds_out = (
        xr.open_dataset(
            out_file,
            dim_byte_fields={"iline": 189, "xline": 193},
            extra_byte_fields={"cdp_x": 181, "cdp_y": 185},
        )
        .rename_dims({"samples": "twt"})
        .rename_vars({"samples": "twt"})
    )

    # because some traces are dropped, the full original dimensions may not be
    # recoverable, therefore we broadcast like the original and fill with zero
    # to recover the dead traces
    ds_out = ds_out.broadcast_like(ds).fillna(0.0)

    for dim in ds.dims:
        assert np.array_equal(ds[dim].values, ds_out[dim].values)

    if any(isinstance(slc, int) for slc in slices.values()):
        ds_out = ds_out.squeeze()

    assert ds.dims == ds_out.dims
    assert np.allclose(ds["data"], ds_out["data"])
