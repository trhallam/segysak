import pytest

from segysak.segy._segy_freewriter import _chunk_iterator, segy_freewriter


def test_chunk_iterator(f3_dataset):
    ds = f3_dataset.chunk({"iline": 2, "xline": 2})
    chunks = [chk for chk in _chunk_iterator(ds)]
    assert len(chunks) == 12 * 9

    chunks = [chk for chk in _chunk_iterator(f3_dataset)]
    assert len(chunks) == 1


def test_segy_freewriter_ds(f3_dataset, temp_dir):
    sgyf = temp_dir / "f3_test.sgy"
    segy_freewriter(
        f3_dataset,
        str(sgyf),
        trace_header_map=dict(cdp_x=181, cdp_y=185),
        iline=189,
        xline=193,
    )
    assert sgyf.exists()
