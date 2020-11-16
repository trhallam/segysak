import pytest

# from test_fixtures import segyio_all_test_files

from segysak.segy import get_segy_texthead
from segysak._richstr import _upgrade_txt_richstr


def test_upgrade_txt_richstr(segyio_all_test_files):
    file, segyio_kwargs = segyio_all_test_files
    text = get_segy_texthead(file, no_richstr=True, **segyio_kwargs)
    text = _upgrade_txt_richstr(text)
    assert isinstance(text, str)
    assert hasattr(text, "_repr_html_")
    assert hasattr(text, "_repr_pretty_")


## TODO: Add tests to the pretty printing functions
