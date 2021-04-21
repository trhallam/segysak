import pytest

from segysak._richstr import _upgrade_txt_richstr


def test_richstring(volve_2d_dataset):
    richstr = _upgrade_txt_richstr(volve_2d_dataset.text)
    html = richstr._repr_html_()
    print(richstr)
    print(html)
    assert True
