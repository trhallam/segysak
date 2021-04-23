from numpy import isin
import pytest
import shutil

from segysak.segy._segy_text import (
    put_segy_texthead,
    get_segy_texthead,
    _isascii,
    _clean_texthead,
    create_default_texthead,
)


@pytest.fixture
def fake_header():
    fake_header = ""
    for i in range(40):
        fake_header += f"C{i:02d} A SEG-Y HEADER" + " " * 63
    return fake_header


@pytest.fixture
def default_header():
    head_dict = create_default_texthead(
        {20: "Hello World!", 21: "A Very Long Line Repeated" * 10}
    )
    return head_dict


@pytest.fixture
def ascii_bytes_header(fake_header):
    return bytes(fake_header, "ascii")


@pytest.fixture
def segy_ascii(temp_dir, ascii_bytes_header, segy_with_nonascii_text):
    name = temp_dir / "ascii_segy.sgy"
    shutil.copy(segy_with_nonascii_text, name)
    with open(name, mode="wb") as segyf:
        segyf.seek(0, 0)
        segyf.write(ascii_bytes_header)

    return name


def test_isascii(
    ascii_bytes_header,
):
    assert _isascii(ascii_bytes_header)


def test_get_segy_texthead(temp_dir, temp_segy):
    ebcidc = get_segy_texthead(temp_segy)
    ebcidc = get_segy_texthead(temp_segy, no_richstr=True)

    assert isinstance(ebcidc, str)
    assert len(ebcidc) < 3200
    assert ebcidc[:3] == "C01"


def test_get_segy_texthead_odd(temp_dir, segy_with_nonascii_text):
    ebcidc = get_segy_texthead(segy_with_nonascii_text)
    print(ebcidc)
    assert isinstance(ebcidc, str)


def test_get_segy_texthead_ascii(segy_ascii):
    ebcidc = get_segy_texthead(segy_ascii)
    print(ebcidc)
    assert isinstance(ebcidc, str)


# TODO: NEED AN EXAMPLE WITH MULTIPLE EXTENDED EBCIDC HEADERS


def test_clean_texthead(default_header):
    cleaned = _clean_texthead(default_header)
    for key, val in cleaned.items():
        assert isinstance(key, int)
        assert len(val) <= 80


def test_put_segy_texthead(temp_segy, default_header):
    put_segy_texthead(temp_segy, default_header)
    ebcidc = get_segy_texthead(temp_segy)
    assert "Hello World" in ebcidc
    put_segy_texthead(temp_segy, default_header, line_counter=False)
    ebcidc = get_segy_texthead(temp_segy)
    assert "Hello World" in ebcidc

    default_header.update({34.5: "Bad Key", 30: "Long Line" * 20})
    put_segy_texthead(temp_segy, default_header)
    ebcidc = get_segy_texthead(temp_segy)
    assert "Hello World" in ebcidc


def test_put_segy_texthead_bytes(temp_segy, ascii_bytes_header):
    put_segy_texthead(temp_segy, ascii_bytes_header)
    ebcidc = get_segy_texthead(temp_segy)
    assert "A SEG-Y HEADER" in ebcidc

    put_segy_texthead(temp_segy, ascii_bytes_header * 10)
    ebcidc = get_segy_texthead(temp_segy)
    assert "A SEG-Y HEADER" in ebcidc


def test_put_segy_texthead_str(temp_segy, ascii_bytes_header):
    put_segy_texthead(temp_segy, ascii_bytes_header.decode("ascii"))
    ebcidc = get_segy_texthead(temp_segy)
    assert "A SEG-Y HEADER" in ebcidc

    put_segy_texthead(temp_segy, ascii_bytes_header.decode("ascii") * 2)
    ebcidc = get_segy_texthead(temp_segy)
    assert "A SEG-Y HEADER" in ebcidc


def test_put_segy_texthead_unknown(temp_segy):
    with pytest.raises(ValueError):
        _ = put_segy_texthead(temp_segy, None)
