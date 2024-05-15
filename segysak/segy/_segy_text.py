from typing import Dict, List, Union, ByteString, Any
from warnings import warn
import os
import re
import textwrap
from functools import reduce
from collections import defaultdict
import operator
import segyio
from segysak._richstr import _upgrade_txt_richstr
from segysak.tools import _get_userid, _get_datetime


def _text_fixes(text: str) -> str:
    # hacky fixes for software
    text = text.replace("�Cro", "    ")
    text = text.replace("\x00", " ")
    return text


def _isascii(txt: str) -> str:
    # really want to use b"".isascii() but this is Python 3.7+
    try:
        txt.decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_segy_texthead(
    segy_file: Union[str, os.PathLike],
    ext_headers: bool = False,
    no_richstr: bool = False,
    **segyio_kwargs: Dict[str, Any],
) -> str:
    """Return the ebcidc header as a Python string. New lines are separated by the `\\n` char.

    Args:
        segy_file: Segy File Path
        ext_headers: Return EBCIDC and extended headers in list. Defaults to False
        no_richstr: Defaults to False. If true the returned string
            will not be updated for pretty HTML printing.
        segyio_kwargs: Key word arguments to pass to segyio.open

    Returns:
        text: Returns the EBCIDC text as a formatted paragraph.
    """

    with open(segy_file, mode="rb") as f:
        f.seek(0, 0)  # Locate our position to first byte of file
        data = f.read(3200)  # Read the first 3200 byte from our position

    if _isascii(data) and ext_headers == False:
        encoding = "ascii"
    elif ext_headers == False:
        encoding = "cp500"  # text is ebcidc
    else:
        encoding = "ebcidc"

    if encoding in ["ascii", "cp500"]:
        lines = []
        # doing it this way ensure we split the bytes appropriately across the 40 lines.
        for i in range(0, 3200, 80):
            lines.append(data[i : i + 80].decode("cp500"))
            text = "\n".join(lines)
    else:
        segyio_kwargs["ignore_geometry"] = True
        try:  # pray that the encoding is ebcidc
            with segyio.open(segy_file, "r", **segyio_kwargs) as segyf:
                text = segyf.text[0].decode("ascii", "replace")
                text = _text_fixes(text)
                text = segyio.tools.wrap(text)
                if segyf.ext_headers and ext_headers:
                    text2 = segyf.text[1].decode("ascii", "replace")
                    text = [text, text2]
        except UnicodeDecodeError as err:
            print(err)
            print("The segy text header could not be decoded.")

    text = _text_fixes(text)

    if no_richstr:
        return text
    else:
        return _upgrade_txt_richstr(text)


def _process_string_texthead(string: str, n: int, nlines: int = 40) -> List[str]:
    """New lines are preserved.

    Args:
        string: The textheader as a string.
        n: The number of allowable chars per line.
        nlines: The number of allowable lines in the text header. Use for for extended text headers.

    Returns:
        list: The string broken into lines and pre-padded for line breaks.
    """
    txt = reduce(
        operator.add, [textwrap.wrap(s, width=n) for s in string.split("\n")], []
    )
    txt = list(map(lambda x: x.strip().ljust(n), txt))
    while len(txt) < nlines:
        txt.append("".ljust(n))
    return txt


def _process_dict_texthead(
    strdict: Dict[int, str], n: int, nlines: int = 40
) -> List[str]:
    """Left justifies each value of the dictionary with padding to length n
    Ensures there are values for all line numbers (1-n)

    Args:
        strdict: The text header as a dict with numeric keys for line numbers e.g. {1: 'line 1'}.
        n: The number of allowable chars per line.
        nlines: The number of allowable lines in the text header. Use for for extended text headers.

    Returns:
        lines: The string broken into lines and pre-padded for line breaks.
    """
    tdict = defaultdict(lambda: "")
    tdict.update(strdict)
    return [tdict[i].ljust(n) for i in range(1, nlines + 1)]


def _process_line(line: str, i: int, line_counter: bool = True) -> str:

    # trim the right white space
    line = line.rstrip()

    # add a counter to start of line like `C 1`, `C 2` ... `C40` if not already.
    if line_counter:
        counter = f"C{i+1:2d}"
        if not re.match(r"C[\s|\d]\d.*$", line):
            line = f"{counter} {line:s}"

    # check line length
    if len(line) > 81:
        warn(f"EBCIDC line {line} is too long - truncating", UserWarning)

    line = line.ljust(80)
    return line


def put_segy_texthead(
    segy_file: Union[str, os.PathLike],
    ebcidc: Union[str, List[str], Dict[int, str], ByteString],
    line_counter: bool = True,
    **segyio_kwargs,
):
    """Puts a text header (ebcidc) into a SEG-Y file.

    Args:
        segy_file: The path to the file to update.
        ebcidc:
            A standard string, new lines will be preserved.
            A list or lines to add.
            A dict with numeric keys for line numbers e.g. {1: 'line 1'}.
            A pre-encoded byte header to add to the SEG-Y file directly.
        line_counter: Add a line counter with format "CXX " to the start of each line.
            This reduces the maximum content per line to 76 chars.
    """
    header = ""
    n = 80

    if not isinstance(ebcidc, (str, list, dict, bytes)):
        raise ValueError(f"Unknown type for ebcidc: {type(ebcidc)}")

    if isinstance(ebcidc, dict):
        lines = _process_dict_texthead(ebcidc, n)
    elif isinstance(ebcidc, str):
        lines = _process_string_texthead(ebcidc, n)
    elif isinstance(ebcidc, list):
        lines = ebcidc
    else:
        lines = []

    if not isinstance(ebcidc, bytes):
        lines = [
            _process_line(line, i, line_counter=line_counter)
            for i, line in enumerate(lines)
        ]
        # convert to bytes line by line to ensure end lines don't get pushed,
        # truncate lines with bad chars instead
        header = b"".join([ln.encode("utf-8")[:n] for ln in lines])
    else:
        header = ebcidc

    # check size
    if len(header) > 3200:
        warn("Byte EBCIDC is too large - truncating", UserWarning)
        header = header[:3200]

    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segy_file, "r+", **segyio_kwargs) as segyf:
        segyf.text[0] = header


def _clean_texthead(text_dict: Dict[int, str], n: int = 75) -> Dict[int, str]:
    """Reduce texthead dictionary to 75 characters per line.

    The first 4 Characters of a segy EBCIDC should have the form "C01 " which
    is then followed by 75 ascii characters.

    Input should have integer keys. Other keys will be ignored.
    Missing keys will be filled by blank lines.

    Args:
        text_dict: line no and string pairs

    Returns:
        text_dict: line no and string pairs ready for ebcidc input
    """
    output = dict()
    for line in range(1, 41, 1):
        try:
            line_str = text_dict[line]
            if len(line_str) > n:
                line_str = line_str[0:n]
        except KeyError:
            line_str = ""
        output[line] = line_str
    return output


def create_default_texthead(
    override: Union[Dict[int, str], None] = None
) -> Dict[int, str]:
    """Returns a simple default textual header dictionary.

    Basic fields are auto populated and a dictionary indexing lines 1-40 can
    be passed to override keyword for adjustment. By default lines 6-34 are
    empty.

    Line length rules apply, so overrides will be truncated if they have >80 chars.

    Args:
        override: Override any line with custom values. Defaults to None.

    Returns:
        text_header: Dictionary with keys 1-40 for textual header of SEG-Y file

    !!! example
        Override lines 7 and 8 of the default text header.

        ```python
        >>> create_default_texthead(override={7:'Hello', 8:'World!'})
        {1: 'segysak SEG-Y Output',
        2: 'Data created by: username ',
        3: '',
        4: 'DATA FORMAT: SEG-Y;  DATE: 2019-06-09 15:14:00',
        5: 'DATA DESCRIPTION: SEG-Y format data output from segysak',
        6: '',
        7: 'Hello',
        8: 'World!',
        9: '',
        ...
        34: '',
        35: '*** BYTE LOCATION OF KEY HEADERS ***',
        36: 'CMP UTM-X 181-184, ALL COORDS X100, CMP UTM-Y 185-188',
        37: 'INLINE 189-193, XLINE 194-198, ',
        38: '',
        39: '',
        40: 'END TEXTUAL HEADER'}
        ```

    """
    user = _get_userid()
    today, time = _get_datetime()
    text_dict = {
        #      123456789012345678901234567890123456789012345678901234567890123456
        1: "segysak Python Library SEG-Y Output",
        2: f"Data created by: {user} ",
        4: f"DATA FORMAT: SEG-Y;  DATE: {today} {time}",
        5: "DATA DESCRIPTION: SEG-Y format data output from segysak using segyio",
        6: "",
        40: "END TEXTUAL HEADER",
    }
    if override is not None:
        for key, line in override.items():
            text_dict[key] = line
    return _clean_texthead(text_dict)


def trace_header_map_to_text(trace_header_map: Dict[str, int]) -> List[str]:
    """Convert a trace header map to lines of text to say where header info was
    put in a file.

    Args:
        trace_header_map: Header byte locations.

    Returns:
        description: List of lines to output.
    """
    header_locs_text = [f"{key}: {value}," for key, value in trace_header_map.items()]

    lines = ["*** BYTE LOCATION OF KEY HEADERS ***"]
    line = ""
    for byte_loc in header_locs_text:
        if len(line + byte_loc) < 79:
            line += byte_loc
            line += " "
        else:
            lines.append(line)
            line = byte_loc
    lines.append(line)
    return lines
