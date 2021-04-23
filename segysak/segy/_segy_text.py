from warnings import warn
import segyio
from segysak._richstr import _upgrade_txt_richstr
from segysak.tools import _get_userid, _get_datetime


def _text_fixes(text):
    # hacky fixes for software
    text = text.replace("�Cro", "    ")
    text = text.replace("\x00", " ")
    return text


def _isascii(txt):
    # really want to use b"".isascii() but this is Python 3.7+
    try:
        txt.decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_segy_texthead(segyfile, ext_headers=False, no_richstr=False, **segyio_kwargs):
    """Return the ebcidc

    Args:
        segyfile (str): Segy File Path
        ext_headers (bool): Return EBCIDC and extended headers in list.
            Defaults to False
        no_richstr (bool, optional): Defaults to False. If true the returned string
            will not be updated for pretty HTML printing.
        segyio_kwargs: Key word arguments to pass to segyio.open
    Returns:
        str: Returns the EBCIDC text as a formatted paragraph.
    """

    with open(segyfile, mode="rb") as f:
        f.seek(0, 0)  # Locate our position to first byte of file
        data = f.read(3200)  # Read the first 3200 byte from our position

    if _isascii(data) and ext_headers == False:
        text = data.decode("ascii")  # EBCDIC encoding
        text = _text_fixes(text)
        text = segyio.tools.wrap(text)
    elif ext_headers == False:
        text = data.decode("cp500")  # text is ebcidc
        text = _text_fixes(text)
        text = segyio.tools.wrap(text)
    else:
        segyio_kwargs["ignore_geometry"] = True
        try:  # pray that the encoding is ebcidc
            with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
                text = segyf.text[0].decode("ascii", "replace")
                text = _text_fixes(text)
                text = segyio.tools.wrap(text)
                if segyf.ext_headers and ext_headers:
                    text2 = segyf.text[1].decode("ascii", "replace")
                    text = [text, text2]
        except UnicodeDecodeError as err:
            print(err)
            print("The segy text header could not be decoded.")

    if no_richstr:
        return text
    else:
        return _upgrade_txt_richstr(text)


def put_segy_texthead(segyfile, ebcidc, line_counter=True, **segyio_kwargs):

    header = ""
    if isinstance(ebcidc, dict):
        for key in ebcidc:
            if not isinstance(key, int):
                warn(
                    "ebcidc dict contains not integer keys that will be ignored",
                    UserWarning,
                )
        for line in range(1, 41):
            if line_counter:
                lc = f"C{line:02d} "
                n = 75
                content = "{:<76}"
            else:
                lc = ""
                n = 79
                content = "{:<80}"
            try:
                test = ebcidc[line]
                if len(test) > 75:
                    warn(f"EBCIDC line {line} is too long - truncating", UserWarning)
                header = header + lc + content.format(ebcidc[line][:n])
            except KeyError:
                # line not specified in dictionary
                header = header + lc + " " * n
        header = bytes(header, "utf8")
    elif isinstance(ebcidc, bytes):
        if len(ebcidc) > 3200:
            warn("Byte EBCIDC is too large - truncating", UserWarning)
        header = ebcidc[:3200]
    elif isinstance(ebcidc, str):
        if len(ebcidc) > 3200:
            warn("String EBCIDC is too large - truncating", UserWarning)
        header = bytes(ebcidc[:3200], "utf8")
    else:
        raise ValueError("Unknown ebcidc type expect bytes, str or dict")

    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r+", **segyio_kwargs) as segyf:
        segyf.text[0] = header


def _clean_texthead(text_dict, n=75):
    """Reduce texthead dictionary to 75 characters per line.

    The first 4 Characters of a segy EBCIDC should have the form "C01 " which
    is then follwed by 75 ascii characters.

    Input should have interger keys. Other keys will be ignored.
    Missing keys will be filled by blank lines.

    Args:
        text_dict (dict): line no and string pairs

    Returns:
        (dict): line no and string pairs ready for ebcidc input
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


def create_default_texthead(override=None):
    """Returns a simple default textual header dictionary.

    Basic fields are auto populated and a dictionary indexing lines 1-40 can
    be passed to override keyword for adjustment. By default lines 6-34 are
    empty.

    Args:
        override (dict, optional): Overide any line . Defaults to None.

    Returns:
        (dict): Dictionary with keys 1-40 for textual header of segy file

    Example:
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


def trace_header_map_to_text(trace_header_map):
    """Convert a trace header map to lines of text to say where header info was
    put in a file.

    Args:
        trace_header_map (dict): Header byte locations.

    Returns:
        list: List of lines to output.
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
