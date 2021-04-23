import numpy as np
import pandas as pd
import segyio

from ._segy_core import (
    _active_tracefield_segyio,
    _active_binfield_segyio,
    tqdm,
    check_tracefield,
)


TQDM_ARGS = dict(unit_scale=True, unit=" traces")


def segy_header_scan(segyfile, max_traces_scan=1000, silent=False, **segyio_kwargs):
    """Perform a scan of the segy file headers and return ranges.

    To get the raw header information see segy_header_scrape

    Args:
        segyfile (string): Segy File Path
        max_traces_scan (int/str, optional): Number of traces to scan.
            For all set to 0 or 'all'. Defaults to 1000.
        silent (bool): Disable progress bar.

    Returns:
        pandas.DataFrame: Uses pandas describe to return statistics of your headers.
    """
    if max_traces_scan == 0 or max_traces_scan == "all":
        max_traces_scan = None
    else:
        if not isinstance(max_traces_scan, int):
            raise ValueError("max_traces_scan must be int")

    head_df = segy_header_scrape(
        segyfile, max_traces_scan, silent=silent, **segyio_kwargs
    )

    header_keys = head_df.describe().T
    pre_cols = list(header_keys.columns)
    header_keys["byte_loc"] = list(_active_tracefield_segyio().values())
    header_keys = header_keys[["byte_loc"] + pre_cols]
    header_keys.nscan = head_df.shape[0]
    return header_keys


def segy_bin_scrape(segyfile, **segyio_kwargs):
    """Scrape binary header

    Args:
        segyfile (str): SEG-Y File path

    Returns:
        dict: Binary header
    """
    bk = _active_binfield_segyio()
    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        return {key: segyf.bin[item] for key, item in bk.items()}


def segy_header_scrape(
    segyfile,
    partial_scan=None,
    silent=False,
    bytes_filter=None,
    chunk=100_000,
    **segyio_kwargs,
):
    """Scape all data from segy trace headers

    Args:
        segyfile (str): SEG-Y File path
        partial_scan (int): Setting partial scan to a positive int will scan only
            that many traces. Defaults to None.
        silent (bool): Disable progress bar.
        bytes_filter (list): List of byte locations to load exclusively.
        chunk (int): Number of traces to read in one go.

    Returns:
        pandas.DataFrame: Raw header information in table for scanned traces.
    """
    check_tracefield(bytes_filter)

    assert (chunk > 0) and isinstance(chunk, int)
    header_keys = _active_tracefield_segyio()
    enum_byte_index = {
        int(byte_loc): i for i, byte_loc in enumerate(header_keys.values())
    }

    if bytes_filter:
        for byte_loc in bytes_filter:
            assert byte_loc in enum_byte_index
        bytes_filter_index = [enum_byte_index[byte_loc] for byte_loc in bytes_filter]
        enum_filter = [segyio.TraceField(byte_loc) for byte_loc in bytes_filter]
    else:
        bytes_filter_index = list(enum_byte_index.values())
        enum_filter = [segyio.TraceField(byte_loc) for byte_loc in header_keys.values()]

    columns = [list(header_keys.keys())[i] for i in bytes_filter_index]
    segyio_kwargs["ignore_geometry"] = True

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        segyf_hgen = segyf.header[:]
        ntraces = segyf.tracecount
        if partial_scan is not None:
            ntraces = min(ntraces, int(partial_scan))

        head_df = pd.DataFrame(index=pd.Index(range(ntraces)), columns=columns)
        slc_end = chunk
        with tqdm(total=ntraces, disable=silent, **TQDM_ARGS) as pbar:
            while slc_end <= ntraces + chunk - 1:
                slc = slice(slc_end - chunk, min(slc_end, ntraces), 1)
                # take headers returned from segyio and create lists for a dataframe
                head_df.iloc[slc, :] = np.vstack(
                    [
                        list(next(segyf_hgen).values())
                        for _ in range(slc.stop - slc.start)
                    ]
                )[:, bytes_filter_index]
                slc_end += chunk
                pbar.update(slc.stop - slc.start)

    head_df.replace(to_replace=-2147483648, value=np.nan, inplace=True)
    return head_df


def what_geometry_am_i(head_df):
    """Try to work out file type from a header scan.

    This is a limited method to try and determine a file geometry type from
    a header scan. It will often fail.
    """

    ntraces = head_df.shape[0]

    stats = head_df.describe().T
    stats = stats[stats["std"] > 0]

    header_dict = _active_tracefield_segyio()
    pre_cols = list(stats.columns)
    stats["byte_loc"] = [
        header_dict[key] if key in header_dict.keys() else np.nan for key in stats.index
    ]
    stats = stats[["byte_loc"] + pre_cols]

    # find n unqiue
    stats["unique"] = [head_df[key].unique().size for key in stats.index]

    byte_call = dict()

    # check if offset
    if "offset" in stats.index:
        byte_call["offset"] = 37
        nof = stats.loc["offset", "unique"]
    else:
        nof = 1

    # check for xy
    if {"SourceX", "SourceY"}.issubset(set(stats.index)):
        byte_call["cdpx"] = 73
        byte_call["cdpy"] = 77
    elif {"CDP_X", "CDP_Y"}.issubset(set(stats.index)):
        has_xy = True
        byte_call["cdpx"] = 181
        byte_call["cdpy"] = 185

    # check for ilxl
    if {"INLINE_3D", "CROSSLINE_3D"}.issubset(set(stats.index)):
        byte_call["iline"] = 189
        byte_call["xline"] = 193
        n = stats.loc["INLINE_3D", "unique"] * stats.loc["CROSSLINE_3D", "unique"]
    elif "CDP" in stats.index:
        byte_call["cdp"] = 21
        n = stats.loc["CDP", "unique"]
    else:
        byte_call["cdp"] = 5
        n = stats["TRACE_SEQUENCE_FILE", "unique"]

    # sanity check number
    if nof * n != ntraces:
        raise ValueError(
            f"Couldn't determin geometry set byte locations manually. \n {print(stats)}"
        )

    # print(has_offsets)
    return byte_call
