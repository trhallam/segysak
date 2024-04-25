from typing import Dict, Union, List
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


def segy_header_scan(
    segyfile: str, max_traces_scan: int = 1000, silent: bool = False, **segyio_kwargs
) -> pd.DataFrame:
    """Perform a scan of the segy file headers and return ranges.

    To get the complete raw header values see `segy_header_scrape`

    Args:
        segyfile: Segy File Path
        max_traces_scan: Number of traces to scan.
            For scan all traces set to <= 0. Defaults to 1000.
        silent: Disable progress bar.

    Returns:
        Uses pandas describe to return statistics of your headers.
    """
    if max_traces_scan <= 0:
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


def segy_bin_scrape(segyfile: str, **segyio_kwargs) -> Dict:
    """Scrape binary header

    Args:
        segyfile: SEG-Y File path

    Returns:
        Binary header
    """
    bk = _active_binfield_segyio()
    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        return {key: segyf.bin[item] for key, item in bk.items()}


def segy_header_scrape(
    segyfile: str,
    partial_scan: Union[int, None] = None,
    silent: bool = False,
    bytes_filter: Union[List[int], None] = None,
    chunk: int = 100_000,
    **segyio_kwargs,
) -> pd.DataFrame:
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
    else:
        bytes_filter_index = list(enum_byte_index.values())

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

    for col in head_df:
        head_df[col] = pd.to_numeric(head_df[col], downcast="unsigned")

    return head_df


def header_as_dimensions(head_df: pd.DataFrame, **dim_kwargs) -> Dict[str, np.array]:
    """Convert dim_kwargs to a diction of dimensions. Also useful for checking
    geometry is correct and unique for each trace in a segy file header.

    Args:
        head_df: The header Dataframe from `segy_header_scrape`.
        dim_kwargs: Dimension names (str) and byte location (int) pairs. Byte
            locations must be valid segy byte locations for the trace headers.
    """
    # creating dimensions and new dataset
    dims = dict()
    dim_index_names = list()
    dim_fields = list()

    for dim in dim_kwargs:
        # check the dimension byte location to get the head_df column name
        trace_field = str(segyio.TraceField(dim_kwargs[dim]))
        if trace_field == "Unknown Enum":
            raise ValueError(f"{dim}:{dim_kwargs[dim]} was not a valid byte header")
        dim_fields.append(trace_field)

        # get unique values of dimension and sort them ascending
        as_unique = head_df[trace_field].unique()
        dims[dim] = np.sort(as_unique)

    if (
        head_df[dim_index_names].shape
        != head_df[dim_index_names].drop_duplicates().shape
    ):
        raise ValueError(
            "The selected dimensions results in multiple traces per "
            "dimension location, add additional dimensions or use "
            "trace numbering byte location to load as 2D."
        )

    return dims


def what_geometry_am_i(head_df: pd.DataFrame):
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
        byte_call["cdp_x"] = 73
        byte_call["cdp_y"] = 77
    elif {"CDP_X", "CDP_Y"}.issubset(set(stats.index)):
        has_xy = True
        byte_call["cdp_x"] = 181
        byte_call["cdp_y"] = 185

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
            f"SEGYSAK couldn't determine geometry set byte locations manually. \n {print(stats)}"
        )

    # print(has_offsets)
    return byte_call
