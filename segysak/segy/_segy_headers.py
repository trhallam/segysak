from typing import Dict, Union, List, Generator, Any
import os
import numpy as np
import pandas as pd
import segyio

from ._segy_core import (
    _active_tracefield_segyio,
    _active_binfield_segyio,
    check_tracefield,
    check_tracefield_names,
)

from ..progress import Progress


class TraceHeaders:
    """A convenience class for accessing and iterating over a SEG-Y files trace
    headers. This class should be used with a context manager.

    Examples:

        >>> with TraceHeaders(segy_file, bytes_filter=bytes_filter, **segyio_kwargs) as headers:
                ntraces = headers.ntraces
                df = headers.to_dataframe(selection=slice(0, 100)))

    """

    def __init__(
        self,
        segy_file: Union[str, os.PathLike],
        bytes_filter: Union[List[int], None] = None,
        tracefield_filter: Union[List[str], None] = None,
        **segyio_kwargs: Any,
    ):

        check_tracefield(bytes_filter)
        check_tracefield_names(tracefield_filter)

        self.filter = self._combine_filters(bytes_filter, tracefield_filter)

        self.bytes_filter = bytes_filter

        self.segy_file = segy_file
        _segyio_kwargs = segyio_kwargs.copy()
        _segyio_kwargs.update({"ignore_geometry": True})
        self.fh = segyio.open(self.segy_file, "r", **_segyio_kwargs)
        self.ntraces = self.fh.tracecount

    def _combine_filters(
        self,
        bytes_filter: Union[List[int], None],
        tracefield_filter: Union[List[str], None],
    ) -> List[segyio.tracefield.TraceField]:

        filter_list = []
        if bytes_filter is not None:
            filter_list += [
                segyio.tracefield.TraceField(byte_loc) for byte_loc in bytes_filter
            ]

        if tracefield_filter is not None:
            filter_list += [
                segyio.tracefield.TraceField(segyio.tracefield.keys[key])
                for key in tracefield_filter
            ]

        if filter_list:
            filter_list = list(set(filter_list))
        else:
            filter_list = [
                segyio.tracefield.TraceField(byte)
                for byte in segyio.tracefield.keys.values()
            ]

        return filter_list

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.fh.close()

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        return self[:]

    def __getitem__(
        self, i: Union[int, slice]
    ) -> Generator[Dict[str, Any], None, None]:
        silent = Progress._segysak_tqdm_kwargs["disable"]
        if isinstance(i, int):
            silent = True
            n = 1
        else:
            silent = False
            n = len(range(*i.indices(self.ntraces)))

        with Progress(unit=" traces", total=n) as pbar:
            for header in self.fh.header[i]:
                pbar.update(1)
                yield {key: header[key] for key in self.filter}

    def to_dataframe(self, selection: Union[int, slice, None] = None) -> pd.DataFrame:
        """Return the Trace Headers as a DataFrame

        Args:
            selection: A subset of trace headers will be returned based on trace numbering.
        """
        if isinstance(selection, int):
            index = pd.Index(range(i, i + 1))
        elif isinstance(selection, slice):
            index = pd.Index(range(*selection.indices(self.ntraces)))
        else:
            index = pd.Index(range(self.ntraces))
            selection = slice(None, None, None)

        columns = tuple(str(f) for f in self.filter)

        head_df = pd.DataFrame(index=index, columns=columns)
        # This is slightly faster than building from dicts
        head_df.iloc[:, :] = np.vstack([list(h.values()) for h in self[selection]])

        # fix bad values
        # head_df = head_df.replace(to_replace=-2147483648, value=np.nan)
        # convert numeric
        for col in head_df:
            head_df[col] = pd.to_numeric(head_df[col], downcast="integer")

        return head_df


def segy_header_scan(
    segy_file: Union[str, os.PathLike],
    max_traces_scan: int = 1000,
    **segyio_kwargs: Any,
) -> pd.DataFrame:
    """Perform a scan of the segy file headers and return ranges.

    To get the complete raw header values see `segy_header_scrape`

    Args:
        segy_file: SEG-Y file path
        max_traces_scan: Number of traces to scan. For scan all traces set to <= 0. Defaults to 1000.
        segyio_kwargs: Arguments passed to segyio.open

    Returns:
        Uses pandas describe to return statistics of your headers.
    """
    if max_traces_scan <= 0:
        max_traces_scan = None
    else:
        if not isinstance(max_traces_scan, int):
            raise ValueError("max_traces_scan must be int")

    head_df = segy_header_scrape(segy_file, max_traces_scan, **segyio_kwargs)

    header_keys = head_df.describe().T
    pre_cols = list(header_keys.columns)
    header_keys["byte_loc"] = [segyio.tracefield.keys[key] for key in header_keys.index]
    header_keys = header_keys[["byte_loc"] + pre_cols]
    header_keys.nscan = head_df.shape[0]
    return header_keys


def segy_bin_scrape(segy_file: Union[str, os.PathLike], **segyio_kwargs: Any) -> Dict:
    """Scrape binary header

    Args:
        segy_file: SEG-Y file path
        segyio_kwargs: Arguments passed to segyio.open

    Returns:
        Binary header key value pairs
    """
    bk = _active_binfield_segyio()
    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segy_file, "r", **segyio_kwargs) as segyf:
        return {key: segyf.bin[item] for key, item in bk.items()}


def segy_header_scrape(
    segy_file: Union[str, os.PathLike],
    partial_scan: Union[int, None] = None,
    bytes_filter: Union[List[int], None] = None,
    chunk: int = 100_000,
    **segyio_kwargs: Any,
) -> pd.DataFrame:
    """Scape all data from segy trace headers

    Args:
        segy_file: SEG-Y File path
        partial_scan: Setting partial scan to a positive int will scan only
            that many traces. Defaults to None.
        bytes_filter: List of byte locations to load exclusively.
        chunk: Number of traces to read in one go.
        segyio_kwargs: Arguments passed to segyio.open

    Returns:
        pandas.DataFrame: Raw header information in table for scanned traces.
    """
    with TraceHeaders(segy_file, bytes_filter=bytes_filter, **segyio_kwargs) as headers:
        if partial_scan is not None:
            ntraces = partial_scan
        else:
            ntraces = headers.ntraces

        chunks = ntraces // chunk + min(ntraces % chunk, 1)
        _dfs = []
        with Progress(unit=" trace-chunks", total=chunks) as pbar:
            for chk in range(0, chunks):
                chk_slc = slice(chk * chunk, min((chk + 1) * chunk, ntraces), None)
                _dfs.append(headers.to_dataframe(selection=chk_slc))
                pbar.update(1)

    head_df = pd.concat(_dfs)
    return head_df


def header_as_dimensions(head_df: pd.DataFrame, dims: tuple) -> Dict[str, np.array]:
    """Convert dim_kwargs to a dictionary of dimensions. Also useful for checking
    geometry is correct and unique for each trace in a segy file header.

    Args:
        head_df: The header DataFrame from `segy_header_scrape`.
        dims: Dimension names as per head_df column names.

    Returns:
        dims: Dimension name and label pairs.
    """
    unique_dims = dict()
    for dim in dims:
        # get unique values of dimension and sort them ascending
        as_unique = head_df[dim].unique()
        unique_dims[dim] = np.sort(as_unique)

    if head_df[list(dims)].shape != head_df[list(dims)].drop_duplicates().shape:
        raise ValueError(
            "The selected dimensions results in multiple traces per "
            "dimension location, add additional dimensions or use "
            "trace numbering byte location to load as 2D."
        )

    return unique_dims


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
