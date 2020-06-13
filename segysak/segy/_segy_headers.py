import importlib
import numpy as np
import pandas as pd
import segyio

try:
    has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None
    if has_ipywidgets:
        from tqdm.autonotebook import tqdm
    else:
        from tqdm import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm


TQDM_ARGS = dict(unit_scale=True, unit=" traces")


def _active_tracefield_segyio():
    header_keys = segyio.tracefield.keys.copy()
    # removed unused byte locations
    _ = header_keys.pop("UnassignedInt1")
    _ = header_keys.pop("UnassignedInt2")
    return header_keys


def _active_binfield_segyio():
    bin_keys = segyio.binfield.keys.copy()
    _ = bin_keys.pop("Unassigned1")
    _ = bin_keys.pop("Unassigned2")
    return bin_keys


def segy_header_scan(segyfile, max_traces_scan=1000, silent=False, **segyio_kwargs):
    """Perform a scan of the segy file headers and return ranges.

    Args:
        segyfile (string): Segy File Path
        max_traces_scan (int/str, optional): Number of traces to scan.
            For all set to 0 or 'all'. Defaults to 1000.
        silent (bool): Disable progress bar.

    Returns:
        dict: Contains keys from segyio.tracefield.keys with scanned values in
        a list [byte location, min, max]
        int: Number of scanned traces
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
    header_keys["byte_loc"] = list(_active_tracefield_segyio().values())
    header_keys.nscan = head_df.shape[0]
    return header_keys


def segy_bin_scrape(segyfile, **segyio_kwargs):
    """Scrape binary header

    Args:
        segyfile (str): SEGY File path

    Returns:
        dict: Binary header
    """
    bk = _active_binfield_segyio()
    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        return {key: segyf.bin[item] for key, item in bk.items()}


def segy_header_scrape(segyfile, partial_scan=None, silent=False, **segyio_kwargs):
    """Scape all data from segy trace headers

    Args:
        segyfile (str): SEGY File path
        partial_scan (int): Setting partial scan to a positive int will scan only
            that many traces. Defaults to None.
        silent (bool): Disable progress bar.

    Returns:
        (pandas.DataFrame): Header information
    """
    header_keys = _active_tracefield_segyio()
    columns = header_keys.keys()
    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        ntraces = segyf.tracecount
        if partial_scan is not None:
            ntraces = int(partial_scan)
        slc = slice(0, ntraces, 1)
        # take headers returned from segyio and create lists for a dataframe
        hv = map(
            lambda x: np.array(list(x.values())),
            tqdm(segyf.header[slc], total=ntraces, disable=silent, **TQDM_ARGS),
        )
        head_df = pd.DataFrame(np.vstack(list(hv)), columns=columns)
        head_df.replace(to_replace=-2147483648, value=np.nan, inplace=True)
    return head_df
