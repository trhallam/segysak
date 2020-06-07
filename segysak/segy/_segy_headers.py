import importlib
import numpy as np
import pandas as pd
import segyio

try:
    has_ipywidgets = importlib.find_loader("ipywidgets") is not None
    if has_ipywidgets:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm


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

    hi = 0

    segyio_kwargs["ignore_geometry"] = True

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        header_keys = _active_tracefield_segyio()
        lh = len(header_keys.keys())
        hmin = np.full(lh, np.nan)
        hmax = np.full(lh, np.nan)
        for hi, h in enumerate(
            tqdm(
                segyf.header,
                desc="Scanning Headers",
                total=max_traces_scan,
                disable=silent,
            )
        ):
            val = np.array(list(h.values()))
            hmin = np.nanmin(np.vstack((hmin, val)), axis=0)
            hmax = np.nanmax(np.vstack((hmax, val)), axis=0)
            if max_traces_scan is not None and hi + 1 >= max_traces_scan:
                break  # scan to max_traces_scan

    for i, (key, item) in enumerate(header_keys.items()):
        header_keys[key] = [item, hmin[i], hmax[i]]

    return header_keys, hi + 1


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


def segy_header_scrape(segyfile, silent=False, **segyio_kwargs):
    """Scape all data from segy trace headers

    Args:
        segyfile (str): SEGY File path
        silent (bool): Disable progress bar.

    Returns:
        (pandas.DataFrame): Header information
    """
    header_keys = _active_tracefield_segyio()
    columns = header_keys.keys()
    lc = len(columns)
    segyio_kwargs["ignore_geometry"] = True
    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:
        ntraces = segyf.tracecount
        head_df = pd.DataFrame(
            data=np.full((ntraces, lc), np.nan, dtype=int), columns=columns
        )
        pb = tqdm(total=segyf.tracecount, desc="Scraping Headers", disable=silent)
        for hi, h in enumerate(segyf.header):
            head_df.iloc[hi, :] = np.array(list(h.values()))
            pb.update()
        pb.close()
        head_df.replace(to_replace=-2147483648, value=np.nan, inplace=True)
    return head_df
