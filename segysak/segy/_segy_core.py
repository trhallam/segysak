from typing import Union, List
import importlib
import numpy as np
import segyio
import segyio.trace

try:
    has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None
    if has_ipywidgets:
        from tqdm.auto import tqdm
    else:
        from tqdm import tqdm as tqdm
except ModuleNotFoundError:
    from tqdm import tqdm as tqdm


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


def check_tracefield(byte_list: Union[list, None] = None) -> bool:
    """Check that the byte fields requested by the user are valid Enums in segyio

    Args:
        byte_list: List of int byte fields.
    """
    if byte_list is None:
        return True

    tracefield_keys = segyio.tracefield.keys.copy()
    tracefield_bytes = list(tracefield_keys.values())
    failed = [byte_loc for byte_loc in byte_list if byte_loc not in tracefield_bytes]
    if failed:
        raise ValueError(f"Invalid byte locations: {failed}")

    return True


def check_tracefield_names(field_list: Union[List[str], None] = None) -> bool:
    """Check that tracefield names match segyio TraceField names.

    Args:
        field_list: List of field names.

    Returns:
        True if valid field_list.

    Raises:
        ValueError:
    """
    if field_list is None:
        return True

    failed = [field for field in field_list if field not in segyio.tracefield.keys]
    if failed:
        raise ValueError(f"Invalid field names: {failed}")

    return True


def sample_range(ns0: float, sample_rate: float, nsamp: int) -> np.array:
    """Return the samples or SEGY file.

    Args:
        ns0: First sample value.
        sample_rate: The sampling interval.
        nsamp: The number of samples in the range.
    """
    return np.arange(ns0, ns0 + sample_rate * nsamp, sample_rate, dtype=np.float32)
