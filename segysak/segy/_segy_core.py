import segyio
import importlib

try:
    has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None
    if has_ipywidgets:
        from tqdm.autonotebook import tqdm
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


def check_tracefield(byte_list=None):
    """Check that the byte fields requested by the user are valid Enums in segyio

    Args:
        byte_list (list, optional): List of int byte fields.
    """
    if byte_list is None:
        return True

    tracefield_keys = header_keys = segyio.tracefield.keys.copy()
    failed = [
        byte_loc
        for byte_loc in byte_list
        if byte_loc not in list(tracefield_keys.values())
    ]
    if failed:
        raise ValueError(f"Invalid byte locations: {failed}")

    return True