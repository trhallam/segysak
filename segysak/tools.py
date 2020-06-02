
import os
import datetime

def _check_crop(crop, limits):
    """Make sure a pair of limits is cropped correctly.

    Args:
        crop (list): Crop range [min, max]
        limits (list): Limits to crop [min, max]
    """
    if crop[0] > crop[1]:
        raise ValueError(f"Crop min: '{crop[0]}' is greater than crop max: '{crop[1]}'")
    if limits[0] > limits[1]:
        raise ValueError(f"Limits min: '{limits[0]}' is greater than limits max: '{limits[1]}'")
    if limits[0] > crop[1] or limits[1] < crop[0]:
        raise ValueError(f"Crop range is outside data limits")

def _crop(crop, limits):
    """Perform cropping operation.

    Args:
        crop (list): Cropping limits [min, max]
        limits (list): Limits to crop [min, max]

    Returns:
        list: Cropped limits [min, max]
    """
    checked = limits.copy()
    if limits[0] < crop[0] < limits[1]: checked[0] = crop[0]
    if limits[0] < crop[1] < limits[1]: checked[1] = crop[1]
    return checked

def check_crop(crop, limits):
    """Make sure mins and maxs or horizontal cropping box are good.

    Args:
        crop (list): Input crop [min_il, max_il, min_xl, max_xl]
        limits (list): Input data limits [min_il, max_il, min_xl, max_xl]

    Returns:
        list: cropped limits [min_il, max_il, min_xl, max_xl]
    """
    checked = limits.copy()

    # check crop
    _check_crop(crop[:2], limits[:2])
    _check_crop(crop[2:], limits[2:])

    # crop
    checked[:2] = _crop(crop[:2], limits[:2])
    checked[2:] = _crop(crop[2:], limits[2:])

    return checked

def check_zcrop(crop, limits):
    """Make sure mins and maxs of vertical cropping box are good.

    Args:
        crop (list): Input crop [min, max]
        limits (list): Input data limits to crop [min, max]

    Output:
        list: cropped limits
    """
    _check_crop(crop, limits)
    return _crop(crop, limits)

def fix_bad_chars(text, sub='#'):
    """Remove unencodeable characters. And replace them with sub.

    Args:
        text (str): Text string to fix.
        sub (str): Sting to replace bad chars.

    Returns:
        (str): Cleaned text string.
    """
    good_text = ""
    for i, char in enumerate(text):
        if len(char.encode()) == 1:
            good_text = good_text + char
        else:
            good_text = good_text + "#"
    return good_text

def _get_datetime():
    """Return current date and time as formatted strings

    Returns:
        str, str: Date "YYYY-MM-DD" Time "HH:MM:SS"
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

def _get_userid():
    """Try to get the user id for reporting.

    Returns:
        str: The current user login ID.
    """
    try:
        return os.getlogin()
    except OSError: # couldn't get it this way
        pass
    finally:
        return "segysak_user"