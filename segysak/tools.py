def check_crop(crop, limits):
    """Maxke sure mins and maxs are good.
    
    Args:
        crop (list): Input crop [min_il, max_il, min_xl, max_xl]
        limits (list): Input data limits [min_il, max_il, min_xl, max_xl]
    """
    checked = limits.copy()

    # check crop
    if crop[0] > crop[1] or crop[2] > crop[3]:
        raise ValueError('Crop min is greater than max')
    if limits[0] > limits[1] or limits[2] > limits[3]:
        raise ValueError('Limits min is greater than max')
    
    # crop
    if limits[0] < crop[0] < limits[1]: checked[0] = crop[0]
    if limits[0] < crop[1] < limits[1]: checked[1] = crop[1]
    if limits[2] < crop[2] < limits[3]: checked[2] = crop[2]
    if limits[2] < crop[3] < limits[3]: checked[3] = crop[3]

    return checked