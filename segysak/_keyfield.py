"""Enumerations of dataset keys.
"""
from enum import Enum


class CoordKeyField(Enum):

    cdp = "cdp"
    iline = "iline"
    xline = "xline"
    offset = "offset"
    depth = "depth"
    twt = "twt"
    cdp_x = "cdp_x"
    cdp_y = "cdp_y"
    lat = "lat"
    long = "lon"


class DimensionKeyField(Enum):

    cdp_3d = ("iline", "xline")
    cdp_2d = ("cdp",)
    threed_twt = ("iline", "xline", "twt")
    threed_depth = ("iline", "xline", "depth")
    threed_head = ("iline", "xline")
    threed_ps_twt = ("iline", "xline", "twt", "offset")
    threed_ps_depth = ("iline", "xline", "depth", "offset")
    threed_ps_head = ("iline", "xline", "offset")
    twod_twt = ("cdp", "twt")
    twod_depth = ("cdp", "depth")
    twod_head = ("cdp",)
    twod_ps_twt = ("cdp", "twt", "offset")
    twod_ps_depth = ("cdp", "depth", "offset")
    twod_ps_head = ("cdp", "offset")


class VariableKeyField(Enum):
    data = "data"


class AttrKeyField(Enum):

    ns = "ns"
    ds = "ds"
    text = "text"
    d3_units = "d3_units"
    d3_domain = "d3_domain"
    epsg = "epsg"
    corner_points = "corner_points"
    corner_points_xy = "corner_points_xy"
    source_file = "source_file"
    srd = "srd"
    datatype = "datatype"


class VerticalKeyField(Enum):

    twt = "TWT"
    depth = "DEPTH"


class VerticalUnits(Enum):

    ms = "ms"  # milliseconds
    s = "s"  # seconds
    cm = "cm"  # centimetres
    m = "m"  # metres
    km = "km"  # kilometres
    ft = "ft"  # feet
