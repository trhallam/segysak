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

    cdp_3d = ("d1", "d2")
    cdp_2d = ("d1",)
    threed = ("d1", "d2", "d3")
    threed_head = ("d1", "d2")
    threed_ps = ("d1", "d2", "d3", "d4")
    threed_ps_head = ("d1", "d2", "d4")
    twod = ("d1", "d3")
    twod_head = ("d1",)
    twod_ps = ("d1", "d3", "d4")
    twod_ps_head = ("d1", "d4")
    cdp = "cdp"
    iline = "iline"
    xline = "xline"
    offset = "offset"
    twt = "twt"
    depth = "depth"


class DimensionLabelKeyField(Enum):

    threed = {"d1": "iline", "d2": "xline", "d3": "twt"}
    threed_ps = {"d1": "iline", "d2": "xline", "d3": "twt", "d4": "offset"}
    twod = {"d1": "cdp", "d3": "twt"}
    twod_ps = {"d1": "cdp", "d3": "twt", "d4": "offset"}


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
