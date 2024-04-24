"""Enumerations of dataset keys.
"""

from typing import Tuple
from dataclasses import dataclass
from ._core import FrozenDict


class MetaDataClass:

    def __iter__(self):
        for match in self.__match_args__:
            yield match

    def keys(self):
        return self.__match_args__

    def values(self):
        for match in self.__match_args__:
            yield self.__getattr__(match)

    def items(self):
        return {key: value for key, value in zip(self.keys(), self.values())}

    def __getitem__(self, item):
        return self.__dict__[item]


@dataclass(order=True, frozen=True)
class _CoordKeyField(MetaDataClass):
    """Coordinate Keys to use for naming key components of the Dataset"""

    cdp: str = "cdp"
    iline: str = "iline"
    xline: str = "xline"
    offset: str = "offset"
    depth: str = "depth"
    twt: str = "twt"
    cdp_x: str = "cdp_x"
    cdp_y: str = "cdp_y"
    lat: str = "lat"
    long: str = "lon"


CoordKeyField = _CoordKeyField()


@dataclass(order=True, frozen=True)
class _DimensionKeyField(MetaDataClass):
    """Dimensionality Keys to use for naming components of the Dataset"""

    cdp_3d: Tuple[str] = (CoordKeyField.iline, CoordKeyField.xline)
    cdp_2d: Tuple[str] = (CoordKeyField.cdp,)
    threed_twt: Tuple[str] = (
        CoordKeyField.iline,
        CoordKeyField.xline,
        CoordKeyField.twt,
    )
    threed_xline_twt: Tuple[str] = (CoordKeyField.iline, CoordKeyField.twt)
    threed_iline_twt: Tuple[str] = (CoordKeyField.xline, CoordKeyField.twt)
    threed_depth: Tuple[str] = (
        CoordKeyField.iline,
        CoordKeyField.xline,
        CoordKeyField.depth,
    )
    threed_xline_depth: Tuple[str] = (CoordKeyField.iline, CoordKeyField.depth)
    threed_iline_depth: Tuple[str] = (CoordKeyField.xline, CoordKeyField.depth)
    threed_head: Tuple[str] = (CoordKeyField.iline, CoordKeyField.xline)
    threed_ps_twt: Tuple[str] = (
        CoordKeyField.iline,
        CoordKeyField.xline,
        CoordKeyField.twt,
        CoordKeyField.offset,
    )
    threed_ps_depth: Tuple[str] = (
        CoordKeyField.iline,
        CoordKeyField.xline,
        CoordKeyField.depth,
        CoordKeyField.offset,
    )

    threed_ps_head: Tuple[str] = (
        CoordKeyField.iline,
        CoordKeyField.xline,
        CoordKeyField.offset,
    )
    twod_twt: Tuple[str] = (CoordKeyField.cdp, CoordKeyField.twt)
    twod_depth: Tuple[str] = (CoordKeyField.cdp, CoordKeyField.depth)
    twod_head: Tuple[str] = (CoordKeyField.cdp,)
    twod_ps_twt: Tuple[str] = (
        CoordKeyField.cdp,
        CoordKeyField.twt,
        CoordKeyField.offset,
    )
    twod_ps_depth: Tuple[str] = (
        CoordKeyField.cdp,
        CoordKeyField.depth,
        CoordKeyField.offset,
    )
    twod_ps_head: Tuple[str] = (CoordKeyField.cdp, CoordKeyField.offset)


DimensionKeyField = _DimensionKeyField()


@dataclass(frozen=True)
class _VariableKeyField(MetaDataClass):
    data: str = "data"


VariableKeyField = _VariableKeyField()


@dataclass(frozen=True)
class _AttrKeyField(MetaDataClass):
    ns: str = "ns"
    sample_rate: str = "sample_rate"
    text: str = "text"
    measurement_system: str = "measurement_system"
    d3_domain: str = "d3_domain"
    epsg: str = "epsg"
    corner_points: str = "corner_points"
    corner_points_xy: str = "corner_points_xy"
    source_file: str = "source_file"
    srd: str = "srd"
    datatype: str = "datatype"
    percentiles: str = "percentiles"
    coord_scalar: str = "coord_scalar"


AttrKeyField = _AttrKeyField()


@dataclass(frozen=True)
class _VerticalKeyField(MetaDataClass):
    twt: str = "TWT"
    depth: str = "DEPTH"


VerticalKeyField = _VerticalKeyField()


@dataclass(frozen=True)
class _VerticalKeyDim(MetaDataClass):
    TWT: str = "twt"
    DEPTH: str = "depth"


VerticalKeyDim = _VerticalKeyDim()


@dataclass(frozen=True)
class _VerticalUnits(MetaDataClass):
    ms: str = "ms"  # milliseconds
    s: str = "s"  # seconds
    cm: str = "cm"  # centimetres
    m: str = "m"  # metres
    km: str = "km"  # kilometres
    ft: str = "ft"  # feet


VerticalUnits = _VerticalUnits()
