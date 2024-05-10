"""Enumerations of dataset keys.
"""

from typing import Tuple
from dataclasses import dataclass
from ._core import MetaDataClass


@dataclass(order=True, frozen=True)
class _HorDimKeyField(MetaDataClass):
    """Horizontal Dimensions of seismic data."""

    cdp: str = "cdp"
    iline: str = "iline"
    xline: str = "xline"


HorDimKeyField = _HorDimKeyField()


@dataclass(order=True, frozen=True)
class _VerDimKeyField(MetaDataClass):
    """Vertical Dimensions of seismic data."""

    depth: str = "depth"
    twt: str = "twt"
    samples: str = "samples"


VerDimKeyField = _VerDimKeyField()


@dataclass(order=True, frozen=True)
class _DimKeyField(_HorDimKeyField, _VerDimKeyField):
    """Dimension keys to use for naming."""

    offset: str = "offset"
    angle: str = "angle"


DimKeyField = _DimKeyField()


@dataclass(order=True, frozen=True)
class _CoordKeyField(MetaDataClass):
    """Coordinate Keys to use for naming key components of the Dataset"""

    cdp_x: str = "cdp_x"
    cdp_y: str = "cdp_y"
    lat: str = "lat"
    long: str = "lon"


CoordKeyField = _CoordKeyField()


@dataclass(order=True, frozen=True)
class _DimensionKeyField(MetaDataClass):
    """Dimensionality Keys to use for naming components of the Dataset"""

    cdp_3d: Tuple[str] = (DimKeyField.iline, DimKeyField.xline)
    cdp_2d: Tuple[str] = (DimKeyField.cdp,)
    threed_twt: Tuple[str] = (
        DimKeyField.iline,
        DimKeyField.xline,
        DimKeyField.twt,
    )
    threed_xline_twt: Tuple[str] = (DimKeyField.iline, DimKeyField.twt)
    threed_iline_twt: Tuple[str] = (DimKeyField.xline, DimKeyField.twt)
    threed_depth: Tuple[str] = (
        DimKeyField.iline,
        DimKeyField.xline,
        DimKeyField.depth,
    )
    threed_xline_depth: Tuple[str] = (DimKeyField.iline, DimKeyField.depth)
    threed_iline_depth: Tuple[str] = (DimKeyField.xline, DimKeyField.depth)
    threed_head: Tuple[str] = (DimKeyField.iline, DimKeyField.xline)
    threed_ps_twt: Tuple[str] = (
        DimKeyField.iline,
        DimKeyField.xline,
        DimKeyField.twt,
        DimKeyField.offset,
    )
    threed_ps_depth: Tuple[str] = (
        DimKeyField.iline,
        DimKeyField.xline,
        DimKeyField.depth,
        DimKeyField.offset,
    )

    threed_ps_head: Tuple[str] = (
        DimKeyField.iline,
        DimKeyField.xline,
        DimKeyField.offset,
    )
    twod_twt: Tuple[str] = (DimKeyField.cdp, DimKeyField.twt)
    twod_depth: Tuple[str] = (DimKeyField.cdp, DimKeyField.depth)
    twod_head: Tuple[str] = (DimKeyField.cdp,)
    twod_ps_twt: Tuple[str] = (
        DimKeyField.cdp,
        DimKeyField.twt,
        DimKeyField.offset,
    )
    twod_ps_depth: Tuple[str] = (
        DimKeyField.cdp,
        DimKeyField.depth,
        DimKeyField.offset,
    )
    twod_ps_head: Tuple[str] = (DimKeyField.cdp, DimKeyField.offset)


DimensionKeyField = _DimensionKeyField()


@dataclass(order=True, frozen=True)
class _CardinalityKeyField(MetaDataClass):
    """Cardinality Keys to use for naming seisnc Data

    This is the type of data in terms of horizontal coordinates like iline/xline, or iline/xline/offset.
    This avoids the vertical dimension which is handled separately.
    """

    seisnc_3d: Tuple[str] = (DimKeyField.iline, DimKeyField.xline)
    seisnc_3doff: Tuple[str] = (
        DimKeyField.iline,
        DimKeyField.xline,
        DimKeyField.offset,
    )
    seisnc_3dang: Tuple[str] = (
        DimKeyField.iline,
        DimKeyField.xline,
        DimKeyField.angle,
    )
    seisnc_2d: Tuple[str] = (DimKeyField.cdp,)
    seisnc_2doff: Tuple[str] = (
        DimKeyField.cdp,
        DimKeyField.offset,
    )
    seisnc_2dang: Tuple[str] = (
        DimKeyField.cdp,
        DimKeyField.angle,
    )


CardinalityKeyField = _CardinalityKeyField()


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
    coord_scaled: str = "coord_scaled"
    dimensions: str = "dimensions"
    vert_dimension: str = "vert_dimension"
    vert_domain: str = "vert_domain"


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
    samples: str = "samples"


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
