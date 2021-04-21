"""Enumerations of dataset keys.
"""

from ._core import FrozenDict

CoordKeyField = FrozenDict(
    dict(
        cdp="cdp",
        iline="iline",
        xline="xline",
        offset="offset",
        depth="depth",
        twt="twt",
        cdp_x="cdp_x",
        cdp_y="cdp_y",
        lat="lat",
        long="lon",
    )
)

DimensionKeyField = FrozenDict(
    dict(
        cdp_3d=("iline", "xline"),
        cdp_2d=("cdp",),
        threed_twt=("iline", "xline", "twt"),
        threed_xline_twt=("iline", "twt"),
        threed_iline_twt=("xline", "twt"),
        threed_depth=("iline", "xline", "depth"),
        threed_xline_depth=("iline", "depth"),
        threed_iline_depth=("xline", "depth"),
        threed_head=("iline", "xline"),
        threed_ps_twt=("iline", "xline", "twt", "offset"),
        threed_ps_depth=("iline", "xline", "depth", "offset"),
        threed_ps_head=("iline", "xline", "offset"),
        twod_twt=("cdp", "twt"),
        twod_depth=("cdp", "depth"),
        twod_head=("cdp",),
        twod_ps_twt=("cdp", "twt", "offset"),
        twod_ps_depth=("cdp", "depth", "offset"),
        twod_ps_head=("cdp", "offset"),
    )
)

VariableKeyField = FrozenDict(dict(data="data"))


AttrKeyField = FrozenDict(
    dict(
        ns="ns",
        sample_rate="sample_rate",
        text="text",
        measurement_system="measurement_system",
        d3_domain="d3_domain",
        epsg="epsg",
        corner_points="corner_points",
        corner_points_xy="corner_points_xy",
        source_file="source_file",
        srd="srd",
        datatype="datatype",
        percentiles="percentiles",
        coord_scalar="coord_scalar",
    )
)


VerticalKeyField = FrozenDict(dict(twt="TWT", depth="DEPTH"))
VerticalKeyDim = FrozenDict(dict(TWT="twt", DEPTH="depth"))


VerticalUnits = FrozenDict(
    dict(
        ms="ms",  # milliseconds
        s="s",  # seconds
        cm="cm",  # centimetres
        m="m",  # metres
        km="km",  # kilometres
        ft="ft",  # feet
    )
)
