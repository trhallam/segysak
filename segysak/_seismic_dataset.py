# pylint: disable=invalid-name,no-member
"""
Basic definitions and tools for the creation of seismic datasets. This is an
xarray.Dataset with particular dimensions, coordinates and attributes that inform
other tools and file format converters in segysak library.

"""

import numpy as np

from ._keyfield import CoordKeyField, VerticalUnits, AttrKeyField, VerticalKeyField
from ._accessor import xr


def create_seismic_dataset(
    twt=None, depth=None, cmp=None, iline=None, xline=None, offset=None, **dim_args
):
    """Create a blank seismic dataset by setting the dimension sizes (d#) or by passing
    arrays for known dimensions.

    Args:
        dim_args (int): With keys d1, d2, d3, .., dN. The dimension sizes if extra
            dimensions are needed or local coordinates are unknown.
        twt (int/float/array-like, optional): Two-way time vertical sampling coordinates, this will fill dimension d3.
            Cannot be used with depth argument. Defaults to None.
        depth (int/float/array-like, optional): Depth vertical sampling coordinates, this will fill dimension d3.
            Cannot be used with twt argument. Defaults to None.
        cmp (int/float/array-like, optional): The CMP numbering, this will fill dimension d1 and cannot be used
            with iline or xline. Use for 2D seismic data. Defaults to None.
        iline (int/float/array-like, optional): The iline numbering, this will fill dimension d1 and cannot be
            used with cmp argument. Use for 3D seismic data. Defaults to None.
        xline (int/float/array-like, optional): The xline numbering, this will fill dimension d2 and cannot be
            used with cmp argument. Use for 3D seismic data. Defaults to None.
        offset (int/float/array-like, optional): The offset. This will fill dimension d4.
            Use for pre-stack data. Defaults to None.

    Returns:
        xarray.Dataset: A dataset with the defined dimensions of input setup to work with seisnc standards.
    """
    # cmp not allowed with iline or xline
    if cmp is not None and (iline is not None or xline is not None):
        raise ValueError("cmp cannot be used with 3D dimensions (iline or xline)")

    # make sure dim args are sensible and to spec
    for key in dim_args.keys():
        try:
            if key[0] is not "d":
                raise ValueError
            n = int(key[1:])
        except (KeyError, IndexError, ValueError):
            ValueError(f"Unknown dimension name {key}, expected like d1, d2, ..., dN")

    # argument sanity checks
    if (cmp is not None or iline is not None) and "d1" in dim_args:
        raise ValueError("d1 cannot be defined with cmp or iline arguments")

    if xline is not None and "d2" in dim_args:
        raise ValueError("d2 cannot be defined with xline argument")

    if (twt is not None or depth is not None) and "d3" in dim_args:
        raise ValueError("d3 cannot be defined with vert arguments twt or depth")

    if offset is not None and "d4" in dim_args:
        raise ValueError("d4 cannot be defined with offset argument")

    if twt is not None and depth is not None:
        raise ValueError("Only one vertical domain specification allowed")

    dimensions = dict()
    # create dimension d1
    if cmp is not None:
        dimensions[CoordKeyField.cmp.value] = (["d1"], np.asarray(cmp))
    elif iline is not None:
        dimensions[CoordKeyField.iline.value] = (["d1"], np.asarray(iline))
    elif "d1" in dim_args and "d2" in dim_args:
        d1 = dim_args.pop("d1")
        dimensions[CoordKeyField.iline.value] = (["d1"], np.arange(1, d1 + 1))
    elif "d1" in dim_args and "d2" not in dim_args:
        d1 = dim_args.pop("d1")
        dimensions[CoordKeyField.cmp.value] = (["d1"], np.arange(1, d1 + 1))
    else:
        dimensions[CoordKeyField.iline.value] = (["d1"], np.r_[1])

    # create dimension d2
    if xline is not None:
        dimensions[CoordKeyField.xline.value] = (["d2"], np.asarray(xline))
    elif "d2" in dim_args:
        d2 = dim_args.pop("d2")
        dimensions[CoordKeyField.xline.value] = (["d2"], np.arange(1, d2 + 1))

    # create dimension d3
    if twt is not None:
        dimensions[CoordKeyField.twt.value] = (["d3"], np.asarray(twt))
    elif depth is not None:
        dimensions[CoordKeyField.depth.value] = (["d3"], np.asarray(depth))
    elif "d3" in dim_args:
        d3 = dim_args.pop("d3")
        dimensions["d3"] = (["d3"], np.arange(1, d3 + 1))
    else:
        dimensions["d3"] = (["d3"], np.r_[0])

    # create dimension d4
    if offset is not None:
        dimensions[CoordKeyField.offset.value] = (["d4"], np.asarray(offset))
    elif "d4" in dim_args:
        d4 = dim_args.pop("d4")
        dimensions["d4"] = (["d4"], np.arange(1, d4 + 1))

    # any left over dims
    for arg, val in dim_args.items():
        if isinstance(val, int):
            dimensions[arg] = ([arg], np.arange(1, val + 1))
        else:
            dimensions[arg] = ([arg], val)

    ds = xr.Dataset(coords=dimensions)
    ds.attrs.update({name: None for name in AttrKeyField._member_names_})
    ds.attrs[AttrKeyField.ns.value] = ds.dims["d3"]

    return ds


def _check_vert_units(units):
    # check units
    if units is not None:
        try:
            units = VerticalUnits[units].value
        except AttributeError:
            raise ValueError(
                f"The vert_units is unknown, got {units}, expected one of {VerticalUnits._member_names_}"
            )
    return units


def _dataset_coordinate_helper(vert, vert_domain, **kwargs):
    if vert_domain == "TWT":
        kwargs["twt"] = vert
        domain = VerticalKeyField.twt.value
    elif vert_domain == "DEPTH":
        kwargs["depth"] = vert
        domain = VerticalKeyField.depth.value
    else:
        raise ValueError(
            f"Unknown vert_domain {vert_domain}, expected one of TWT or DEPTH"
        )
    return kwargs, domain


def create3d_dataset(
    dims,
    first_sample=0,
    sample_rate=1,
    first_iline=1,
    iline_step=1,
    first_xline=1,
    xline_step=1,
    first_offset=None,
    offset_step=None,
    vert_domain="TWT",
    vert_units=None,
):
    """Create a regular 3D seismic dataset from basic grid geometry with optional
    offset dimension for pre-stack data.

    Args:
        dims (tuple of int): The dimensions of the dataset to create (iline, xline, vertical).
            If first_offset is specified then (iline, xline, vertical, offset)
        first_sample (int, optional): The first vertical sample. Defaults to 0.
        sample_rate (int, optional): The vertical sample rate. Defaults to 1.
        first_iline (int, optional): First inline number. Defaults to 1.
        iline_step (int, optional): Inline increment. Defaults to 1.
        first_xline (int, optional): First crossline number. Defaults to 1.
        xline_step (int, optional): Crossline increment. Defaults to 1.
        first_offset (int/float, optional): If not none, the offset dimension will be added starting
            at first offset. Defaults to None.
        offset_step (int, float, optional): Required if first_offset is specified. The offset increment.
        vert_domain (str, optional): Vertical domain, one of ('DEPTH', 'TWT'). Defaults to 'TWT'.
        vert_units(str, optional): Measurement system of of vertical coordinates.
            One of ('ms', 's', 'm', 'km', 'ft'): Defaults to None for unknown.
    """
    if first_offset is None:
        ni, nx, ns = dims
    else:
        ni, nx, ns, no = dims

    units = _check_vert_units(vert_units)

    vert = np.arange(
        first_sample, first_sample + sample_rate * ns, sample_rate, dtype=int
    )
    ilines = np.arange(
        first_iline, first_iline + iline_step * ni, iline_step, dtype=int
    )
    xlines = np.arange(
        first_xline, first_xline + xline_step * nx, xline_step, dtype=int
    )

    if first_offset is None:
        offset = None
    else:
        offset = np.arange(
            first_offset, first_offset + offset_step * no, offset_step, dtype=float
        )

    builder, domain = _dataset_coordinate_helper(
        vert, vert_domain, iline=ilines, xline=xlines, offset=offset
    )

    ds = create_seismic_dataset(**builder)
    ds.attrs[AttrKeyField.d3_units.value] = units
    ds.attrs[AttrKeyField.d3_domain.value] = domain
    ds.attrs[AttrKeyField.ds.value] = sample_rate
    ds.attrs[AttrKeyField.text.value] = "SEGY-SAK Create 3D Dataset"
    ds.attrs[AttrKeyField.corner_points.value] = [
        (ilines[0], xlines[0]),
        (ilines[-1], xlines[0]),
        (ilines[-1], xlines[-1]),
        (ilines[0], xlines[-1]),
        (ilines[0], xlines[0]),
    ]

    return ds


def create2d_dataset(
    dims,
    first_sample=0,
    sample_rate=1,
    first_cmp=1,
    cmp_step=1,
    first_offset=None,
    offset_step=None,
    vert_domain="TWT",
    vert_units=None,
):
    """Create a regular 2D seismic dataset from basic geometry.

    Args:
        dims (tuple of int): The dimensions of the dataset to create (ncdp, vertical).
            If first_offset is specified then (iline, xline, vertical, offset)
        first_sample (int, optional): The first vertical sample. Defaults to 0.
        sample_rate (int, optional): The vertical sample rate. Defaults to 1.
        first_cmp (int, optional): First cmp number. Defaults to 1.
        cmp_step (int, optional): CMP increment. Defaults to 1.
        first_offset (int/float, optional): If not none, the offset dimension will be added starting
            at first offset. Defaults to None.
        offset_step (int, float, optional): Required if first_offset is specified. The offset increment.
        vert_domain (str, optional): Vertical domain, one of ('DEPTH', 'TWT'). Defaults to 'TWT'.
        vert_units(str, optional): Measurement system of of vertical coordinates.
            One of ('ms', 's', 'm', 'km', 'ft'): Defaults to None for unknown.
    """
    if first_offset is None:
        ncmp, ns = dims
    else:
        ncmp, ns, no = dims

    # check units
    units = _check_vert_units(vert_units)

    vert = np.arange(
        first_sample, first_sample + sample_rate * ns, sample_rate, dtype=int
    )
    cmps = np.arange(first_cmp, first_cmp + cmp_step * ncmp, cmp_step, dtype=int)

    if first_offset is None:
        offset = None
    else:
        offset = np.arange(
            first_offset, first_offset + offset_step * no, offset_step, dtype=float
        )

    builder, domain = _dataset_coordinate_helper(
        vert, vert_domain, cmp=cmps, offset=offset
    )

    ds = create_seismic_dataset(**builder)
    ds.attrs[AttrKeyField.d3_units.value] = units
    ds.attrs[AttrKeyField.d3_domain.value] = domain
    ds.attrs[AttrKeyField.ds.value] = sample_rate
    ds.attrs[AttrKeyField.text.value] = "SEGY-SAK Create 2D Dataset"

    return ds
