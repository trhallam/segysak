# pylint: disable=invalid-name,no-member
"""
Basic definitions and tools for the creation of seismic datasets. This is an
xarray.Dataset with particular dimensions, coordinates and attributes that inform
other tools and file format converters in segysak library.

"""

import numpy as np

from ._keyfield import (
    CoordKeyField,
    VerticalUnits,
    AttrKeyField,
    VerticalKeyField,
    DimensionKeyField,
)
from ._accessor import xr


def _check_input(var):
    if var is None:
        return var
    if isinstance(var, int):
        return np.array([i for i in range(1, var + 1)], dtype=int)
    else:
        var = np.asarray(var)
        if var.ndim > 1:
            raise ValueError("Inputs for geometry must be one dimensional")
        return var


def create_seismic_dataset(
    twt=None, depth=None, cdp=None, iline=None, xline=None, offset=None, **dim_args
):
    """Create a blank seismic dataset by setting the dimension sizes (d#) or by passing
    arrays for known dimensions.

    iline and xline must be specified together and are mutually exclusive to cdp argument.

    Args:
        twt (int/array-like, optional): Two-way time vertical sampling coordinates.
            Cannot be used with depth argument. Defaults to None.
        depth (int/array-like, optional): Depth vertical sampling coordinates.
            Cannot be used with twt argument. Defaults to None.
        cdp (int/array-like, optional): The CDP numbering for 2D data, cannot be used
            with iline or xline. Use for 2D seismic data. Defaults to None.
        iline (int/array-like, optional): The iline numbering, cannot be
            used with cdp argument. Use for 3D seismic data. Defaults to None.
        xline (int/array-like, optional): The xline numbering, cannot be
            used with cdp argument. Use for 3D seismic data. Defaults to None.
        offset (int/array-like, optional): The offset. This will fill dimension d4.
            Use for pre-stack data. Defaults to None.
        dim_args (int/array-like): Other dimensions you would like in your dataset. The key will be the dimension name.

    Returns:
        xarray.Dataset: A dataset with the defined dimensions of input setup to work with seisnc standards.
    """
    # cdp not allowed with iline or xline
    if cdp is not None and (iline is not None or xline is not None):
        raise ValueError(
            "cdp argument cannot be used with 3D dimensions (iline or xline)"
        )

    if iline is not None and xline is None:
        raise ValueError("xline needed with iline argument, 3d geometry requires both")

    if xline is not None and iline is None:
        raise ValueError("xline needed with iline argument, 3d geometry requires both")

    # check inputs
    twt = _check_input(twt)
    depth = _check_input(depth)
    cdp = _check_input(cdp)
    iline = _check_input(iline)
    xline = _check_input(xline)
    offset = _check_input(offset)
    # # make sure other dim args are sensible and to spec
    for key in dim_args.keys():
        dim_args[key] = _check_input(dim_args[key])

    dimensions = dict()
    # create dimension d1
    if cdp is not None:
        dimensions[CoordKeyField.cdp] = ([CoordKeyField.cdp], cdp)
    elif iline is not None:  # 3d data
        dimensions[CoordKeyField.iline] = ([CoordKeyField.iline], iline)
        dimensions[CoordKeyField.xline] = ([CoordKeyField.xline], xline)

    # create dimension d3
    if twt is not None:
        dimensions[CoordKeyField.twt] = ([CoordKeyField.twt], twt)
    if depth is not None:
        dimensions[CoordKeyField.depth] = ([CoordKeyField.depth], depth)

    # create dimension d4
    if offset is not None:
        dimensions[CoordKeyField.offset] = (
            [CoordKeyField.offset],
            offset,
        )

    # any left over dims
    for arg, val in dim_args.items():
        dimensions[arg] = ([arg], val)

    ds = xr.Dataset(coords=dimensions)

    if twt is not None:
        ds.attrs[AttrKeyField.ns] = twt.size
    elif depth is not None:
        ds.attrs[AttrKeyField.ns] = depth.size

    ds.attrs.update({name: None for name in AttrKeyField})

    return ds


def _check_vert_units(units):
    # check units
    if units is not None:
        try:
            units = VerticalUnits[units]
        except (AttributeError, KeyError):
            raise ValueError(
                f"The vert_units is unknown, got {units}, expected one of {VerticalUnits.keys()}"
            )
    return units


def _dataset_coordinate_helper(vert, vert_domain, **kwargs):
    if vert_domain == "TWT":
        kwargs["twt"] = vert
        domain = VerticalKeyField.twt
    elif vert_domain == "DEPTH":
        kwargs["depth"] = vert
        domain = VerticalKeyField.depth
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
    vert_domain = vert_domain.upper()

    if first_offset is None:
        ni, nx, ns = dims
    else:
        ni, nx, ns, no = dims

    units = _check_vert_units(vert_units)

    # 1e-10 to stabilise range on rounding errors for weird floats from hypothesis

    vert = np.arange(
        first_sample,
        first_sample + sample_rate * ns - 1e-10,
        sample_rate,
        dtype=int,
    )
    ilines = np.arange(
        first_iline,
        first_iline + iline_step * ni - 1e-10,
        iline_step,
        dtype=int,
    )
    xlines = np.arange(
        first_xline,
        first_xline + xline_step * nx - 1e-10,
        xline_step,
        dtype=int,
    )

    if first_offset is None:
        offset = None
    else:
        offset = np.arange(
            first_offset,
            first_offset + offset_step * no - 1e-10,
            offset_step,
            dtype=float,
        )

    builder, domain = _dataset_coordinate_helper(
        vert, vert_domain, iline=ilines, xline=xlines, offset=offset
    )

    ds = create_seismic_dataset(**builder)
    ds.attrs[AttrKeyField.measurement_system] = units
    ds.attrs[AttrKeyField.d3_domain] = domain
    ds.attrs[AttrKeyField.sample_rate] = sample_rate
    ds.attrs[AttrKeyField.text] = "SEGY-SAK Create 3D Dataset"
    ds.attrs[AttrKeyField.corner_points] = [
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
    first_cdp=1,
    cdp_step=1,
    first_offset=None,
    offset_step=None,
    vert_domain="TWT",
    vert_units=None,
):
    """Create a regular 2D seismic dataset from basic geometry.

    Args:
        dims (tuple of int): The dimensions of the dataset to create (ncdp, vertical).
            If first_offset is specified then (ncdp, vertical, offset)
        first_sample (int, optional): The first vertical sample. Defaults to 0.
        sample_rate (int, optional): The vertical sample rate. Defaults to 1.
        first_cdp (int, optional): First CDP number. Defaults to 1.
        cdp_step (int, optional): CDP increment. Defaults to 1.
        first_offset (int/float, optional): If not none, the offset dimension will be added starting
            at first offset. Defaults to None.
        offset_step (int, float, optional): Required if first_offset is specified. The offset increment.
        vert_domain (str, optional): Vertical domain, one of ('DEPTH', 'TWT'). Defaults to 'TWT'.
        vert_units(str, optional): Measurement system of of vertical coordinates.
            One of ('ms', 's', 'm', 'km', 'ft'): Defaults to None for unknown.
    """
    vert_domain = vert_domain.upper()

    if first_offset is None:
        ncdp, ns = dims
    else:
        ncdp, ns, no = dims

    # check units
    units = _check_vert_units(vert_units)

    vert = np.arange(
        first_sample, first_sample + sample_rate * ns, sample_rate, dtype=int
    )
    cdps = np.arange(first_cdp, first_cdp + cdp_step * ncdp, cdp_step, dtype=int)

    if first_offset is None:
        offset = None
    else:
        offset = np.arange(
            first_offset, first_offset + offset_step * no, offset_step, dtype=float
        )

    builder, domain = _dataset_coordinate_helper(
        vert, vert_domain, cdp=cdps, offset=offset
    )

    ds = create_seismic_dataset(**builder)
    ds.attrs[AttrKeyField.measurement_system] = units
    ds.attrs[AttrKeyField.d3_domain] = domain
    ds.attrs[AttrKeyField.sample_rate] = sample_rate
    ds.attrs[AttrKeyField.text] = "SEGY-SAK Create 2D Dataset"

    return ds
