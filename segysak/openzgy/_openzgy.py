"""Open ZGY Wrapper Module for Xarray"""

try:
    from openzgy.api import ZgyReader, ZgyWriter, SampleDataType, UnitDimension
except ImportError:
    raise ImportError(
        "The OpenZgy Module requires that Open ZGY for Python"
        " be installed. Checkout the module ReadMe for more information."
    )

import numpy as np
from .._seismic_dataset import create3d_dataset
from .._keyfield import CoordKeyField, VariableKeyField, AttrKeyField, VerticalKeyField
from ..segy._segy_writer import _check_dimension


def zgy_loader(filename):
    """Load a ZGY File.

    Args:
        filename (pathlib.Path/str): The path or file location of the file to load.

    Returns:
        xarray.Dataset: Loaded data.
    """
    reader = ZgyReader(str(filename))
    header = reader.meta

    data_shape = header["size"]
    zstart = header["zstart"]
    zinc = header["zinc"]
    zunit_fact = header["zunitfactor"]
    ilstep, xlstep = header["annotinc"]
    ilstart, xlstart = header["annotstart"]

    corners = header["corners"]

    ds = create3d_dataset(
        data_shape,
        first_sample=zstart,
        sample_rate=zinc,
        iline_step=ilstep,
        xline_step=xlstep,
        first_iline=ilstart,
        first_xline=xlstart,
    )

    # load attributes
    ds.attrs[AttrKeyField.corner_points_xy] = [c for c in corners]
    ds.attrs[AttrKeyField.source_file] = str(filename)
    ds.attrs[AttrKeyField.text] = "SEGY-SAK Created from ZGY Loader"
    ds.attrs[AttrKeyField.ns] = data_shape[2]
    ds.attrs[AttrKeyField.coord_scalar] = header["hunitfactor"]

    # load data
    data = np.zeros(data_shape, dtype=np.float32)
    reader.read((0, 0, 0), data)
    ds[VariableKeyField.data] = (
        (CoordKeyField.iline, CoordKeyField.xline, "twt"),
        data,
    )

    # create XY world coordinates.
    ds[CoordKeyField.cdp_x] = (
        (CoordKeyField.iline, CoordKeyField.xline),
        np.zeros(data_shape[0:2]),
    )
    ds[CoordKeyField.cdp_y] = ds[CoordKeyField.cdp_x].copy()

    pos = np.dstack(
        [
            ds.iline.broadcast_like(ds[CoordKeyField.cdp_x]).values,
            ds.xline.broadcast_like(ds[CoordKeyField.cdp_x]).values,
        ]
    )

    pos = np.apply_along_axis(reader.annotToWorld, -1, pos)

    ds[CoordKeyField.cdp_x] = ((CoordKeyField.iline, CoordKeyField.xline), pos[:, :, 0])
    ds[CoordKeyField.cdp_y] = ((CoordKeyField.iline, CoordKeyField.xline), pos[:, :, 1])

    return ds


def zgy_writer(seisnc_dataset, filename, dimension=None):
    """Write a seisnc dataset to ZGY file format.

    Args:
        seisnc_dataset (xr.Dataset): This should be a seisnc dataset.
        filename (pathlib.Path/str): The filename to write to.
        dimension (str, optional): The dimension to write out, if
            None uses available dimenions twt first.
    """
    assert seisnc_dataset.seis.is_3d()

    dimension = _check_dimension(seisnc_dataset, dimension)
    if dimension is "twt":
        zunitdim = UnitDimension(2001)
        vdim = "TWT"
    elif dimension is "depth":
        zunitdim = UnitDimension(2002)
    else:
        zunitdim = UnitDimension(2000)
    try:
        coord_scalar_mult = seisnc_dataset.attrs["coord_scalar_mult"]
    except KeyError:
        coord_scalar_mult = 1.0

    # dimensions
    ni, nj, nk = (
        seisnc_dataset.dims[CoordKeyField.iline],
        seisnc_dataset.dims[CoordKeyField.xline],
        seisnc_dataset.dims[dimension],
    )
    # vertical
    z0 = int(seisnc_dataset[dimension].values[0])
    dz = int(seisnc_dataset.sample_rate)

    # annotation
    il0 = seisnc_dataset[CoordKeyField.iline].values[0]
    xl0 = seisnc_dataset[CoordKeyField.xline].values[0]
    dil = int(np.median(np.diff(seisnc_dataset[CoordKeyField.iline].values)))
    dxl = int(np.median(np.diff(seisnc_dataset[CoordKeyField.xline].values)))

    dtype_float = SampleDataType(1003)

    # TODO: Add corner coordinates if available.

    writer = ZgyWriter(
        str(filename),
        size=(ni, nj, nk),
        datatype=dtype_float,
        zunitdim=zunitdim,
        zstart=z0,
        zinc=dz,
        annotstart=(il0, xl0),
        annotinc=(dil, dxl),
    )

    order = (CoordKeyField.iline, CoordKeyField.xline, dimension)

    writer.write(
        (0, 0, 0),
        seisnc_dataset[VariableKeyField.data]
        .transpose(*order)
        .values.astype(np.float32),
    )


# TODO: Create a converter which does just a few ilines at a time. For large files.
