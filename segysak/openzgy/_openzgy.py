"""Open ZGY Wrapper Module for Xarray"""

try:
    from openzgy.api import ZgyReader, ZgyWriter
except ImportError:
    raise ImportError(
        "The OpenZgy Module requires that Open ZGY for Python"
        " be installed. Checkout the module ReadMe for more information."
    )

import numpy as np
from .._seismic_dataset import create3d_dataset
from .._keyfield import CoordKeyField, VariableKeyField, AttrKeyField, VerticalKeyField


def zgy_loader(filepath):
    """Load a ZGY File.

    Args:
        filepath (pathlib.Path/str): The path or file location of the file to load.

    Returns:
        xarray.Dataset: Loaded data.
    """
    reader = ZgyReader(str(filepath))
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
    ds.attrs[AttrKeyField.source_file] = str(filepath)
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
