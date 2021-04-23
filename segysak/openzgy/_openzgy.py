"""Open ZGY Wrapper Module for Xarray"""

# look for local installation
import sys
import importlib
import pathlib

_missing_local = True
_missing_submodule = True

try:
    import openzgy

    _missing_local = False
except ImportError:
    pass

if _missing_local:
    _here = pathlib.Path(__file__).absolute().parent
    _finder = importlib.machinery.FileFinder(str(_here / "open-zgy/python"))
    _spec = _finder.find_spec("openzgy")
    if _spec is not None:
        _module = importlib.util.module_from_spec(_spec)
        sys.modules["openzgy"] = _module
        _spec.loader.exec_module(_module)
        _missing_submodule = False

if _missing_local and _missing_submodule:
    raise ImportError(
        "The OpenZgy Module requires that Open ZGY for Python"
        " be installed. Checkout the module ReadMe for more information."
    )

from openzgy.api import ZgyReader, ZgyWriter, SampleDataType, UnitDimension

import numpy as np
from .._seismic_dataset import create3d_dataset
from .._keyfield import CoordKeyField, VariableKeyField, AttrKeyField
from ..segy._segy_writer import _check_dimension


def zgy_loader(filename):
    """Load a ZGY File.

    Args:
        filename (pathlib.Path/str): The path or file location of the file to load.

    Returns:
        xarray.Dataset: Loaded data.
    """
    with ZgyReader(str(filename)) as reader:
        header = reader.meta

        data_shape = header["size"]
        zstart = header["zstart"]
        zinc = header["zinc"]
        # zunit_fact = header["zunitfactor"] may need this later
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

        ds[CoordKeyField.cdp_x] = (
            (CoordKeyField.iline, CoordKeyField.xline),
            pos[:, :, 0],
        )
        ds[CoordKeyField.cdp_y] = (
            (CoordKeyField.iline, CoordKeyField.xline),
            pos[:, :, 1],
        )

        ds.seis.calc_corner_points()

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

    seisnc_dataset.seis.calc_corner_points()
    corners = tuple(seisnc_dataset.attrs["corner_points_xy"][i] for i in [0, 3, 1, 2])

    dimension = _check_dimension(seisnc_dataset, dimension)
    if dimension == "twt":
        zunitdim = UnitDimension(2001)
    elif dimension == "depth":
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

    dtype_float = SampleDataType["float"]

    with ZgyWriter(
        str(filename),
        size=(ni, nj, nk),
        datatype=dtype_float,
        zunitdim=zunitdim,
        zstart=z0,
        zinc=dz,
        annotstart=(il0, xl0),
        annotinc=(dil, dxl),
        corners=corners,
    ) as writer:

        order = (CoordKeyField.iline, CoordKeyField.xline, dimension)

        writer.write(
            (0, 0, 0),
            seisnc_dataset[VariableKeyField.data]
            .transpose(*order)
            .values.astype(np.float32),
        )


# TODO: Create a converter which does just a few ilines at a time. For large files.
