# pylint: disable=invalid-name
"""
All functions for handling and manipulating SEISNC files. This also include the
basic functions for creating a valid blank seisnc spec Dataset and file.

Author:
    Tony Hallam 2020
"""

import warnings

warnings.warn(
    "the seisnc module is depreciated for seismic_dataset", DeprecationWarning
)

import netCDF4

import numpy as np
import xarray as xr

SEINC_DIMS = ["i", "j", "k"]
SEISNC_COORDS = ["iline", "xline", "z"]
SEISNC_NONDATA_VAR = ["CDP_X", "CDP_Y", "CDP_TRACE"]
SEISNC_ATTR = [
    "ns",
    "sample_rate",
    "text",
]

DS_VALID_COORDS = ["iline", "xline", "depth", "twt", "cdp_x", "cdp_y", "lat", "lon"]

DS_DEFAULT_ATTR = dict(
    ns=None,
    sample_rate=None,
    text=None,
    measurement_system=None,
    epsg=None,
    corner_points=None,
    corner_points_xy=None,
    source_file=None,
    srd=None,
    datatype=None,
)


def create_empty_seisnc(ncfile, dims):
    """Creates an on disk blank seisnc (NetCDF) file.

    Args:
        ncfile: The output file path.
        dims (tuple): The (inline, xline, vert) dimensions of the seismic cube.
    """
    # calculate vert, inline and crossline ranges/meshgrids
    ni, nx, ns = dims

    ilines = np.arange(ni, dtype=int)
    xlines = np.arange(nx, dtype=int)
    vert = np.arange(ns, dtype=int)

    # use i j k here to match REM and because xarray orders dimensions at this
    # step based upon the alphanumeric name
    seisnc = xr.Dataset(
        coords={"iind": (["i"], ilines), "jind": (["j"], xlines), "kind": (["k"], vert)}
    )
    seisnc.to_netcdf(path=ncfile, mode="w")


def set_seisnc_dims(
    ncfile,
    first_sample=0,
    sample_rate=1,
    first_iline=1,
    iline_step=1,
    first_xline=1,
    xline_step=1,
    vert_domain="TWT",
    measurement_system=0,
):
    """

    Args:
        ncfile ([type]): [description]
        first_sample (int, optional): [description]. Defaults to 0.
        sample_rate (int, optional): [description]. Defaults to 1.
        first_iline (int, optional): [description]. Defaults to 0.
        iline_step (int, optional): [description]. Defaults to 1.
        first_xline (int, optional): [description]. Defaults to 0.
        xline_step (int, optional): [description]. Defaults to 1.
        vert_domain (str, optional): [description]. Defaults to 'TWT'.
        measurement_system(int/str, optional): Measurement system of
            coordinates, one of ['m', 'ft' 0]: Defaults to 0 for unknown.
    """

    if vert_domain == "TWT":
        units = "ms"
    elif vert_domain == "Z":
        units = "m"

    with xr.open_dataset(ncfile) as seisnc:
        dims = seisnc.dims

    # create dimension arrays
    ns = dims["k"]
    ni = dims["i"]
    nx = dims["j"]
    vert = np.arange(
        first_sample, first_sample + sample_rate * ns, sample_rate, dtype=int
    )
    ilines = np.arange(
        first_iline, first_iline + iline_step * ni, iline_step, dtype=int
    )
    xlines = np.arange(
        first_xline, first_xline + xline_step * nx, xline_step, dtype=int
    )

    # update seisnc on disk
    with netCDF4.Dataset(str(ncfile), mode="a") as seisnc:
        seisnc.createVariable("CDP_X", "f4", ("i", "j",))
        seisnc.createVariable("CDP_Y", "f4", ("i", "j",))
        seisnc.createVariable("CDP_TRACE", "i4", ("i", "j",))
        seisnc.createVariable("data", "f4", ("i", "j", "k"))

        seisnc.createVariable("vert", "f4", ("k"))
        seisnc.createVariable("iline", "i4", ("i"))
        seisnc.createVariable("xline", "i4", ("j"))

        seisnc["data"].units = units
        seisnc["iline"][:] = ilines
        seisnc["xline"][:] = xlines
        seisnc["vert"][:] = vert
        seisnc.ns = ns
        seisnc.sample_rate = sample_rate
        seisnc.measurement_system = measurement_system

        seisnc.text = "Blank seisnc"
