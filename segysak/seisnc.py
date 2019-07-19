# pylint: disable=invalid-name
"""
All functions for handling and manipulating SEISNC files.

Author:
    Tony Hallam 2019
"""
import netCDF4

import numpy as np
import xarray as xr

SEINC_DIMS = ['il', 'xl', 'v']
SEISNC_COORDS = ['iline', 'xline', 'z']
SEISNC_NONDATA_VAR = ['CDP_X', 'CDP_Y', 'CDP_TRACE']
SEISNC_ATTR = ['ns', 'ds', 'text', ]

def create_empty_seisnc(ncfile, dims):
    """Creates an on disk blank NetCDF File

    Args:
        ncfile: The output file path.
        dims (tuple): The (inline, xline, vert) dimensions of the seismic cube.
    """
    # calculate vert, inline and crossline ranges/meshgrids
    ni, nx, ns = dims

    ilines = np.arange(ni, dtype=int)
    xlines = np.arange(nx, dtype=int)
    vert = np.arange(ns, dtype=int)

    seisnc = xr.Dataset(
        coords={
            'iline': (['il'], ilines),
            'xline': (['xl'], xlines),
            'z':  (['v'], vert)
        }
    )

    seisnc.to_netcdf(path=ncfile, mode='w')

def set_seisnc_dims(ncfile, first_sample=0, sample_rate=1, first_iline=1,
                    iline_step=1, first_xline=1, xline_step=1, vert_domain='TWT',
                    measurement_system=0):
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

    if vert_domain == 'TWT':
        units = 'ms'
    elif vert_domain == 'Z':
        units = 'm'

    with xr.open_dataset(ncfile) as seisnc:
        dims = seisnc.dims

    # create dimension arrays
    ns = dims['v']
    ni = dims['il']
    nx = dims['xl']
    vert = np.arange(first_sample, first_sample + sample_rate*ns, sample_rate, dtype=int)
    ilines = np.arange(first_iline, first_iline + iline_step*ni, iline_step, dtype=int)
    xlines = np.arange(first_xline, first_xline + xline_step*nx, xline_step, dtype=int)

    # update seisnc on disk
    with netCDF4.Dataset(str(ncfile), mode="a") as seisnc:
        seisnc.createVariable("CDP_X", 'f4', ('il', 'xl', ))
        seisnc.createVariable("CDP_Y", 'f4', ('il', 'xl', ))
        seisnc.createVariable("CDP_TRACE", 'i4', ('il', 'xl', ))
        seisnc.createVariable("data", 'f4', ('il', 'xl', 'v'))

        seisnc['z'][:] = vert
        seisnc['iline'][:] = ilines
        seisnc['xline'][:] = xlines

        seisnc['data'].units = units
        seisnc.ns = ns
        seisnc.ds = sample_rate
        seisnc.measurement_system = measurement_system

        seisnc.text = "Blank seisnc"