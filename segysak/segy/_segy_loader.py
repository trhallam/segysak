# pylint: disable=invalid-name, unused-variable
"""Functions to interact with segy data
"""

from warnings import warn
from functools import partial
import importlib

import segyio

# import netCDF
import h5netcdf
import numpy as np
import pandas as pd
import xarray as xr

try:
    has_ipywidgets = importlib.find_loader("ipywidgets") is not None
    if has_ipywidgets:
        from tqdm.autonotebook import tqdm
    else:
        from tqdm import tqdm as tqdm
except ModuleNotFoundError:
    from tqdm import tqdm as tqdm

TQDM_ARGS = dict(unit_scale=True, unit=" traces")

from segysak._keyfield import (
    CoordKeyField,
    AttrKeyField,
    VariableKeyField,
    DimensionKeyField,
)
from segysak._seismic_dataset import (
    create_seismic_dataset,
    create3d_dataset,
    _dataset_coordinate_helper,
)

# from segysak.seisnc import create_empty_seisnc, set_seisnc_dims
from segysak.tools import check_crop, check_zcrop
from segysak._accessor import open_seisnc

from ._segy_headers import segy_bin_scrape, segy_header_scrape
from ._segy_text import get_segy_texthead
from ._segy_globals import _SEGY_MEASUREMENT_SYSTEM


class SegyLoadError(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


def _get_tf(var):
    return str(segyio.TraceField(var))


def _header_to_index_mapping(series):
    as_unique = series.unique()
    sorted_unique = np.sort(as_unique)
    mapper = {val: i for i, val in enumerate(sorted_unique)}
    return series.replace(mapper)


def _segy3d_ncdf(
    segyfile,
    ncfile,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    il_head_loc,
    xl_head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 3d data into a netcdf4 file. This function should
    stream data to disk avoiding excessive memory usage.
    """

    if vert_domain == "TWT":
        dims = DimensionKeyField.threed_twt.value
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.threed_depth.value
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf, h5netcdf.File(
        ncfile, "a"
    ) as seisnc:

        segyf.mmap()

        # create data variable
        seisnc_data = seisnc.create_variable(VariableKeyField.data.value, dims, float)

        seisnc.flush()

        # work out fast and slow dir
        if head_df[xl_head_loc].diff().min() < 0:
            contig_dir = il_head_loc
            broken_dir = xl_head_loc
            slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = xl_head_loc
            broken_dir = il_head_loc
            slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                seisnc_data[val.il_index, val.xl_index, :] = segyf.trace[trc][
                    n0 : ns + 1
                ]
                pb.update()
        pb.close()

    return open_seisnc(ncfile)


def _segy3dps_ncdf(
    segyfile,
    ncfile,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    il_head_loc,
    xl_head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 3d pre-stack data into a netcdf4 file. This function should
    stream data to disk avoiding excessive memory usage.
    """

    if vert_domain == "TWT":
        dims = DimensionKeyField.threed_ps_twt.value
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.threed_ps_depth.value
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf, h5netcdf.File(
        ncfile, "a"
    ) as seisnc:

        segyf.mmap()

        # create data variable
        seisnc_data = seisnc.create_variable(VariableKeyField.data.value, dims, float)

        seisnc.flush()

        # work out fast and slow dir
        if head_df[xl_head_loc].diff().min() < 0:
            contig_dir = il_head_loc
            broken_dir = xl_head_loc
            slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = xl_head_loc
            broken_dir = il_head_loc
            slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                seisnc_data[val.il_index, val.xl_index, :, val.off_index] = segyf.trace[
                    trc
                ][n0 : ns + 1]
                pb.update()
        pb.close()

    return open_seisnc(ncfile)


def _segy3d_xr(
    segyfile,
    ds,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    il_head_loc,
    xl_head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 3d data into an xarray with the seisnc form.
    """

    if vert_domain == "TWT":
        dims = DimensionKeyField.threed_twt.value
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.threed_depth.value
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()

        # work out fast and slow dir
        if head_df[xl_head_loc].diff().min() < 0:
            contig_dir = il_head_loc
            broken_dir = xl_head_loc
            slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = xl_head_loc
            broken_dir = il_head_loc
            slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )
        shape = [ds.dims[d] for d in dims]
        volume = np.zeros(shape)

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                volume[int(val.il_index), int(val.xl_index), :] = segyf.trace[trc][
                    n0 : ns + 1
                ]
                pb.update()
        pb.close()

    ds[VariableKeyField.data.value] = (dims, volume)

    return ds


def _segy3dps_xr(
    segyfile,
    ds,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    il_head_loc,
    xl_head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 3d pre-stack data into an xarray with the seisnc form.
    """

    if vert_domain == "TWT":
        dims = DimensionKeyField.threed_ps_twt.value
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.threed_ps_depth.value
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()

        # work out fast and slow dir
        if head_df[xl_head_loc].diff().min() < 0:
            contig_dir = il_head_loc
            broken_dir = xl_head_loc
            slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = xl_head_loc
            broken_dir = il_head_loc
            slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )
        shape = [ds.dims[d] for d in dims]
        volume = np.zeros(shape)

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                volume[
                    int(val.il_index), int(val.xl_index), :, int(val.off_index)
                ] = segyf.trace[trc][n0 : ns + 1]
                pb.update()
        pb.close()

    ds[VariableKeyField.data.value] = (dims, volume)

    return ds


def _3dsegy_loader(
    segyfile,
    head_df,
    head_bin,
    ncfile=None,
    iline=189,
    xline=193,
    offset=None,
    vert_domain="TWT",
    data_type="AMP",
    crop=None,
    zcrop=None,
    silent=False,
    return_geometry=False,
    **segyio_kwargs,
):
    """Convert SEGY data to Xarray or Netcdf4

    This is a helper function for segy_loader. Users should use that function
    directly to load all segy data.

    """

    # get names of columns where stuff we want is
    il_head_loc = _get_tf(iline)
    xl_head_loc = _get_tf(xline)

    # get vertical sample ranges
    n0 = 0
    nsamp = head_bin["Samples"]
    ns0 = head_df.DelayRecordingTime.min()

    # short way to get inlines/xlines
    ilines = head_df[il_head_loc].unique()
    xlines = head_df[xl_head_loc].unique()
    inlines = np.sort(ilines)
    xlines = np.sort(xlines)
    iline_index_map = {il: i for i, il in enumerate(ilines)}
    xline_index_map = {xl: i for i, xl in enumerate(xlines)}
    head_df.loc[:, "il_index"] = head_df[il_head_loc].replace(iline_index_map)
    head_df.loc[:, "xl_index"] = head_df[xl_head_loc].replace(xline_index_map)

    # binary header translation
    ns = head_bin["Samples"]
    ds = head_bin["Interval"] / 1000.0
    msys = _SEGY_MEASUREMENT_SYSTEM[head_bin["MeasurementSystem"]]

    # for offset
    if offset is not None:
        off_head_loc = _get_tf(offset)
        offsets = head_df[off_head_loc].unique()
        offsets = np.sort(offsets)
        offset_index_map = {off: i for i, off in enumerate(offsets)}
        head_df["off_index"] = head_df[off_head_loc].replace(offset_index_map)
    else:
        offsets = None

    if zcrop is not None:
        zcrop = check_zcrop(zcrop, [0, ns])
        n0, ns = zcrop
        ns0 = ds * n0
        nsamp = ns - n0 + 1
    vert_samples = np.arange(ns0, ns0 + ds * nsamp, ds, dtype=int)

    builder, domain = _dataset_coordinate_helper(
        vert_samples, vert_domain, iline=ilines, xline=xlines, offset=offsets
    )

    ds = create_seismic_dataset(**builder)

    # create_seismic_dataset(d1=ni, d2=nx, d3=nsamp)
    text = get_segy_texthead(segyfile, **segyio_kwargs)
    ds.attrs[AttrKeyField.text.value] = text

    if ncfile is not None and return_geometry == False:
        ds.seisio.to_netcdf(ncfile)
    elif return_geometry:
        # return geometry -> e.g. don't process segy traces
        return ds
    else:
        ncfile = ds

    segyio_kwargs.update(dict(ignore_geometry=True, iline=iline, xline=xline))

    # not prestack data
    if offset is None and not isinstance(ncfile, xr.Dataset):
        ds = _segy3d_ncdf(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            il_head_loc,
            xl_head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    # not prestack data load into memory
    if offset is None and isinstance(ncfile, xr.Dataset):
        ds = _segy3d_xr(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            il_head_loc,
            xl_head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    # prestack data
    if offset is not None and not isinstance(ncfile, xr.Dataset):
        ds = _segy3dps_ncdf(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            il_head_loc,
            xl_head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    # prestack data load into memory
    if offset is not None and isinstance(ncfile, xr.Dataset):
        ds = _segy3dps_xr(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            il_head_loc,
            xl_head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    return ds


def _segy2d_xr(
    segyfile,
    ds,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    cdp_head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 2d data into an xarray with the seisnc form.
    """

    if vert_domain == "TWT":
        dims = DimensionKeyField.twod_twt.value
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.twod_ps_depth.value
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )
        shape = [ds.dims[d] for d in dims]
        volume = np.zeros(shape)

        # this can probably be done as a block - leaving for now just incase sorting becomes an issue
        for trc, val in head_df.iterrows():
            volume[int(val.cdp_index), :] = segyf.trace[trc][n0 : ns + 1]
            pb.update()
        pb.close()

    ds[VariableKeyField.data.value] = (
        dims,
        volume[:, n0 : ns + 1],
    )

    return ds


def _segy2d_ps_xr(
    segyfile,
    ds,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    cdp_head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 2d data into an xarray with the seisnc form.
    """

    if vert_domain == "TWT":
        dims = DimensionKeyField.twod_twt.value
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.twod_ps_depth.value
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )
        shape = [ds.dims[d] for d in dims]
        volume = np.zeros(shape)

        # this can probably be done as a block - leaving for now just incase sorting becomes an issue
        for trc, val in head_df.iterrows():
            volume[int(val.cdp_index), :, int(val.off_index)] = segyf.trace[trc][
                n0 : ns + 1
            ]
            pb.update()
        pb.close()

    ds[VariableKeyField.data.value] = (
        dims,
        volume,
    )

    return ds


def _2dsegy_loader(
    segyfile,
    head_df,
    head_bin,
    ncfile=None,
    cdp=None,
    offset=None,
    vert_domain="TWT",
    data_type="AMP",
    crop=None,
    zcrop=None,
    silent=False,
    return_geometry=False,
    **segyio_kwargs,
):
    """Convert SEGY data to Xarray or Netcdf4

    This is a helper function for segy_loader. Users should use that function
    directly to load all segy data.

    """

    # get names of columns where stuff we want is
    if cdp is None:
        cdp = 21
        cdp_head_loc = _get_tf(cdp)
        head_df[cdp_head_loc] = head_df.index.values
    else:
        cdp_head_loc = _get_tf(cdp)

    # get vertical sample ranges
    n0 = 0
    nsamp = head_bin["Samples"]
    ns0 = head_df.DelayRecordingTime.min()

    # short way to get cdps
    cdps = head_df[cdp_head_loc].unique()
    cdps = np.sort(cdps)
    head_df["cdp_index"] = _header_to_index_mapping(head_df[cdp_head_loc])

    # binary header translation
    nsamp = head_bin["Samples"]
    sample_rate = head_bin["Interval"] / 1000.0
    msys = _SEGY_MEASUREMENT_SYSTEM[head_bin["MeasurementSystem"]]

    # for offset
    if offset is not None:
        off_head_loc = _get_tf(offset)
        offsets = head_df[off_head_loc].unique()
        offsets = np.sort(offsets)
        head_df["off_index"] = _header_to_index_mapping(head_df[off_head_loc])
    else:
        offsets = None

    if zcrop is not None:
        zcrop = check_zcrop(zcrop, [0, nsamp])
        n0, nsamp = zcrop
        ns0 = sample_rate * n0
        nsamp = nsamp - n0 + 1
    vert_samples = np.arange(ns0, ns0 + sample_rate * nsamp, sample_rate, dtype=int)

    builder, domain = _dataset_coordinate_helper(
        vert_samples, vert_domain, cdp=cdps, offset=offsets
    )

    ds = create_seismic_dataset(**builder)

    # create_seismic_dataset(d1=ni, d2=nx, d3=nsamp)
    text = get_segy_texthead(segyfile, **segyio_kwargs)
    ds.attrs[AttrKeyField.text.value] = text

    if ncfile is not None and return_geometry == True:
        ds.seisio.to_netcdf(ncfile)
        return ds
    elif return_geometry:
        return ds

    segyio_kwargs.update(dict(ignore_geometry=True))

    # stacked data
    if offset is None:
        ds = _segy2d_xr(
            segyfile,
            ds,
            segyio_kwargs,
            n0,
            nsamp,
            head_df,
            cdp_head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    # # prestack data
    if offset is not None:
        ds = _segy2d_ps_xr(
            segyfile,
            ds,
            segyio_kwargs,
            n0,
            nsamp,
            head_df,
            cdp_head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    if ncfile is not None:
        ds.seisio.to_netcdf(ncfile)

    return ds


def well_known_byte_locs(name):
    """Convert SEGY data to NetCDF4 File

        Returns a dict containing the byte locations for well known SEGY variants in the wild.

        Caller should provide the name
        e.g.
            - standard_3d
            - petrel_3d
            - ...

        The intention is that this can be used with segy_loader to easily setup the load:

            seismic = segy_loader(filepath, **well_known_byte_locs('petrel_3d'))

    """
    if name == "standard_3d":
        # todo maybe this should be a rev1.2 or something?
        return dict(iline=181, xline=185, cdpx=189, cdpy=193)
    elif name == "petrel_3d":
        return dict(iline=5, xline=21, cdpx=73, cdpy=77)
    else:
        raise ValueError(f"No byte locatons for {name}")


def segy_loader(
    segyfile,
    ncfile=None,
    cdp=None,
    iline=None,
    xline=None,
    cdpx=None,
    cdpy=None,
    offset=None,
    vert_domain="TWT",
    data_type="AMP",
    ix_crop=None,
    cdp_crop=None,
    xy_crop=None,
    z_crop=None,
    return_geometry=False,
    silent=False,
    extra_byte_fields=None,
    **segyio_kwargs,
):
    """Convert SEGY data to NetCDF4 File

    The output ncfile has the following structure
        Dimensions:
            d1 - CDP or Inline axis
            d2 - Xline axis
            d3 - The vertical axis
            d4 - Offset/Angle Axis
        Coordinates:
            iline - The inline numbering
            xline - The xline numbering
            cdp_x - Eastings
            cdp_y - Northings
            cdp - Trace Number for 2d
        Variables
            data - The data volume
        Attributes:
            TBC

    Args:
        segyfile (str): Input segy file path
        ncfile (str, optional): Output SEISNC file path. If none the loaded data will be
            returned in memory as an xarray.Dataset.
        iline (int, optional): Inline byte location, usually 189
        xline (int, optional): Cross-line byte location, usally 193
        vert (str, optional): Vertical sampling domain. One of ['TWT', 'DEPTH']. Defaults to 'TWT'.
        cdp (int, optional): The CDP byte location, usually 21.
        data_type (str, optional): Data type ['AMP', 'VEL']. Defaults to 'AMP'.
        cdp_crop (list, optional): List of minimum and maximum cmp values to output.
            Has the form '[min_cmp, max_cmp]'. Ignored for 3D data.
        ix_crop (list, optional): List of minimum and maximum inline and crossline to output.
            Has the form '[min_il, max_il, min_xl, max_xl]'. Ignored for 2D data.
        xy_crop (list, optional): List of minimum and maximum cdp_x and cdp_y to output.
            Has the form '[min_x, max_x, min_y, max_y]'. Ignored for 2D data.
        z_crop (list, optional): List of minimum and maximum vertical samples to output.
            Has the form '[min, max]'.
        return_geometry (bool, optional): If true returns an xarray.dataset which doesn't contain data but mirrors
            the input volume header information.
        extra_byte_fields (list/mapping): A list of int or mapping of byte fields that should be returned as variables in the dataset.
        silent (bool): Disable progress bar.
        **segyio_kwargs: Extra keyword arguments for segyio.open

    Returns:
        xarray.Dataset: If ncfile keyword is specified returns open handle to disk netcdf4,
            otherwise the data in memory. If return_geometry is True does not load trace data and
            returns headers in geometry.
    """
    # Input sanity checks
    if cdp is not None and (iline is not None or xline is not None):
        raise ValueError("cdp cannot be defined with iline and xiline")

    if iline is None and xline is not None:
        raise ValueError("iline must be defined with xline")

    if xline is None and iline is not None:
        raise ValueError("xline must be defined with iline")

    if isinstance(extra_byte_fields, list):
        extra_byte_fields = {
            _get_tf(field): _get_tf(field) for field in extra_byte_fields
        }
    elif isinstance(extra_byte_fields, dict):
        extra_byte_fields = {
            key: _get_tf(field) for key, field in extra_byte_fields.items()
        }
    elif extra_byte_fields is None:
        extra_byte_fields = dict()
    else:
        raise ValueError("Unknown type for extra_byte_fields")

    if cdpx is None:
        cdpx = 181  # Assume standard location if misisng
    x_head_loc = _get_tf(cdpx)
    if cdpy is None:
        cdpy = 185  # Assume standard location if missing
    y_head_loc = _get_tf(cdpy)

    extra_byte_fields[CoordKeyField.cdp_x.value] = x_head_loc
    extra_byte_fields[CoordKeyField.cdp_y.value] = y_head_loc

    # Start by scraping the headers.
    head_df = segy_header_scrape(segyfile, silent=silent, **segyio_kwargs)
    head_bin = segy_bin_scrape(segyfile, **segyio_kwargs)

    # Scale Coordinates
    coord_scalar = head_df.SourceGroupScalar.median()
    coord_scalar_mult = np.power(abs(coord_scalar), np.sign(coord_scalar))
    head_df[x_head_loc] = head_df[x_head_loc].astype(float)
    head_df[y_head_loc] = head_df[y_head_loc].astype(float)
    head_df[x_head_loc] = head_df[x_head_loc] * coord_scalar_mult * 1.0
    head_df[y_head_loc] = head_df[y_head_loc] * coord_scalar_mult * 1.0

    # TODO: might need to scale offsets as well?

    # Cropping
    if cdp_crop and cdp is not None:  # 2d cdp cropping
        cmp_head_loc = _get_tf(cdp)
        crop_min, crop_max = check_crop(
            cdp_crop, [head_df[cmp_head_loc].min(), head_df[cmp_head_loc].max()]
        )

        head_df = head_df.query(
            "@cmp_head_loc >= @crop_min & @cmp_head_loc <= @crop_max"
        )

    if ix_crop is not None and cdp is None:  # 3d inline/xline cropping
        il_head_loc = _get_tf(iline)
        xl_head_loc = _get_tf(xline)
        il_min, il_max, xl_min, xl_max = check_crop(
            ix_crop,
            [
                head_df[il_head_loc].min(),
                head_df[il_head_loc].max(),
                head_df[xl_head_loc].min(),
                head_df[xl_head_loc].max(),
            ],
        )
        query = " & ".join(
            [
                f"{il_head_loc} >= @il_min",
                f"{il_head_loc} <= @il_max",
                f"{xl_head_loc} >= @xl_min",
                f"{xl_head_loc} <= @xl_max",
            ]
        )
        head_df = head_df.query(query).copy(deep=True)

    # TODO: -> Could implement some cropping with a polygon here
    if xy_crop is not None and cdp is None:
        x_min, x_max, y_min, y_max = check_crop(
            xy_crop,
            [
                head_df[x_head_loc].min(),
                head_df[x_head_loc].max(),
                head_df[y_head_loc].min(),
                head_df[y_head_loc].max(),
            ],
        )
        query = " & ".join(
            [
                f"{x_head_loc} >= @x_min",
                f"{x_head_loc} <= @x_max",
                f"{y_head_loc} >= @y_min",
                f"{y_head_loc} <= @y_max",
            ]
        )
        head_df = head_df.query(query).copy(deep=True)

    common_kwargs = dict(
        zcrop=z_crop,
        ncfile=ncfile,
        offset=offset,
        vert_domain=vert_domain,
        data_type=data_type,
        return_geometry=return_geometry,
        silent=silent,
    )

    # 3d data needs iline and xline
    if iline is not None and xline is not None:
        ds = _3dsegy_loader(
            segyfile,
            head_df,
            head_bin,
            iline=iline,
            xline=xline,
            **common_kwargs,
            **segyio_kwargs,
        )
        indexer = ["il_index", "xl_index"]
        dims = (
            DimensionKeyField.threed_head.value
            if offset is None
            else DimensionKeyField.threed_ps_head.value
        )

    # 2d data
    elif cdp is not None:
        ds = _2dsegy_loader(
            segyfile, head_df, head_bin, cdp=cdp, **common_kwargs, **segyio_kwargs
        )
        indexer = ["cdp_index"]
        dims = (
            DimensionKeyField.twod_head.value
            if offset is None
            else DimensionKeyField.twod_ps_head.value
        )

    # fallbak to just a 2d array of traces
    else:
        ds = _2dsegy_loader(
            segyfile, head_df, head_bin, **common_kwargs, **segyio_kwargs
        )
        indexer = []
        dims = DimensionKeyField.cdp_2d.value

    indexer = indexer + ["off_index"] if offset is not None else indexer

    # we have some some geometry to assign headers to
    if cdp is not None or iline is not None:
        head_ds = head_df.set_index(indexer).to_xarray()
        for key, field in extra_byte_fields.items():
            ds[key] = (dims, head_ds[field].values)
        ds = ds.set_coords([CoordKeyField.cdp_x.value, CoordKeyField.cdp_y.value])
    # geometry is not known
    else:
        for key, field in extra_byte_fields.items():
            ds[key] = (dims, head_df[field].values)

    return ds
