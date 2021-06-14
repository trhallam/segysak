# pylint: disable=invalid-name, unused-variable
"""Functions to interact with segy data
"""

import pathlib
from numpy.core.fromnumeric import trace

import segyio

# import netCDF
import h5netcdf

# from attrdict import AttrDict
from addict import Dict
import numpy as np
import xarray as xr

from ._segy_core import tqdm, check_tracefield

TQDM_ARGS = dict(unit_scale=True, unit=" traces")
PERCENTILES = [0, 0.1, 10, 50, 90, 99.9, 100]

KNOWN_BYTES = dict(
    standard_3d=dict(iline=181, xline=185, cdpx=189, cdpy=193),
    standard_3d_gath=dict(iline=181, xline=185, cdpx=189, cdpy=193, offset=37),
    standard_2d=dict(cdp=21, cdpx=189, cdpy=193),
    standard_2d_gath=dict(cdp=21, cdpx=189, cdpy=193, offset=37),
    petrel_3d=dict(iline=5, xline=21, cdpx=73, cdpy=77),
    petrel_2d=dict(cdp=21, cdpx=73, cdpy=77),
)

from segysak._keyfield import (
    CoordKeyField,
    AttrKeyField,
    VariableKeyField,
    VerticalKeyDim,
    DimensionKeyField,
)
from segysak._seismic_dataset import (
    create_seismic_dataset,
    _dataset_coordinate_helper,
)

# from segysak.seisnc import create_empty_seisnc, set_seisnc_dims
from segysak.tools import check_crop, check_zcrop
from segysak._accessor import open_seisnc
from segysak._core import FrozenDict

KNOWN_BYTES = FrozenDict(KNOWN_BYTES)

from ._segy_headers import (
    segy_bin_scrape,
    segy_header_scrape,
    what_geometry_am_i,
    _active_tracefield_segyio,
)
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
    return series.map(mapper.get)


def _segy3d_ncdf(
    segyfile,
    ncfile,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 3d data into a netcdf4 file. This function should
    stream data to disk avoiding excessive memory usage.
    """

    if vert_domain == "TWT":
        dims = DimensionKeyField.threed_twt
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.threed_depth
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf, h5netcdf.File(
        ncfile, "a"
    ) as seisnc:

        segyf.mmap()

        # create data variable
        seisnc_data = seisnc.create_variable(VariableKeyField.data, dims, np.float32)

        seisnc.flush()

        # work out fast and slow dir
        if head_df[head_loc.xline].diff().min() < 0:
            contig_dir = head_loc.iline
            broken_dir = head_loc.xline
            # slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = head_loc.xline
            broken_dir = head_loc.iline
            # slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )

        percentiles = np.zeros_like(PERCENTILES)
        for _, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                i1, i2 = val[["il_index", "xl_index"]].values.astype(int)
                seisnc_data[i1, i2, :] = segyf.trace[trc][n0 : ns + 1].astype(
                    np.float32
                )
                percentiles = (
                    np.percentile(segyf.trace[trc][n0 : ns + 1], PERCENTILES)
                    + percentiles
                ) / 2.0

                pb.update()
        pb.close()
        seisnc.attrs[AttrKeyField.percentiles] = list(percentiles)
        seisnc.attrs[AttrKeyField.coord_scalar] = head_df.SourceGroupScalar.median()
        seisnc.flush()

    return ncfile


def _segy3dps_ncdf(
    segyfile,
    ncfile,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 3d pre-stack data into a netcdf4 file. This function should
    stream data to disk avoiding excessive memory usage.
    """

    if vert_domain == "TWT":
        dims = DimensionKeyField.threed_ps_twt
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.threed_ps_depth
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf, h5netcdf.File(
        ncfile, "a"
    ) as seisnc:

        segyf.mmap()

        # create data variable
        seisnc_data = seisnc.create_variable(VariableKeyField.data, dims, np.float32)

        seisnc.flush()

        # work out fast and slow dir
        if head_df[head_loc.xline].diff().min() < 0:
            contig_dir = head_loc.iline
            broken_dir = head_loc.xline
            slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = head_loc.xline
            broken_dir = head_loc.iline
            slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )

        percentiles = np.zeros_like(PERCENTILES)

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                seisnc_data[val.il_index, val.xl_index, :, val.off_index] = segyf.trace[
                    trc
                ][n0 : ns + 1]

                percentiles = (
                    np.percentile(segyf.trace[trc][n0 : ns + 1], PERCENTILES)
                    + percentiles
                ) / 2.0

                pb.update()
        pb.close()

        seisnc.attrs[AttrKeyField.percentiles] = list(percentiles)
        seisnc.flush()

    return ncfile


def _segy3d_xr(
    segyfile,
    ds,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 3d data into an xarray with the seisnc form."""

    if vert_domain == "TWT":
        dims = DimensionKeyField.threed_twt
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.threed_depth
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()

        # work out fast and slow dir
        if head_df[head_loc.xline].diff().min() < 0:
            contig_dir = head_loc.iline
            broken_dir = head_loc.xline
            # slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = head_loc.xline
            broken_dir = head_loc.iline
            # slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )
        shape = [ds.dims[d] for d in dims]
        volume = np.zeros(shape, dtype=np.float32)
        percentiles = np.zeros_like(PERCENTILES)

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                volume[int(val.il_index), int(val.xl_index), :] = segyf.trace[trc][
                    n0 : ns + 1
                ]
                percentiles = (
                    np.percentile(segyf.trace[trc][n0 : ns + 1], PERCENTILES)
                    + percentiles
                ) / 2.0
                pb.update()
        pb.close()

    ds[VariableKeyField.data] = (dims, volume)
    ds.attrs[AttrKeyField.percentiles] = list(percentiles)

    return ds


def _segy3dps_xr(
    segyfile,
    ds,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 3d pre-stack data into an xarray with the seisnc form."""

    if vert_domain == "TWT":
        dims = DimensionKeyField.threed_ps_twt
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.threed_ps_depth
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()

        # work out fast and slow dir
        if head_df[head_loc.xline].diff().min() < 0:
            contig_dir = head_loc.iline
            broken_dir = head_loc.xline
            slicer = lambda x, y: slice(x, y, ...)
        else:
            contig_dir = head_loc.xline
            broken_dir = head_loc.iline
            slicer = lambda x, y: slice(y, x, ...)

        print(f"Fast direction is {broken_dir}")

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )
        shape = [ds.dims[d] for d in dims]
        volume = np.zeros(shape, dtype=np.float32)
        percentiles = np.zeros_like(PERCENTILES)

        for contig, grp in head_df.groupby(contig_dir):
            for trc, val in grp.iterrows():
                volume[
                    int(val.il_index), int(val.xl_index), :, int(val.off_index)
                ] = segyf.trace[trc][n0 : ns + 1]
                percentiles = (
                    np.percentile(segyf.trace[trc][n0 : ns + 1], PERCENTILES)
                    + percentiles
                ) / 2.0
                pb.update()
        pb.close()

    ds[VariableKeyField.data] = (dims, volume)
    ds.attrs[AttrKeyField.percentiles] = list(percentiles)

    return ds


def _3dsegy_loader(
    segyfile,
    head_df,
    head_bin,
    head_loc,
    ncfile=None,
    vert_domain="TWT",
    data_type="AMP",
    crop=None,
    zcrop=None,
    silent=False,
    return_geometry=False,
    **segyio_kwargs,
):
    """Convert SEG-Y data to Xarray or Netcdf4

    This is a helper function for segy_loader. Users should use that function
    directly to load all segy data.

    """

    # get vertical sample ranges
    n0 = 0
    nsamp = head_bin["Samples"]
    ns0 = head_df.DelayRecordingTime.min()

    # short way to get inlines/xlines
    ilines = head_df[head_loc.iline].unique()
    xlines = head_df[head_loc.xline].unique()
    ilines = np.sort(ilines)
    xlines = np.sort(xlines)
    iline_index_map = {il: i for i, il in enumerate(ilines)}
    xline_index_map = {xl: i for i, xl in enumerate(xlines)}
    head_df.loc[:, "il_index"] = head_df[head_loc.iline].map(iline_index_map.get)
    head_df.loc[:, "xl_index"] = head_df[head_loc.xline].map(xline_index_map.get)

    # binary header translation
    ns = head_bin["Samples"]
    sample_rate = head_bin["Interval"] / 1000.0
    msys = _SEGY_MEASUREMENT_SYSTEM[head_bin["MeasurementSystem"]]

    # for offset
    if head_loc.offset is not None:
        offsets = head_df[head_loc.offset].unique()
        offsets = np.sort(offsets)
        offset_index_map = {off: i for i, off in enumerate(offsets)}
        head_df["off_index"] = head_df[head_loc.offset].map(offset_index_map.get)
    else:
        offsets = None

    if zcrop is not None:
        zcrop = check_zcrop(zcrop, [0, ns])
        n0, ns = zcrop
        ns0 = sample_rate * n0
        nsamp = ns - n0 + 1
    vert_samples = np.arange(ns0, ns0 + sample_rate * nsamp, sample_rate)

    builder, domain = _dataset_coordinate_helper(
        vert_samples, vert_domain, iline=ilines, xline=xlines, offset=offsets
    )

    ds = create_seismic_dataset(**builder)

    # create_seismic_dataset(d1=ni, d2=nx, d3=nsamp)
    text = get_segy_texthead(segyfile, **segyio_kwargs)
    ds.attrs[AttrKeyField.text] = text
    ds.attrs[AttrKeyField.source_file] = pathlib.Path(segyfile).name
    ds.attrs[AttrKeyField.measurement_system] = msys
    ds.attrs[AttrKeyField.sample_rate] = sample_rate
    ds.attrs[AttrKeyField.coord_scalar] = head_df.SourceGroupScalar.median()

    if ncfile is not None and return_geometry == False:
        ds.seisio.to_netcdf(ncfile)
    elif return_geometry:
        # return geometry -> e.g. don't process segy traces
        return ds
    else:
        ncfile = ds

    segyio_kwargs.update(
        dict(ignore_geometry=True, iline=head_loc.iline, xline=head_loc.xline)
    )

    # not prestack data
    if head_loc.offset is None and not isinstance(ncfile, xr.Dataset):
        ds = _segy3d_ncdf(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    # not prestack data load into memory
    if head_loc.offset is None and isinstance(ncfile, xr.Dataset):
        ds = _segy3d_xr(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    # prestack data
    if head_loc.offset is not None and not isinstance(ncfile, xr.Dataset):
        ds = _segy3dps_ncdf(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    # prestack data load into memory
    if head_loc.offset is not None and isinstance(ncfile, xr.Dataset):
        ds = _segy3dps_xr(
            segyfile,
            ncfile,
            segyio_kwargs,
            n0,
            ns,
            head_df,
            head_loc,
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
    head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 2d data into an xarray with the seisnc form."""

    if vert_domain == "TWT":
        dims = DimensionKeyField.twod_twt
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.twod_depth
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )
        shape = [ds.dims[d] for d in dims]
        volume = np.zeros(shape, dtype=np.float32)
        percentiles = np.zeros_like(PERCENTILES)

        # this can probably be done as a block - leaving for now just incase sorting becomes an issue
        for trc, val in head_df.iterrows():
            volume[int(val.cdp_index), :] = segyf.trace[trc][n0 : ns + 1]
            pb.update()
            percentiles = (
                np.percentile(segyf.trace[trc][n0 : ns + 1], PERCENTILES) + percentiles
            ) / 2.0
        pb.close()

    ds[VariableKeyField.data] = (
        dims,
        volume[:, n0 : ns + 1],
    )
    ds.attrs[AttrKeyField.percentiles] = list(percentiles)

    return ds


def _segy2d_ps_xr(
    segyfile,
    ds,
    segyio_kwargs,
    n0,
    ns,
    head_df,
    head_loc,
    vert_domain="TWT",
    silent=False,
):
    """Helper function to load 2d data into an xarray with the seisnc form."""

    if vert_domain == "TWT":
        dims = DimensionKeyField.twod_ps_twt
    elif vert_domain == "DEPTH":
        dims = DimensionKeyField.twod_ps_depth
    else:
        raise ValueError(f"Unknown vert_domain: {vert_domain}")

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()

        pb = tqdm(
            total=segyf.tracecount, desc="Converting SEGY", disable=silent, **TQDM_ARGS
        )
        shape = [ds.dims[d] for d in dims]
        volume = np.zeros(shape, dtype=np.float32)
        percentiles = np.zeros_like(PERCENTILES)

        # this can probably be done as a block - leaving for now just incase sorting becomes an issue
        for trc, val in head_df.iterrows():
            volume[int(val.cdp_index), :, int(val.off_index)] = segyf.trace[trc][
                n0 : ns + 1
            ]
            percentiles = (
                np.percentile(segyf.trace[trc][n0 : ns + 1], PERCENTILES) + percentiles
            ) / 2.0
            pb.update()
        pb.close()

    ds[VariableKeyField.data] = (
        dims,
        volume,
    )
    ds.attrs[AttrKeyField.percentiles] = list(percentiles)

    return ds


def _2dsegy_loader(
    segyfile,
    head_df,
    head_bin,
    head_loc,
    ncfile=None,
    vert_domain="TWT",
    data_type="AMP",
    crop=None,
    zcrop=None,
    silent=False,
    return_geometry=False,
    **segyio_kwargs,
):
    """Convert SEG-Y data to Xarray or Netcdf4

    This is a helper function for segy_loader. Users should use that function
    directly to load all segy data.

    """

    # get names of columns where stuff we want is
    if head_loc.cdp is None:
        cdp = 21
        head_loc.cdp = _get_tf(cdp)
        head_df[head_loc.cdp] = head_df.index.values

    # get vertical sample ranges
    n0 = 0
    nsamp = head_bin["Samples"]
    ns0 = head_df.DelayRecordingTime.min()

    # short way to get cdps
    cdps = head_df[head_loc.cdp].unique()
    cdps = np.sort(cdps)
    head_df["cdp_index"] = _header_to_index_mapping(head_df[head_loc.cdp])

    # binary header translation
    nsamp = head_bin["Samples"]
    sample_rate = head_bin["Interval"] / 1000.0
    msys = _SEGY_MEASUREMENT_SYSTEM[head_bin["MeasurementSystem"]]

    # for offset
    if head_loc.offset is not None:
        offsets = head_df[head_loc.offset].unique()
        offsets = np.sort(offsets)
        head_df["off_index"] = _header_to_index_mapping(head_df[head_loc.offset])
    else:
        offsets = None

    if zcrop is not None:
        zcrop = check_zcrop(zcrop, [0, nsamp])
        n0, nsamp = zcrop
        ns0 = sample_rate * n0
        nsamp = nsamp - n0 + 1
    vert_samples = np.arange(ns0, ns0 + sample_rate * nsamp, sample_rate)

    builder, domain = _dataset_coordinate_helper(
        vert_samples, vert_domain, cdp=cdps, offset=offsets
    )

    ds = create_seismic_dataset(**builder)

    # create_seismic_dataset(d1=ni, d2=nx, d3=nsamp)
    text = get_segy_texthead(segyfile, **segyio_kwargs)
    ds.attrs[AttrKeyField.text] = text
    ds.attrs[AttrKeyField.source_file] = pathlib.Path(segyfile).name
    ds.attrs[AttrKeyField.measurement_system] = msys
    ds.attrs[AttrKeyField.sample_rate] = sample_rate
    ds.attrs[AttrKeyField.coord_scalar] = head_df.SourceGroupScalar.median()

    if ncfile is not None and return_geometry == True:
        ds.seisio.to_netcdf(ncfile)
        return ds
    elif return_geometry:
        return ds

    segyio_kwargs.update(dict(ignore_geometry=True))

    # stacked data
    if head_loc.offset is None:
        ds = _segy2d_xr(
            segyfile,
            ds,
            segyio_kwargs,
            n0,
            nsamp,
            head_df,
            head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    # # prestack data
    if head_loc.offset is not None:
        ds = _segy2d_ps_xr(
            segyfile,
            ds,
            segyio_kwargs,
            n0,
            nsamp,
            head_df,
            head_loc,
            vert_domain=vert_domain,
            silent=silent,
        )

    if ncfile is not None:
        ds.seisio.to_netcdf(ncfile)
        del ds
        ds = ncfile

    return ds


def well_known_byte_locs(name):
    """Return common bytes position kwargs_dict for segy_loader and segy_converter.

    Returns a dict containing the byte locations for well known SEG-Y variants in the wild.

    Args:
        name (str): Takes one of keys from KNOWN_BYTES

    Returns:
        dict: A dictionary of SEG-Y byte positions.

    Example:

    Use the output of this function to unpack arguments into ``segy_loader``

    >>> seismic = segy_loader(filepath, **well_known_byte_locs('petrel_3d'))

    """
    try:
        return KNOWN_BYTES[name]
    except KeyError:
        raise ValueError(
            f"No byte locatons for {name}, select from {list(KNOWN_BYTES.keys())}"
        )


def _loader_converter_checks(cdp, iline, xline, extra_byte_fields):
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
    return extra_byte_fields


def _loader_converter_header_handling(
    segyfile,
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
    head_df=None,
    optimised_load=False,
    **segyio_kwargs,
):

    if optimised_load:
        scrape_bytes = [
            val for val in [cdp, iline, xline, cdpx, cdpy, offset] if val is not None
        ]
        scrape_bytes += list(extra_byte_fields.values())
        scrape_bytes += [
            _active_tracefield_segyio()["DelayRecordingTime"],
            _active_tracefield_segyio()["SourceGroupScalar"],
        ]
    else:
        scrape_bytes = None

    if head_df is None:
        # Start by scraping the headers.
        head_df = segy_header_scrape(
            segyfile, silent=silent, bytes_filter=scrape_bytes, **segyio_kwargs
        )

    head_bin = segy_bin_scrape(segyfile, **segyio_kwargs)
    head_loc = Dict(
        dict(cdp=cdp, offset=offset, iline=iline, xline=xline, cdpx=cdpx, cdpy=cdpy)
    )

    if all(map(lambda x: x is None, (cdp, iline, xline, offset))):
        # lets try and guess the data types if no hints given
        try:
            head_loc.update(what_geometry_am_i(head_df))
        except (KeyError, ValueError):
            print("Couldn't determine geometry, will load traces as flat 2D.")
            pass

    head_loc = Dict(
        {
            key: (_get_tf(val) if val is not None else None)
            for key, val in head_loc.items()
        }
    )

    if all(v is not None for v in (head_loc.cdpx, head_loc.cdpy)):
        extra_byte_fields[CoordKeyField.cdp_x] = head_loc.cdpx
        extra_byte_fields[CoordKeyField.cdp_y] = head_loc.cdpy

        # Scale Coordinates
        coord_scalar = head_df.SourceGroupScalar.median()
        coord_scalar_mult = np.power(abs(coord_scalar), np.sign(coord_scalar))
        head_df[head_loc.cdpx] = head_df[head_loc.cdpx].astype(np.float32)
        head_df[head_loc.cdpy] = head_df[head_loc.cdpy].astype(np.float32)
        head_df[head_loc.cdpx] = head_df[head_loc.cdpx] * coord_scalar_mult * 1.0
        head_df[head_loc.cdpy] = head_df[head_loc.cdpy] * coord_scalar_mult * 1.0

    # TODO: might need to scale offsets as well? This isn't available in segy standard

    # Cropping
    if cdp_crop and cdp is not None:  # 2d cdp cropping
        crop_min, crop_max = check_crop(
            cdp_crop, [head_df[head_loc.cdp].min(), head_df[head_loc.cdp].max()]
        )

        head_df = head_df.query(
            "@head_loc.cdp >= @crop_min & @head_loc.cdp <= @crop_max"
        )

    if ix_crop is not None and cdp is None:  # 3d inline/xline cropping
        il_min, il_max, xl_min, xl_max = check_crop(
            ix_crop,
            [
                head_df[head_loc.iline].min(),
                head_df[head_loc.iline].max(),
                head_df[head_loc.xline].min(),
                head_df[head_loc.xline].max(),
            ],
        )
        query = " & ".join(
            [
                f"{head_loc.iline} >= @il_min",
                f"{head_loc.iline} <= @il_max",
                f"{head_loc.xline} >= @xl_min",
                f"{head_loc.xline} <= @xl_max",
            ]
        )
        head_df = head_df.query(query).copy(deep=True)

    # TODO: -> Could implement some cropping with a polygon here
    if xy_crop is not None and cdp is None:
        x_min, x_max, y_min, y_max = check_crop(
            xy_crop,
            [
                head_df[head_loc.cdpx].min(),
                head_df[head_loc.cdpx].max(),
                head_df[head_loc.cdpy].min(),
                head_df[head_loc.cdpy].max(),
            ],
        )
        query = " & ".join(
            [
                f"{head_loc.cdpx} >= @x_min",
                f"{head_loc.cdpx} <= @x_max",
                f"{head_loc.cdpy} >= @y_min",
                f"{head_loc.cdpy} <= @y_max",
            ]
        )
        head_df = head_df.query(query).copy(deep=True)

    return head_df, head_bin, head_loc


def _loader_converter_write_headers(
    ds, head_df, indexer, dims, extra_byte_fields, is3d2d=True
):

    if not isinstance(ds, xr.Dataset):
        ds = open_seisnc(ds)

    # we have some some geometry to assign headers to
    if is3d2d:
        head_ds = head_df.set_index(indexer).to_xarray()
        for key, field in extra_byte_fields.items():
            ds[key] = (dims, head_ds[field].values)
    # geometry is not known
    else:
        for key, field in extra_byte_fields.items():
            ds[key] = (dims, head_df[field].values)

    coords = (CoordKeyField.cdp_x, CoordKeyField.cdp_y)
    if set(coords).issubset(ds.variables.keys()):
        ds = ds.set_coords(coords)

    return ds


def segy_freeloader(
    segyfile,
    vert_domain="TWT",
    data_type="AMP",
    return_geometry=False,
    silent=False,
    extra_byte_fields=None,
    head_df=None,
    segyio_kwargs=None,
    **dim_kwargs,
):
    """Freeform loader for SEG-Y data. This loader allows you to load SEG-Y into
    an xarray.Dataset using an arbitrary number of header locations to create
    othogonal dimensions. This is an eager loader and will transfer the entire
    SEG-Y and requested header information to memory.

    From the dimension header locations specified the freeloader will try to
    create a Dataset where each trace is assigned to a dimension.

    Args:
        segyfile (string): The SEG-Y file/path.
        vert_domain (str, optional): One of ('TWT', 'DEPTH'). Defaults to 'TWT'.
        data_type (str, optional): Defaults to "AMP".
        return_geometry (bool, optional): If true, just returned the empty
            dataset based upon the calcuated header geometry. Defaults to False.
        silent (bool, optional): Turn off progress bars. Defaults to False.
        extra_byte_fields (dict, optional): Additional header information to
            load into the Dataset. Defaults to None.
        head_df (pandas.DataFrame): The DataFrame output from `segy_header_scrape`.
            This DataFrame can be filtered by the user
            to load select trace sets. Trace loading is based upon the DataFrame index.
        segyio_kwargs (dict, optional): Extra keyword arguments for segyio.open
        **dim_kwargs: Dimension names and byte location pairs.
    """
    if segyio_kwargs is None:
        segyio_kwargs = dict()

    if head_df is None:
        # Start by scraping the headers.
        head_df = segy_header_scrape(segyfile, silent=silent, **segyio_kwargs)

    head_bin = segy_bin_scrape(segyfile, **segyio_kwargs)

    # get vertical sample ranges
    n0 = 0
    nsamp = head_bin["Samples"]
    ns0 = head_df.DelayRecordingTime.min()

    # binary header translation
    nsamp = head_bin["Samples"]
    sample_rate = head_bin["Interval"] / 1000.0
    msys = _SEGY_MEASUREMENT_SYSTEM[head_bin["MeasurementSystem"]]
    vert_samples = np.arange(ns0, ns0 + sample_rate * nsamp, sample_rate, dtype=int)

    # creating dimensions and new dataset
    dims = dict()
    dim_index_names = list()
    dim_fields = list()
    for dim in dim_kwargs:
        trace_field = str(segyio.TraceField(dim_kwargs[dim]))
        if trace_field == "Unknown Enum":
            raise ValueError(f"{dim}:{dim_kwargs[dim]} was not a valid byte header")
        dim_fields.append(trace_field)
        as_unique = head_df[trace_field].unique()
        dims[dim] = np.sort(as_unique)
        d_map = {dval: i for i, dval in enumerate(as_unique)}
        index_name = f"{dim}_index"
        dim_index_names.append(index_name)
        head_df.loc[:, index_name] = head_df[trace_field].map(d_map)

    if (
        head_df[dim_index_names].shape
        != head_df[dim_index_names].drop_duplicates().shape
    ):
        raise ValueError(
            "The selected dimensions results in multiple traces per "
            "dimension location, add additional dimensions or use "
            "trace numbering to load as 2D."
        )

    builder, domain = _dataset_coordinate_helper(vert_samples, vert_domain, **dims)
    ds = create_seismic_dataset(**builder)

    # getting attributes
    text = get_segy_texthead(segyfile, **segyio_kwargs)
    ds.attrs[AttrKeyField.text] = text
    ds.attrs[AttrKeyField.source_file] = pathlib.Path(segyfile).name
    ds.attrs[AttrKeyField.measurement_system] = msys
    ds.attrs[AttrKeyField.sample_rate] = sample_rate

    # map extra byte fields into ds
    if extra_byte_fields is not None:
        to_add = list()
        for name, byte in extra_byte_fields.items():
            trace_field = str(segyio.TraceField(byte))
            if trace_field == "Unknown Enum":
                raise ValueError(f"{name}:{byte} was not a valid byte header")
            to_add.append(trace_field)
        to_add = to_add + dim_fields
        extras = head_df[to_add].set_index(dim_fields).to_xarray()
        extras = extras.rename_dims(
            {b: a for a, b in zip(dim_kwargs, dim_fields) if a != b}
        )
        for name, xtr in zip(extra_byte_fields, to_add):
            ds[name] = extras[xtr]

    if return_geometry:
        # return geometry -> e.g. don't process segy traces
        return ds

    segyio_kwargs.update(dict(ignore_geometry=True))

    with segyio.open(segyfile, "r", **segyio_kwargs) as segyf:

        segyf.mmap()
        shape = [ds.dims[d] for d in dim_kwargs] + [vert_samples.size]
        volume = np.zeros(shape, dtype=np.float32)

        # this can probably be done as a block - leaving for now just incase sorting becomes an issue
        indexes = tuple([head_df[idx].values for idx in dim_index_names])
        volume[indexes] = segyf.trace.raw[:][head_df.index.values]

    percentiles = np.percentile(volume, PERCENTILES)
    ds[VariableKeyField.data] = (
        list(dim_kwargs) + [VerticalKeyDim[domain]],
        volume,
    )
    ds.attrs[AttrKeyField.percentiles] = list(percentiles)

    return ds


def segy_loader(
    segyfile,
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
    head_df=None,
    **segyio_kwargs,
):
    """Load SEG-Y file into xarray.Dataset

    The output dataset has the following structure
        Dimensions:
            cdp/iline - CDP or Inline axis
            xline - Xline axis
            twt/depth - The vertical axis
            offset - Offset/Angle Axis
        Coordinates:
            iline - The inline numbering
            xline - The xline numbering
            cdp_x - Eastings
            cdp_y - Northings
            cdp - Trace Number for 2d
        Variables
            data - The data volume
        Attributes:
            ns - number of samples vertical
            sample_rate - sample rate in ms/m
            test - text header
            measurement_system : m/ft
            source_file : segy source
            srd : seismic reference datum
            percentiles : data amplitude percentiles
            coord_scalar : from trace headers

    Args:
        segyfile (str): Input segy file path
        cdp (int, optional): The CDP byte location, usually 21.
        iline (int, optional): Inline byte location, usually 189
        xline (int, optional): Cross-line byte location, usually 193
        cdpx (int, optional): UTMX byte location, usually 181
        cdpy (int, optional): UTMY byte location, usually 185
        offset (int, optional): Offset/angle byte location
        vert_domain (str, optional): Vertical sampling domain. One of ['TWT', 'DEPTH']. Defaults to 'TWT'.
        data_type (str, optional): Data type ['AMP', 'VEL']. Defaults to 'AMP'.
        ix_crop (list, optional): List of minimum and maximum inline and crossline to output.
            Has the form '[min_il, max_il, min_xl, max_xl]'. Ignored for 2D data.
        cdp_crop (list, optional): List of minimum and maximum cmp values to output.
            Has the form '[min_cmp, max_cmp]'. Ignored for 3D data.
        xy_crop (list, optional): List of minimum and maximum cdp_x and cdp_y to output.
            Has the form '[min_x, max_x, min_y, max_y]'. Ignored for 2D data.
        z_crop (list, optional): List of minimum and maximum vertical samples to output.
            Has the form '[min, max]'.
        return_geometry (bool, optional): If true returns an xarray.dataset which doesn't contain data but mirrors
            the input volume header information.
        silent (bool): Disable progress bar.
        extra_byte_fields (list/mapping): A list of int or mapping of byte fields that should be returned as variables in the dataset.
        head_df (pandas.DataFrame): The DataFrame output from `segy_header_scrape`. This DataFrame can be filtered by the user
            to load select trace sets. Trace loading is based upon the DataFrame index.
        **segyio_kwargs: Extra keyword arguments for segyio.open

    Returns:
        xarray.Dataset: If ncfile keyword is specified returns open handle to disk netcdf4,
            otherwise the data in memory. If return_geometry is True does not load trace data and
            returns headers in geometry.
    """
    extra_byte_fields = _loader_converter_checks(cdp, iline, xline, extra_byte_fields)

    head_df, head_bin, head_loc = _loader_converter_header_handling(
        segyfile,
        cdp=cdp,
        iline=iline,
        xline=xline,
        cdpx=cdpx,
        cdpy=cdpy,
        offset=offset,
        vert_domain=vert_domain,
        data_type=data_type,
        ix_crop=ix_crop,
        cdp_crop=cdp_crop,
        xy_crop=xy_crop,
        z_crop=z_crop,
        return_geometry=return_geometry,
        silent=silent,
        extra_byte_fields=extra_byte_fields,
        head_df=head_df,
        **segyio_kwargs,
    )

    byte_loc = [
        bytel for bytel in [cdp, iline, xline, cdpx, cdpy, offset] if bytel is not None
    ]
    check_tracefield(byte_loc)

    common_args = (segyfile, head_df, head_bin, head_loc)

    common_kwargs = dict(
        zcrop=z_crop,
        vert_domain=vert_domain,
        data_type=data_type,
        return_geometry=return_geometry,
        silent=silent,
    )

    # 3d data needs iline and xline
    if all(v is not None for v in (head_loc.iline, head_loc.xline)):
        print("Loading as 3D")
        ds = _3dsegy_loader(
            *common_args,
            **common_kwargs,
            **segyio_kwargs,
        )
        indexer = ["il_index", "xl_index"]
        dims = (
            DimensionKeyField.threed_head
            if offset is None
            else DimensionKeyField.threed_ps_head
        )
        is3d2d = True

    # 2d data
    elif head_loc.cdp is not None:
        print("Loading as 2D")
        ds = _2dsegy_loader(*common_args, **common_kwargs, **segyio_kwargs)
        indexer = ["cdp_index"]
        dims = (
            DimensionKeyField.twod_head
            if offset is None
            else DimensionKeyField.twod_ps_head
        )
        is3d2d = True

    # fallbak to just a 2d array of traces
    else:
        ds = _2dsegy_loader(*common_args, **common_kwargs, **segyio_kwargs)
        indexer = []
        dims = DimensionKeyField.cdp_2d
        is3d2d = False

    indexer = indexer + ["off_index"] if offset is not None else indexer

    ds = _loader_converter_write_headers(
        ds, head_df, indexer, dims, extra_byte_fields, is3d2d=is3d2d
    )

    # ds.seis.get_corner_points()
    return ds


def segy_converter(
    segyfile,
    ncfile,
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
    """Convert SEG-Y data to NetCDF4 File

    The output ncfile has the following structure
        Dimensions:
            cdp/iline - CDP or Inline axis
            xline - Xline axis
            twt/depth - The vertical axis
            offset - Offset/Angle Axis
        Coordinates:
            iline - The inline numbering
            xline - The xline numbering
            cdp_x - Eastings
            cdp_y - Northings
            cdp - Trace Number for 2d
        Variables
            data - The data volume
        Attributes:
            ns - number of samples vertical
            sample_rate - sample rate in ms/m
            test - text header
            measurement_system : m/ft
            source_file : segy source
            srd : seismic reference datum
            percentiles : data amplitude percentiles
            coord_scalar : from trace headers

    Args:
        segyfile (str): Input segy file path
        ncfile (str): Output SEISNC file path. If none the loaded data will be
            returned in memory as an xarray.Dataset.
        cdp (int, optional): The CDP byte location, usually 21.
        iline (int, optional): Inline byte location, usually 189
        xline (int, optional): Cross-line byte location, usually 193
        cdpx (int, optional): UTMX byte location, usually 181
        cdpy (int, optional): UTMY byte location, usually 185
        offset (int, optional): Offset/angle byte location
        vert_domain (str, optional): Vertical sampling domain. One of ['TWT', 'DEPTH']. Defaults to 'TWT'.
        data_type (str, optional): Data type ['AMP', 'VEL']. Defaults to 'AMP'.
        ix_crop (list, optional): List of minimum and maximum inline and crossline to output.
            Has the form '[min_il, max_il, min_xl, max_xl]'. Ignored for 2D data.
        cdp_crop (list, optional): List of minimum and maximum cmp values to output.
            Has the form '[min_cmp, max_cmp]'. Ignored for 3D data.
        xy_crop (list, optional): List of minimum and maximum cdp_x and cdp_y to output.
            Has the form '[min_x, max_x, min_y, max_y]'. Ignored for 2D data.
        z_crop (list, optional): List of minimum and maximum vertical samples to output.
            Has the form '[min, max]'.
        return_geometry (bool, optional): If true returns an xarray.dataset which doesn't contain data but mirrors
            the input volume header information.
        silent (bool): Disable progress bar.
        extra_byte_fields (list/mapping): A list of int or mapping of byte fields that should be returned as variables in the dataset.a
        **segyio_kwargs: Extra keyword arguments for segyio.open

    """
    # Input sanity checks
    extra_byte_fields = _loader_converter_checks(cdp, iline, xline, extra_byte_fields)
    byte_loc = [
        bytel for bytel in [cdp, iline, xline, cdpx, cdpy, offset] if bytel is not None
    ]
    check_tracefield(byte_loc)

    head_df, head_bin, head_loc = _loader_converter_header_handling(
        segyfile,
        cdp=cdp,
        iline=iline,
        xline=xline,
        cdpx=cdpx,
        cdpy=cdpy,
        offset=offset,
        vert_domain=vert_domain,
        data_type=data_type,
        ix_crop=ix_crop,
        cdp_crop=cdp_crop,
        xy_crop=xy_crop,
        z_crop=z_crop,
        return_geometry=return_geometry,
        silent=silent,
        extra_byte_fields=extra_byte_fields,
        optimised_load=True,
        **segyio_kwargs,
    )
    print("header_loaded")
    common_args = (segyfile, head_df, head_bin, head_loc)

    common_kwargs = dict(
        zcrop=z_crop,
        ncfile=ncfile,
        vert_domain=vert_domain,
        data_type=data_type,
        return_geometry=return_geometry,
        silent=silent,
    )

    # 3d data needs iline and xline
    if iline is not None and xline is not None:

        print("is_3d")
        ds = _3dsegy_loader(
            *common_args,
            **common_kwargs,
            **segyio_kwargs,
        )
        indexer = ["il_index", "xl_index"]
        dims = (
            DimensionKeyField.threed_head
            if offset is None
            else DimensionKeyField.threed_ps_head
        )
        is3d2d = True

    # 2d data
    elif cdp is not None:
        ds = _2dsegy_loader(*common_args, **common_kwargs, **segyio_kwargs)
        indexer = ["cdp_index"]
        dims = (
            DimensionKeyField.twod_head
            if offset is None
            else DimensionKeyField.twod_ps_head
        )
        is3d2d = True

    # fallbak to just a 2d array of traces
    else:
        ds = _2dsegy_loader(*common_args, **common_kwargs, **segyio_kwargs)
        indexer = []
        dims = DimensionKeyField.cdp_2d
        is3d2d = False

    indexer = indexer + ["off_index"] if offset is not None else indexer

    ds = _loader_converter_write_headers(
        ds, head_df, indexer, dims, extra_byte_fields, is3d2d=is3d2d
    )
    new_vars = {key: ds[key] for key in extra_byte_fields}
    ds.close()
    del ds

    with h5netcdf.File(ncfile, "a") as seisnc:
        for var, darray in new_vars.items():
            seisnc_var = seisnc.create_variable(var, darray.dims, darray.dtype)
            seisnc_var[...] = darray[...]
            seisnc.flush()

    return None
