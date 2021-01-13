import importlib
import xarray as xr
import segyio
import numpy as np
import pandas as pd
import warnings

try:
    has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None
    if has_ipywidgets:
        from tqdm.autonotebook import tqdm
    else:
        from tqdm import tqdm as tqdm
except ModuleNotFoundError:
    from tqdm import tqdm as tqdm

from .._accessor import open_seisnc
from .._core import FrozenDict
from .._keyfield import CoordKeyField
from ._segy_text import create_default_texthead, put_segy_texthead, _clean_texthead
from ._segy_globals import _ISEGY_MEASUREMENT_SYSTEM

TQDM_ARGS = dict(unit=" traces", desc="Writing to SEG-Y")

OUTPUT_BYTES = FrozenDict(
    dict(
        standard_3d=dict(iline=181, xline=185, cdp_x=189, cdp_y=193),
        standard_3d_gath=dict(iline=181, xline=185, cdp_x=189, cdp_y=193, offset=37),
        standard_2d=dict(cdp=21, cdp_x=189, cdp_y=193),
        standard_2d_gath=dict(cdp=21, cdp_x=189, cdp_y=193, offset=37),
        petrel_3d=dict(iline=5, xline=21, cdp_x=73, cdp_y=77),
        petrel_2d=dict(cdp=21, cdp_x=73, cdp_y=77),
    )
)


def ncdf2segy(
    ncfile,
    segyfile,
    CMP=False,
    iline=189,
    xline=193,
    il_chunks=10,
    dimension=None,
    silent=False,
):
    """Convert siesnc format (NetCDF4) to SEGY.

    Args:
        ncfile (string): The input SEISNC file
        segyfile (string): The output SEGY file
        CMP (bool, optional): The data is 2D. Defaults to False.
        iline (int, optional): Inline byte location. Defaults to 189.
        xline (int, optional): Crossline byte location. Defaults to 193.
        il_chunks (int, optional): The size of data to work on - if you have memory
            limitations. Defaults to 10.
        dimension (str): Data dimension to output, defaults to 'twt' or 'depth' whichever is present
        silent (bool, optional): Turn off progress reporting. Defaults to False.
    """

    with open_seisnc(ncfile, chunks={"iline": il_chunks}) as seisnc:
        if dimension is None:
            if seisnc.seis.is_twt():
                dimension = "twt"
            elif seisnc.seis.is_depth():
                dimension = "depth"
            else:
                raise RuntimeError(
                    f"twt and depth dimensions missing, please specify a dimension to convert: {seisnc.dims}"
                )
        elif dimension not in seisnc.dims:
            raise RuntimeError(
                f"Requested output dimension not found in the dataset: {seisnc.dims}"
            )

        z0 = int(seisnc[dimension].values[0])
        ni, nj, nk = seisnc.dims["iline"], seisnc.dims["xline"], seisnc.dims[dimension]
        msys = _ISEGY_MEASUREMENT_SYSTEM[seisnc.seis.get_measurement_system()]
        spec = segyio.spec()

        # to create a file from nothing, we need to tell segyio about the structure of
        # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
        # more structural information, but offsets etc. have sensible defautls. This is
        # the absolute minimal specification for a N-by-M volume
        spec.sorting = 1
        spec.format = 1
        spec.iline = iline
        spec.xline = xline
        spec.samples = range(nk)
        spec.ilines = range(ni)
        spec.xlines = range(nj)

        xl_val = seisnc["xline"].values
        il_val = seisnc["iline"].values

        coord_scalar = seisnc.coord_scalar
        if coord_scalar is None:
            coord_scalar = 0
        coord_scalar_mult = np.power(abs(coord_scalar), np.sign(coord_scalar) * -1)

        il_bags = _bag_slices(seisnc["iline"].values, n=il_chunks)

        with segyio.create(segyfile, spec) as segyf:
            for ilb in tqdm(il_bags, desc="Writing to SEGY", disable=silent):
                ilbl = range(ilb.start, ilb.stop, ilb.step)
                data = seisnc.isel(iline=ilbl)
                for i, il in enumerate(ilbl):
                    il0, iln = il * nj, (il + 1) * nj
                    segyf.header[il0:iln] = [
                        {
                            segyio.su.offset: 1,
                            iline: il_val[il],
                            xline: xln,
                            segyio.su.cdpx: int(cdpx * coord_scalar_mult),
                            segyio.su.cdpy: int(cdpy * coord_scalar_mult),
                            segyio.su.ns: nk,
                            segyio.su.delrt: z0,
                            segyio.su.scalco: int(coord_scalar),
                        }
                        for xln, cdpx, cdpy in zip(
                            xl_val, data.cdp_x.values[i, :], data.cdp_y.values[i, :]
                        )
                    ]
                    segyf.trace[il * nj : (il + 1) * nj] = data.data[
                        i, :, :
                    ].values.astype(np.float32)
            segyf.bin.update(
                tsort=segyio.TraceSortingFormat.INLINE_SORTING,
                hdt=int(seisnc.sample_rate * 1000),
                hns=nk,
                mfeet=msys,
                jobid=1,
                lino=1,
                reno=1,
                ntrpr=ni * nj,
                nart=ni * nj,
                fold=1,
            )


def output_byte_locs(name):
    """Return common bytes position variable_dict for segy_writer.

    Args:
        name (str): One of [standard_3d, petrel_3d]

    Returns:
        dict: A dictionary of SEG-Y byte positions and default seisnc variable
            pairs.

    Example:
        >>> segywriter(ncfile, 'segyfile.sgy', trace_header_map=output_byte_loc('petrel'))
        >>>
    """
    try:
        return OUTPUT_BYTES[name]
    except KeyError:
        raise ValueError(
            f"No byte locatons for {name}, select from {list(OUTPUT_BYTES.keys())}"
        )


def _bag_slices(ind, n=10):
    """Take a list of indices and create a list of bagged indices. Each bag
    will contain n indices except for the last bag which will contain the
    remainder.

    This function is designed to support bagging/chunking of data to support
    memory management or distributed processing.

    Args:
        ind (list/array-like): The input list to create bags from.
        n (int, optional): The number of indices per bag. Defaults to 10.

    Returns:
        [type]: [description]
    """
    if n is None:  # one bad
        return [slice(0, len(ind), 1)]
    else:
        bag = list()
        prev = 0
        i = 0
        for i in range(len(ind)):
            if (i + 1) % n == 0:
                bag.append(slice(prev, i + 1, 1))
                prev = i + 1
        if prev != i + 1:
            bag.append(slice(prev, i + 1, 1))
        return bag


def _check_dimension(seisnc, dimension):
    if dimension is None:
        if seisnc.seis.is_twt():
            dimension = "twt"
        elif seisnc.seis.is_depth():
            dimension = "depth"
        else:
            raise RuntimeError(
                f"twt and depth dimensions missing, please specify a dimension to convert: {seisnc.dims}"
            )
    elif dimension not in seisnc.dims:
        raise RuntimeError(
            f"Requested output dimension not found in the dataset: {seisnc.dims}"
        )

    return dimension


def _build_trace_headers(ds, vert_dimension, other_dim_order, thm_args, cdp_x, cdp_y):
    # build trace headers for iteration and output
    trace_layout = ds.data.isel(**{vert_dimension: 0})
    trace_headers = pd.DataFrame()
    for var in thm_args:
        try:
            trace_headers[thm_args[var]] = (
                ds[var]
                .broadcast_like(trace_layout)
                .transpose(*other_dim_order, transpose_coords=True)
                .values.ravel()
            )
        except KeyError:  # The requested variable isn't in the list, write zeros
            warnings.warn(
                f"The variable {var} was not found in the input data, writing zeros to header instead"
            )

            if ds.seis.is_2d() or ds.seis.is_2dgath():
                base_var = "cdp"
            else:
                base_var = "iline"

            trace_headers[thm_args[var]] = (
                ds[base_var]
                .broadcast_like(trace_layout)
                .transpose(*other_dim_order, transpose_coords=True)
                .values.ravel()
                * 0.0
            )
    # scale coords
    trace_headers[cdp_x] = trace_headers[cdp_x] * ds.coord_scalar_mult
    trace_headers[cdp_y] = trace_headers[cdp_y] * ds.coord_scalar_mult
    trace_headers[cdp_x] = trace_headers[cdp_x]
    trace_headers[cdp_y] = trace_headers[cdp_y]

    trace_headers = trace_headers.astype(np.int32)

    return trace_headers


def _ncdf2segy_3d(
    ds,
    segyfile,
    iline=None,
    xline=None,
    cdp_x=None,
    cdp_y=None,
    il_chunks=10,
    dimension=None,
    text=None,
    silent=False,
    **thm_args,
):
    """Convert etlpy siesnc format (NetCDF4) to SEGY.

    Args:
        ds (xarray.Dataset): The input SEISNC dataset
        segyfile (string): The output SEGY file
        iline (int): Inline byte location.
        xline (int): Crossline byte location.
        cdp_x (int): The byte location to write the cdp_x.
        cdp_y (int): The byte location to write the cdp_y.
        il_chunks (int, optional): The size of data to work on - if you have memory
            limitations. Defaults to 10.
        dimension (str): Data dimension to output, defaults to 'twt' or 'depth' whichever is present
        text (dict, optional): A dictionary of strings to write out to the text header.
        silent (bool, optional): Turn off progress reporting. Defaults to False.
        thm_args (int): Byte locations to write trace header variables defined by key word arguments.
            keys for thm_args must be variables or coordinates or dimensions in seisnc.
    """

    thm_args[CoordKeyField.iline] = iline
    thm_args[CoordKeyField.xline] = xline
    thm_args[CoordKeyField.cdp_x] = cdp_x
    thm_args[CoordKeyField.cdp_y] = cdp_y

    z0 = int(ds[dimension].values[0])
    ni, nj, nk = (
        ds.dims[CoordKeyField.iline],
        ds.dims[CoordKeyField.xline],
        ds.dims[dimension],
    )
    order = (CoordKeyField.iline, CoordKeyField.xline)
    msys = _ISEGY_MEASUREMENT_SYSTEM[ds.seis.get_measurement_system()]
    spec = segyio.spec()

    # to create a file from nothing, we need to tell segyio about the structure of
    # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
    # more structural information, but offsets etc. have sensible defautls. This is
    # the absolute minimal specification for a N-by-M volume
    spec.sorting = 1
    spec.format = 1
    spec.iline = iline
    spec.xline = xline
    spec.samples = range(nk)
    spec.ilines = range(ni)
    spec.xlines = range(nj)

    trace_headers = _build_trace_headers(ds, dimension, order, thm_args, cdp_x, cdp_y)

    # header constants
    for byte, val in zip(
        [segyio.su.offset, segyio.su.ns, segyio.su.delrt, segyio.su.scalco],
        [1, nk, z0, int(ds.coord_scalar)],
    ):
        if byte not in trace_headers.columns:  # don't override user values
            trace_headers[byte] = val

    # to records
    trace_headers = trace_headers.to_dict(orient="records")

    il_bags = _bag_slices(ds[CoordKeyField.iline].values, n=il_chunks)
    with segyio.create(segyfile, spec) as segyf:
        pbar = tqdm(total=ni * nj, disable=silent, **TQDM_ARGS)
        for ilb in il_bags:
            ilbl = range(ilb.start, ilb.stop, ilb.step)
            data = ds.isel(iline=ilbl).transpose(*order, dimension)
            for i, il in enumerate(ilbl):
                il0, iln = il * nj, (il + 1) * nj
                segyf.header[il0:iln] = trace_headers[il0:iln]
                segyf.trace[il0:iln] = data.data[i, :, :].values.astype(np.float32)
                pbar.update(nj)
        pbar.close()

        segyf.bin.update(
            tsort=segyio.TraceSortingFormat.INLINE_SORTING,
            hdt=int(ds.sample_rate * 1000),
            hns=nk,
            mfeet=msys,
            jobid=1,
            lino=1,
            reno=1,
            ntrpr=ni * nj,
            nart=ni * nj,
            fold=1,
        )

    if not text:
        # create text header
        overrides = {
            #     123456789012345678901234567890123456789012345678901234567890123456
            7: f"Original File: {ds.source_file}",
            35: "*** BYTE LOCATION OF KEY HEADERS ***",
            36: f"CMP UTM-X: {cdp_x}, ALL COORDS SCALED BY: {ds.coord_scalar_mult},"
            f" CMP UTM-Y: {cdp_y}",
            37: f"INLINE: {iline}, XLINE: {xline}, ",
            40: "END TEXTUAL HEADER",
        }
        put_segy_texthead(
            segyfile,
            create_default_texthead(overrides),
        )
    else:
        put_segy_texthead(segyfile, text, line_counter=False)


def _ncdf2segy_3dgath(
    ds,
    segyfile,
    iline=None,
    xline=None,
    offset=None,
    cdp_x=None,
    cdp_y=None,
    il_chunks=10,
    dimension=None,
    text=None,
    silent=False,
    **thm_args,
):
    """Convert etlpy siesnc format (NetCDF4) to SEGY.

    Args:
        ds (xarray.Dataset): The input SEISNC dataset
        segyfile (string): The output SEGY file
        iline (int): Inline byte location.
        xline (int): Crossline byte location.
        cdp_x (int): The byte location to write the cdp_x.
        cdp_y (int): The byte location to write the cdp_y.
        offset (int): Offset byte location.
        il_chunks (int, optional): The size of data to work on - if you have memory
            limitations. Defaults to 10.
        dimension (str): Data dimension to output, defaults to 'twt' or 'depth' whichever is present
        text (dict, optional): A dictionary of strings to write out to the text header.
        silent (bool, optional): Turn off progress reporting. Defaults to False.
        thm_args (int): Byte locations to write trace header variables defined by key word arguments.
            keys for thm_args must be variables or coordinates or dimensions in seisnc.
    """

    thm_args[CoordKeyField.iline] = iline
    thm_args[CoordKeyField.xline] = xline
    thm_args[CoordKeyField.cdp_x] = cdp_x
    thm_args[CoordKeyField.cdp_y] = cdp_y
    thm_args[CoordKeyField.offset] = offset

    z0 = int(ds[dimension].values[0])
    ni, nj, nk, nl = (
        ds.dims[CoordKeyField.iline],
        ds.dims[CoordKeyField.xline],
        ds.dims[dimension],
        ds.dims[CoordKeyField.offset],
    )
    order = (CoordKeyField.iline, CoordKeyField.xline, CoordKeyField.offset)
    msys = _ISEGY_MEASUREMENT_SYSTEM[ds.seis.get_measurement_system()]
    spec = segyio.spec()

    # to create a file from nothing, we need to tell segyio about the structure of
    # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
    # more structural information, but offsets etc. have sensible defautls. This is
    # the absolute minimal specification for a N-by-M volume
    spec.format = 1
    spec.iline = iline
    spec.xline = xline
    spec.samples = ds[dimension].astype(int).values
    spec.ilines = ds[CoordKeyField.iline].astype(int).values
    spec.xlines = ds[CoordKeyField.xline].astype(int).values
    spec.offsets = ds[CoordKeyField.offset].astype(int).values

    trace_headers = _build_trace_headers(ds, dimension, order, thm_args, cdp_x, cdp_y)

    # header constants
    for byte, val in zip(
        [segyio.su.ns, segyio.su.delrt, segyio.su.scalco],
        [nk, z0, int(ds.coord_scalar)],
    ):
        if byte not in trace_headers.columns:  # don't override user values
            trace_headers[byte] = val

    # to records
    trace_headers = trace_headers.to_dict(orient="records")

    # print(len(trace_headers))

    il_bags = _bag_slices(ds[CoordKeyField.iline].values, n=il_chunks)
    with segyio.create(segyfile, spec) as segyf:
        pbar = tqdm(total=nj * nl * ni, disable=silent, **TQDM_ARGS)
        for ilb in il_bags:
            ilbl = range(ilb.start, ilb.stop, ilb.step)
            data = ds.isel(iline=ilbl).transpose(*order, dimension)
            for i, il in enumerate(ilbl):
                t0, tn = il * nj * nl, (il + 1) * nj * nl
                segyf.header[t0:tn] = trace_headers[t0:tn]
                segyf.trace[t0:tn] = (
                    data.data[i, :, :, :].values.reshape(nj * nl, nk).astype(np.float32)
                )
                pbar.update(nj * nl)
        pbar.close()
        segyf.bin.update(
            tsort=segyio.TraceSortingFormat.UNKNOWN_SORTING,  # trace sorting
            hdt=int(ds.sample_rate * 1000),  # Interval
            hns=nk,  # samples
            mfeet=msys,  # measurement system
            jobid=1,  # jobid
            lino=1,  # line number
            reno=1,  # reel number
            ntrpr=ni * nj * nl,  # n traces
            nart=0,  # aux traces
            fold=1,
        )

    if not text:
        # create text header
        overrides = {
            #     123456789012345678901234567890123456789012345678901234567890123456
            7: f"Original File: {ds.source_file}",
            35: "*** BYTE LOCATION OF KEY HEADERS ***",
            36: f"CMP UTM-X: {cdp_x}, ALL COORDS SCALED BY: {ds.coord_scalar_mult},"
            f" CMP UTM-Y: {cdp_y}",
            37: f"INLINE: {iline}, XLINE: {xline}, ",
            40: "END TEXTUAL HEADER",
        }
        put_segy_texthead(segyfile, create_default_texthead(overrides))
    else:
        put_segy_texthead(segyfile, text, line_counter=False)


def _ncdf2segy_2d(
    ds,
    segyfile,
    cdp=None,
    cdp_x=None,
    cdp_y=None,
    dimension=None,
    text=None,
    silent=False,
    **thm_args,
):
    """Convert etlpy siesnc format (NetCDF4) to SEGY.

    Args:
        ds (xarray.Dataset): The input SEISNC dataset
        segyfile (string): The output SEGY file
        cdp (int): The byte location to write the cdp.
        cdp_x (int): The byte location to write the cdp_x.
        cdp_y (int): The byte location to write the cdp_y.
        dimension (str): Data dimension to output, defaults to 'twt' or 'depth' whichever is present
        text (dict, optional): A dictionary of strings to write out to the text header.
        silent (bool, optional): Turn off progress reporting. Defaults to False.
        thm_args (int): Byte locations to write trace header variables defined by key word arguments.
            keys for thm_args must be variables or coordinates or dimensions in seisnc.
    """

    thm_args[CoordKeyField.cdp] = cdp
    thm_args[CoordKeyField.cdp_x] = cdp_x
    thm_args[CoordKeyField.cdp_y] = cdp_y

    z0 = int(ds[dimension].values[0])
    ni, nk = (
        ds.dims[CoordKeyField.cdp],
        ds.dims[dimension],
    )
    order = (CoordKeyField.cdp,)
    msys = _ISEGY_MEASUREMENT_SYSTEM[ds.seis.get_measurement_system()]
    spec = segyio.spec()

    # to create a file from nothing, we need to tell segyio about the structure of
    # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
    # more structural information, but offsets etc. have sensible defautls. This is
    # the absolute minimal specification for a N-by-M volume
    spec.format = 1
    spec.iline = cdp
    spec.xline = 185
    spec.samples = range(nk)
    spec.tracecount = ni

    trace_headers = _build_trace_headers(ds, dimension, order, thm_args, cdp_x, cdp_y)

    # header constants
    for byte, val in zip(
        [segyio.su.offset, segyio.su.ns, segyio.su.delrt, segyio.su.scalco],
        [1, nk, z0, int(ds.coord_scalar)],
    ):
        if byte not in trace_headers.columns:  # don't override user values
            trace_headers[byte] = val

    # to records
    trace_headers = trace_headers.to_dict(orient="records")

    with segyio.create(segyfile, spec) as segyf:
        data = ds.transpose(*order, dimension)
        segyf.header[:] = trace_headers
        segyf.trace[:] = data.data.values.astype(np.float32)

        segyf.bin.update(
            tsort=segyio.TraceSortingFormat.INLINE_SORTING,
            hdt=int(ds.sample_rate * 1000),
            hns=nk,
            mfeet=msys,
            jobid=1,
            lino=1,
            reno=1,
            ntrpr=ni,
            nart=0,
            fold=1,
        )

    if not text:
        # create text header
        overrides = {
            #     123456789012345678901234567890123456789012345678901234567890123456
            7: f"Original File: {ds.source_file}",
            35: "*** BYTE LOCATION OF KEY HEADERS ***",
            36: f"CMP UTM-X: {cdp_x}, ALL COORDS SCALED BY: {ds.coord_scalar_mult},"
            f" CMP UTM-Y: {cdp_y}",
            37: f"CDP: {cdp}, ",
            40: "END TEXTUAL HEADER",
        }
        put_segy_texthead(
            segyfile,
            create_default_texthead(overrides),
        )
    else:
        put_segy_texthead(segyfile, text, line_counter=False)


def _ncdf2segy_2d_gath(
    ds,
    segyfile,
    cdp=None,
    cdp_x=None,
    cdp_y=None,
    offset=None,
    dimension=None,
    text=None,
    silent=False,
    **thm_args,
):
    """Convert etlpy siesnc format (NetCDF4) to SEGY.

    Args:
        ds (xarray.Dataset): The input SEISNC dataset
        segyfile (string): The output SEGY file
        cdp (int): The byte location to write the cdp.
        cdp_x (int): The byte location to write the cdp_x.
        cdp_y (int): The byte location to write the cdp_y.
        offset (int): The byte location to write the offset.
        dimension (str): Data dimension to output, defaults to 'twt' or 'depth' whichever is present
        text (dict, optional): A dictionary of strings to write out to the text header.
        silent (bool, optional): Turn off progress reporting. Defaults to False.
        thm_args (int): Byte locations to write trace header variables defined by key word arguments.
            keys for thm_args must be variables or coordinates or dimensions in seisnc.
    """

    thm_args[CoordKeyField.cdp] = cdp
    thm_args[CoordKeyField.cdp_x] = cdp_x
    thm_args[CoordKeyField.cdp_y] = cdp_y
    thm_args[CoordKeyField.offset] = offset

    z0 = int(ds[dimension].values[0])
    ni, nk, nl = (
        ds.dims[CoordKeyField.cdp],
        ds.dims[dimension],
        ds.dims[CoordKeyField.offset],
    )
    order = (CoordKeyField.cdp, CoordKeyField.offset)
    msys = _ISEGY_MEASUREMENT_SYSTEM[ds.seis.get_measurement_system()]
    spec = segyio.spec()

    # to create a file from nothing, we need to tell segyio about the structure of
    # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
    # more structural information, but offsets etc. have sensible defautls. This is
    # the absolute minimal specification for a N-by-M volume
    spec.format = 1
    spec.iline = cdp
    spec.xline = 185
    spec.samples = range(nk)
    spec.tracecount = ni * nl

    trace_headers = _build_trace_headers(ds, dimension, order, thm_args, cdp_x, cdp_y)

    # header constants
    for byte, val in zip(
        [segyio.su.ns, segyio.su.delrt, segyio.su.scalco],
        [nk, z0, int(ds.coord_scalar)],
    ):
        if byte not in trace_headers.columns:  # don't override user values
            trace_headers[byte] = val

    # to records
    trace_headers = trace_headers.to_dict(orient="records")

    with segyio.create(segyfile, spec) as segyf:
        data = ds.transpose(*order, dimension)
        segyf.header[:] = trace_headers
        segyf.trace[:] = data.data.values.reshape(-1, nk).astype(np.float32)

        segyf.bin.update(
            tsort=segyio.TraceSortingFormat.INLINE_SORTING,
            hdt=int(ds.sample_rate * 1000),
            hns=nk,
            mfeet=msys,
            jobid=1,
            lino=1,
            reno=1,
            ntrpr=ni * nl,
            nart=0,
            fold=1,
        )

    if not text:
        # create text header
        overrides = {
            #     123456789012345678901234567890123456789012345678901234567890123456
            7: f"Original File: {ds.source_file}",
            35: "*** BYTE LOCATION OF KEY HEADERS ***",
            36: f"CMP UTM-X: {cdp_x}, ALL COORDS SCALED BY: {ds.coord_scalar_mult},"
            f" CMP UTM-Y: {cdp_y}",
            37: f"CDP: {cdp}, OFFSET: {offset}",
            40: "END TEXTUAL HEADER",
        }
        put_segy_texthead(
            segyfile,
            create_default_texthead(overrides),
        )
    else:
        put_segy_texthead(segyfile, text, line_counter=False)


def _segy_writer_input_handler(
    ds, segyfile, trace_header_map, dimension, silent, il_chunks, text
):
    """Handler for open seisnc dataset"""

    dimension = _check_dimension(ds, dimension)

    # ensure there is a coord_scalar
    coord_scalar = ds.coord_scalar
    if coord_scalar is None:
        coord_scalar = 0
        ds.attrs["coord_scalar"] = coord_scalar
    coord_scalar_mult = np.power(abs(coord_scalar), np.sign(coord_scalar) * -1)
    ds.attrs["coord_scalar_mult"] = coord_scalar_mult

    # create empty trace header map if necessary
    if trace_header_map is None:
        trace_header_map = dict()
    else:  # check that values of thm in ds
        for key in trace_header_map:
            try:
                _ = ds[key]
            except KeyError:
                raise ValueError("keys of trace_header_map must be in seisnc")

    # transfrom text if requested
    if text:
        text = {i + 1: line for i, line in enumerate(ds.text.split("\n"))}
        text = _clean_texthead(text, 80)

    common_kwargs = dict(silent=silent, dimension=dimension, text=text)

    if ds.seis.is_2d():
        thm = output_byte_locs("standard_2d")
        thm.update(trace_header_map)
        _ncdf2segy_2d(ds, segyfile, **common_kwargs, **thm)
    elif ds.seis.is_2dgath():
        thm = output_byte_locs("standard_2d_gath")
        _ncdf2segy_2d_gath(ds, segyfile, **common_kwargs, **thm)
    elif ds.seis.is_3d():
        thm = output_byte_locs("standard_3d")
        thm.update(trace_header_map)
        _ncdf2segy_3d(
            ds,
            segyfile,
            **common_kwargs,
            il_chunks=il_chunks,
            **thm,
        )
    elif ds.seis.is_3dgath():
        thm = output_byte_locs("standard_3d_gath")
        thm.update(trace_header_map)
        _ncdf2segy_3dgath(
            ds,
            segyfile,
            **common_kwargs,
            il_chunks=il_chunks,
            **thm,
        )
    else:
        # cannot determine type of data writing a continuous traces
        raise NotImplementedError()


def segy_writer(
    seisnc,
    segyfile,
    trace_header_map=None,
    il_chunks=None,
    dimension=None,
    silent=False,
    use_text=False,
):
    """Convert siesnc format (NetCDF4) to SEGY.

    Args:
        seisnc (xarray.Dataset, string): The input SEISNC file either a path or the in memory xarray.Dataset
        segyfile (string): The output SEGY file
        trace_header_map (dict, optional): Defaults to None. A dictionary of seisnc variables
            and byte locations. The variable will be written to the trace headers in the
            assigned byte location. By default CMP=23, cdp_x=181, cdp_y=185, iline=189,
            xline=193.
        il_chunks (int, optional): The size of data to work on - if you have memory
            limitations. Defaults to 10. This is primarily used for large 3D and ignored for 2D data.
        dimension (str): Data dimension to output, defaults to 'twt' or 'depth' whichever is present
        silent (bool, optional): Turn off progress reporting. Defaults to False.
        use_text (book, optional): Use the seisnc text for the EBCIDC output. This text usally comes from
            the loaded SEG-Y file and may not match the segysak SEG-Y output. Defaults to False and writes
            the default segysak EBCIDC
    """
    if isinstance(seisnc, xr.Dataset):
        _segy_writer_input_handler(
            seisnc, segyfile, trace_header_map, dimension, silent, None, use_text
        )
    else:
        ncfile = seisnc
        if isinstance(il_chunks, int):
            chunking = {"iline": il_chunks}
        else:
            chunking = None
        with open_seisnc(ncfile, chunks=chunking) as seisnc:
            _segy_writer_input_handler(
                seisnc,
                segyfile,
                trace_header_map,
                dimension,
                silent,
                il_chunks,
                use_text,
            )
