import xarray as xr
import segyio
import numpy as np
import pandas as pd
from more_itertools import split_into, split_when
from itertools import product

from ._segy_core import tqdm, check_tracefield

from .._accessor import open_seisnc
from .._keyfield import CoordKeyField
from ._segy_text import (
    create_default_texthead,
    put_segy_texthead,
    _clean_texthead,
    trace_header_map_to_text,
)
from ._segy_globals import _ISEGY_MEASUREMENT_SYSTEM

TQDM_ARGS = dict(unit=" traces", desc="Writing to SEG-Y")


def _chunk_iterator(ds):
    # get the chunk selection labels
    if ds.chunks:
        ranges = {
            dim: tuple(split_into(ds[dim].values, ds.chunks[dim])) for dim in ds.chunks
        }

    else:
        ranges = {dim: [tuple(ds[dim].values)] for dim in ds.dims}
    lens = {key: len(val) for key, val in ranges.items()}
    order = tuple(ranges.keys())
    ranges = tuple(ranges.values())
    for selection in product(*map(range, tuple(lens.values()))):
        yield {order[i]: list(ranges[i][n]) for i, n in enumerate(selection)}


def _segy_freewriter(
    seisnc,
    segyfile,
    data_array="data",
    trace_header_map=None,
    silent=False,
    use_text=False,
    dead_trace_key=None,
    vert_dimension="twt",
    **dim_kwargs,
):
    """"""
    if trace_header_map:
        byte_fields = list(trace_header_map.values())
    else:
        byte_fields = []
    byte_fields += list(dim_kwargs.values())
    check_tracefield(byte_fields)

    try:
        coord_scalar = seisnc.coord_scalar
    except AttributeError:
        coord_scalar = 0
        seisnc.attrs["coord_scalar"] = coord_scalar
    coord_scalar_mult = np.power(abs(coord_scalar), np.sign(coord_scalar) * -1)
    seisnc.attrs["coord_scalar_mult"] = coord_scalar_mult

    # create empty trace header map if necessary
    if trace_header_map is None:
        trace_header_map = dict()
    else:  # check that values of thm in ds
        for key in trace_header_map:
            try:
                _ = seisnc[key]
            except KeyError:
                raise ValueError("keys of trace_header_map must be in seisnc")

    # remove duplicate dim_kwargs from trace_header_map
    for key in dim_kwargs:
        trace_header_map[key] = dim_kwargs[key]
    thm_vars = tuple(trace_header_map.keys())
    if dead_trace_key:
        thm_vars += (dead_trace_key,)

    dim_order = tuple(dim_kwargs.keys())

    # transfrom text if requested
    if use_text:
        text = {i + 1: line for i, line in enumerate(seisnc.text.split("\n"))}
        text = _clean_texthead(text, 80)
    else:
        text = None

    # create trace headers
    trace_layout = seisnc[data_array].isel(**{vert_dimension: 0})
    trace_headers = pd.DataFrame()
    for var in thm_vars:
        trace_headers[var] = (
            seisnc[var]
            .broadcast_like(trace_layout)
            .transpose(*dim_order, transpose_coords=True)
            .values.ravel()
        )

    # scale coords
    if CoordKeyField.cdp_x in trace_header_map:
        trace_headers[CoordKeyField.cdp_x] = (
            trace_headers[CoordKeyField.cdp_x] * seisnc.coord_scalar_mult
        )
    if CoordKeyField.cdp_y in trace_header_map:
        trace_headers[CoordKeyField.cdp_y] = (
            trace_headers[CoordKeyField.cdp_y] * seisnc.coord_scalar_mult
        )

    trace_headers = trace_headers.astype(np.int32)
    z0 = int(seisnc[vert_dimension].values[0])
    ntraces = trace_headers.shape[0]
    ns = seisnc[vert_dimension].values.size
    msys = _ISEGY_MEASUREMENT_SYSTEM[seisnc.seis.get_measurement_system()]
    spec = segyio.spec()
    spec.format = 1
    spec.iline = 181
    spec.xline = 185
    spec.samples = range(ns)

    if dead_trace_key:
        dead = ~trace_headers[dead_trace_key].astype(bool)
        spec.tracecount = dead.sum()
        live_index = trace_headers.where(dead).dropna().index
        tracen = pd.Series(
            index=live_index, data=np.arange(live_index.size), name="tracen"
        )
    else:
        spec.tracecount = ntraces
        tracen = pd.Series(
            index=trace_headers.index,
            name="tracen",
            data=np.arange(trace_headers.index.size),
        )

    # build trace headers
    trace_headers = trace_headers.astype(np.int32)
    trace_headers = trace_headers.join(
        tracen.astype(np.float32),
    )

    seisnc["tracen"] = trace_headers.set_index(list(dim_order))["tracen"].to_xarray()
    trace_headers = trace_headers.dropna(subset=["tracen"])
    trace_headers["tracen"] = trace_headers["tracen"].astype(np.int32)
    trace_headers = trace_headers.set_index("tracen")
    trace_headers = trace_headers[list(trace_header_map.keys())].rename(
        columns=trace_header_map
    )

    # header constants
    for byte, val in zip(
        [segyio.su.offset, segyio.su.ns, segyio.su.delrt, segyio.su.scalco],
        [1, ns, z0, int(seisnc.coord_scalar)],
    ):
        if byte not in trace_headers.columns:  # don't override user values
            trace_headers[byte] = val

    # write out to segy
    with segyio.create(segyfile, spec) as segyf:
        pbar = tqdm(total=spec.tracecount, disable=silent)  # , **TQDM_ARGS )
        for chk in _chunk_iterator(seisnc):
            _ = chk.pop(vert_dimension)
            # set order
            data = seisnc.sel(**chk).transpose(*dim_order, vert_dimension)
            # create missing trace mask and slicing tools
            mask = ~data.tracen.isnull().values.ravel()
            trace_locs = data.tracen.values.ravel()[mask].astype(int)
            trace_locs = tuple(
                map(
                    lambda x: slice(x[0], x[-1] + 1),
                    split_when(trace_locs, lambda x, y: y != x + 1),
                )
            )
            if data[data_array].size > 0:
                npdata = (
                    data[data_array].values.reshape(-1, ns)[mask, :].astype(np.float32)
                )
                n1 = 0
                for slc_set in trace_locs:
                    segyf.header[slc_set] = trace_headers[slc_set].to_dict(
                        orient="records"
                    )
                    n2 = n1 + slc_set.stop - slc_set.start
                    segyf.trace[slc_set] = npdata[n1:n2]
                    n1 = n2
                    pbar.update(slc_set.stop - slc_set.start)
        pbar.close()

        segyf.bin.update(
            # tsort=segyio.TraceSortingFormat.INLINE_SORTING,
            hdt=int(seisnc.sample_rate * 1000),
            hns=ns,
            mfeet=msys,
            jobid=1,
            lino=1,
            reno=1,
            ntrpr=ntraces,
            nart=ntraces,
            fold=1,
        )

    if not text:
        header_locs_text = trace_header_map_to_text(trace_header_map)

        # create text header
        overrides = {
            #     123456789012345678901234567890123456789012345678901234567890123456
            7: f"Original File: {seisnc.source_file}",
        }

        l1 = 40 - len(header_locs_text)
        overrides.update({l1 + i: line for i, line in enumerate(header_locs_text)})

        put_segy_texthead(
            segyfile,
            create_default_texthead(overrides),
        )
    else:
        put_segy_texthead(segyfile, text, line_counter=False)


def segy_freewriter(
    seisnc,
    segyfile,
    data_array="data",
    trace_header_map=None,
    use_text=False,
    dead_trace_key=None,
    vert_dimension="twt",
    chunk_spec=None,
    silent=False,
    **dim_kwargs,
):
    """Convert siesnc format (NetCDF4) to SEG-Y.

    Args:
        seisnc (xarray.Dataset, string): The input SEISNC file either a path or the in memory xarray.Dataset
        segyfile (string): The output SEG-Y file
        data_array (string): The Dataset variable name for the output volume.
        trace_header_map (dict, optional): Defaults to None. A dictionary of seisnc variables
            and byte locations. The variable will be written to the trace headers in the
            assigned byte location. By default CMP=23, cdp_x=181, cdp_y=185, iline=189,
            xline=193.
        use_text (book, optional): Use the seisnc text for the EBCIDC output. This text usally comes from
            the loaded SEG-Y file and may not match the segysak SEG-Y output. Defaults to False and writes
            the default segysak EBCIDC
        dead_trace_key (str): The key for the Dataset variable to use for trace output filter.
        vert_dimension (str): Data dimension to output, defaults to 'twt' or 'depth' whichever
            is present.
        chunk_spec (dict, optional): Xarray open_dataset chunking spec.
        silent (bool, optional): Turn off progress reporting. Defaults to False.
        dim_kwargs (int): The dimension/byte location pairs to output dimensions to. The number of dim_kwargs should be
            equal to the number of dimensions on the output data_array. The sort order will be as per the order passed
            to the function.
    """
    kwargs = dict(
        data_array=data_array,
        trace_header_map=trace_header_map,
        silent=silent,
        use_text=use_text,
        dead_trace_key=dead_trace_key,
        vert_dimension=vert_dimension,
    )

    if isinstance(seisnc, xr.Dataset):
        _segy_freewriter(
            seisnc,
            segyfile,
            **kwargs,
            **dim_kwargs,
        )
    else:
        ncfile = seisnc
        with open_seisnc(ncfile, chunks=chunk_spec) as seisnc:
            _segy_freewriter(
                seisnc,
                segyfile,
                **kwargs,
                **dim_kwargs,
            )
