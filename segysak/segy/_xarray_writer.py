from typing import Union, Tuple, Dict, List
import os
import pathlib
from more_itertools import split_into, split_when
from itertools import product

import numpy as np
import pandas as pd
import xarray as xr

import segyio

from ._segy_core import tqdm, check_tracefield

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
        ranges = {dim: [tuple(ds[dim].values)] for dim in ds.sizes}
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


class SegyWriter:

    def __init__(
        self,
        segy_file: Union[str, os.PathLike],
        use_text: bool = True,
        write_dead_traces: bool = False,
        coord_scalar: Union[float, None] = None,
        silent=False,
    ):
        self.segy_file = segy_file
        self.use_text = use_text
        self.write_dead_traces = write_dead_traces
        self.silent = silent
        self.byte_fields = []
        self.shape = None
        self.coord_scalar = coord_scalar

    def _create_byte_fields(self, *fields: Union[dict, None]):
        # add the header byte locations to a check list
        self.byte_fields = []
        for field in fields:
            if field:
                self.byte_fields += list(field.values())
        check_tracefield(self.byte_fields)

    def _default_text_header(self, trace_header_map: Dict[str, int]) -> str:
        """Create a default text header from trace header maps highlight trace
        header positions for data.

        Args:
            trace_header_map: The trace header map of coord -> byte position pairs.

        Returns:
            A text header string.
        """
        header_locs_text = trace_header_map_to_text(trace_header_map)

        # create text header
        overrides = {
            #    80 chars --->
            #    123456789012345678901234567890123456789012345678901234567890123456
            1: f"Written by SEGYSAK from Python"
        }

        # append the header_locs_text to the final lines
        l1 = 40 - len(header_locs_text)
        overrides.update({l1 + i: line for i, line in enumerate(header_locs_text)})
        text = create_default_texthead(overrides)
        text = "\n".join(text.values())

        return text

    def write_text_header(self, text: str, line_numbers: bool = True) -> None:
        """Write the text header for the file. The line length limit is 80 characters.

        Args:
            text: The text to write, new lines have newline character `\n`. Long lines will be
                truncated.
            line_numbers: Add line numbers to the output text header.
        """
        try:
            assert pathlib.Path(self.segy_file).exists()
        except AssertionError:
            raise FileNotFoundError(
                "The segy file must already exist to write the text header."
            )

        lines = {i + 1: line for i, line in enumerate(text.split("\n"))}
        clean = _clean_texthead(text, 80)
        put_segy_texthead(self.segy_file, text, line_counter=line_numbers)

    def _extract_attributes(self, da: xr.DataArray):
        try:
            self.coord_scalar = da.attrs["coord_scalar"]
        except AttributeError:
            self.coord_scalar = 0.0

        # calculates the
        self.coord_scalar_mult = np.power(
            abs(self.coord_scalar), np.sign(self.coord_scalar) * -1
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def to_segy(
        self,
        ds: xr.Dataset,
        data_var: str = "data",
        vert_dimension: str = "samples",
        trace_header_map: Dict[str, int] = None,
        chunk_spec=None,
        **dim_kwargs,
    ):
        """Output xarray dataset to SEG-Y format.

        Args:
            ds: The input dataset
            segy_file: The output SEG-Y file and path
            data_var: The Dataset variable name for the output volume.
            vert_dimension: Data dimension to output as vertical trace, defaults to 'samples'.
            trace_header_map: Defaults to None. A dictionary of Dataset variables
                and byte locations. The variable will be written to the trace headers in the
                assigned byte location. By default CMP=23, cdp_x=181, cdp_y=185, iline=189,
                xline=193.
            use_text (book, optional): Use the seisnc text for the EBCIDC output. This text usally comes from
                the loaded SEG-Y file and may not match the segysak SEG-Y output. Defaults to False and writes
                the default segysak EBCIDC
            write_dead_traces: Traces that are all nan will be counted as dead. If true, writes traces as
                value zero to file.

            chunk_spec: Xarray chunking spec, may improve writing performance or optimize lazy reads of large data.
            silent: Turn off progress reporting. Defaults to False.
            dim_kwargs: The dimension/byte location pairs to output dimensions to. The number of dim_kwargs should be
                equal to the number of dimensions on the output data_array. The sort order will be as per the order passed
                to the function.
        """
        # test for bad args
        assert data_var in ds
        for dim in dim_kwargs:
            assert dim in ds.coords

        for th in trace_header_map:
            assert th in ds

        # combine and check header field lists
        self._create_byte_fields(trace_header_map, dim_kwargs)
        self._extract_attributes(ds[data_var])
        #
        self.shape = ds[data_var].shape

        if self.use_text:
            text = ds[data_var].attrs.get("text")
            self.write_text_header(text, line_numbers=True)
