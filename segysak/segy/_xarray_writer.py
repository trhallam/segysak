from typing import Union, Tuple, Dict
import os
import pathlib
from more_itertools import split_when
from itertools import product, chain

import numpy as np
import pandas as pd
import xarray as xr

import segyio

from ._segy_core import check_tracefield

from ._segy_text import (
    create_default_texthead,
    put_segy_texthead,
    _clean_texthead,
    trace_header_map_to_text,
)

from ..progress import Progress


def chunked_index_iterator(chunks: Tuple[int], index):
    # yields a tuple pair of the origin dims chunk islice and chunk labels
    cur_index = 0
    for chunk_size in chunks:
        chk_slc = slice(cur_index, cur_index + chunk_size)
        yield chk_slc, index[cur_index : cur_index + chunk_size]
        cur_index += chunk_size


def chunk_iterator(da: xr.DataArray):
    # get the chunk islice and selection labels
    if da.chunks:
        chunks = da.chunksizes
    else:
        # autochunk
        chunks = da.chunk().chunksizes
    chunk_iterables = (
        chunked_index_iterator(chks, da[var].values) for var, chks in chunks.items()
    )
    for chunk_index in product(*chunk_iterables):
        yield {key: ci for key, ci in zip(chunks.keys(), chunk_index)}


def contiguous_iterator(ar, step=1):
    """Return contiguous indexes as slices, contiguous where step is size 1 (default)"""
    i = 0
    for split in split_when(ar, lambda x, y: y != x + step):
        yield slice(i, i + len(split)), slice(split[0], split[-1] + step)
        i += len(split)


def header_iterator(ds: xr.Dataset, iter_dim: str):
    data_vars = ds.data_vars
    for _, sub_ds in ds.groupby(iter_dim, squeeze=False):
        yield {var: sub_ds[var].item() for var in data_vars}


class SegyWriter:
    """Handles the writing of SEG-Y files from compatible Datasets. Usually this
    class would not be used and SEGY-SAK users can write SEG-Y file via the `seisio` accessor.

    !!! example
        Writing data with the `seisio` accessor.

        ```python
        ds.seisio.to_segy()
        ```

    """

    def __init__(
        self,
        segy_file: Union[str, os.PathLike],
        use_text: bool = True,
        write_dead_traces: bool = False,
        coord_scalar: Union[float, None] = None,
    ):
        self.segy_file = segy_file
        self.use_text = use_text
        self.write_dead_traces = write_dead_traces
        self.byte_fields = []
        self.shape = None
        self.coord_scalar = coord_scalar

    def _process_dimensions(
        self, da_dims: Tuple[str], vert_dimension: str, dim_kwargs: Dict[str, int]
    ) -> Tuple[Tuple[str], Dict[str, int]]:
        """Processes a list of dims against the dim_kwrags to A. check they are adequate, and B. strip any dim_kwargs that
        should be trace_header_map key:values. This might happen if the user does ds.sel(iline=100) instead of ds.sel(iline=[100]).
        Using the former will drop the dimension from the DataArray which causes problems for the writer.
        """

        # The order of the keys determines the order of the segyfile trace numbering.
        trace_order = tuple(dim for dim in dim_kwargs)  # i.e. ('iline', 'xline')
        full_dim_order = trace_order + (vert_dimension,)
        extra_dims = tuple(dim for dim in full_dim_order if dim not in da_dims)

        return trace_order, extra_dims

    def _create_byte_fields(self, *fields: Union[dict, None]):
        # add the header byte locations to a check list
        self.byte_fields = []
        for field in fields:
            if field:
                self.byte_fields += list(field.values())
        check_tracefield(self.byte_fields)

    def _create_output_trace_headers(
        self,
        da: xr.DataArray,
        trace_order: Tuple[str],
        header_vars: Dict[str, xr.DataArray],
        dead_trace_da: Union[xr.DataArray, None] = None,
    ) -> pd.DataFrame:
        """Takes the DataArray to output and optional dead_traces DataArray to
        calculate output trace numbering and coordinate mapping.

        Dead traces are numbered as -1 using a boolean dtype DataArray for an array mask.
        """
        trace_headers = xr.Dataset(coords={key: da[key] for key in trace_order})
        shape = tuple(trace_headers.sizes.values())
        ntraces = np.prod(shape)

        # full numbering
        trace_numbers = np.arange(0, ntraces, 1)

        # change numbering if dead traces
        if dead_trace_da is not None:
            assert dead_trace_da.dtype == bool
            n_dead = dead_trace_da.sum()
            trace_numbers[:] = -1
            # use the dead_trace_da as a mask to assign trace numbering (skipping dead traces)
            # squeeze is needed here to accommodate slices of data (twt length is 1)
            trace_numbers[
                ~dead_trace_da.stack({"ravel": trace_order}).values.squeeze()
            ] = np.arange(0, ntraces - n_dead, 1)

        trace_headers["tracen"] = xr.Variable(trace_order, trace_numbers.reshape(shape))

        # add header values
        for header_var, header_da in header_vars.items():
            trace_headers[header_var] = header_da.astype("int32")
        return trace_headers

    def _create_segyio_spec(self, samples: np.array, trace_count: int) -> segyio.spec:
        spec = segyio.spec()
        spec.format = 1
        # these don't really matter actually
        spec.iline = 181
        spec.xline = 185
        # this is the vertical domain of the DataArray
        spec.samples = samples
        spec.tracecount = trace_count
        return spec

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
            text: The text to write, new lines have newline character `\\n`. Long lines will be
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
        except KeyError:
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
        dead_trace_var: Union[str, None] = None,
        trace_header_map: Dict[str, int] = None,
        **dim_kwargs,
    ):
        """Output xarray dataset to SEG-Y format.

        Args:
            ds: The input dataset
            data_var: The Dataset variable name for the output volume.
            vert_dimension: Data dimension to output as vertical trace, defaults to 'samples'.
            dead_trace_var: A variable in the Dataset which signifies dead traces.
            trace_header_map: Defaults to None. A dictionary of Dataset variables
                and byte locations. The variable will be written to the trace headers in the
                assigned byte location. By default CMP=23, cdp_x=181, cdp_y=185, iline=189,
                xline=193.
            dim_kwargs: The dimension/byte location pairs to output dimensions to. The number of dim_kwargs should be
                equal to the number of dimensions on the output data_array. The sort order will be as per the order passed
                to the function.
        """
        # test for bad args
        assert data_var in ds

        # process dimension arguments
        da_dims = tuple(ds[data_var].dims)
        trace_order, extra_dims = self._process_dimensions(
            da_dims, vert_dimension, dim_kwargs
        )
        full_dim_order = trace_order + (vert_dimension,)

        # check the dims specified match the data_var
        for dim in da_dims:
            try:
                assert dim in full_dim_order
            except AssertionError:
                raise AssertionError(
                    f'{dim} not in xr.Dataset["{data_var}"] dims {ds[data_var].dims}'
                )

        # check the extra dims can be used to expand the input coords
        # this happens because dimensions get squeezed in certain indexing methods.
        for dim in extra_dims:
            try:
                assert dim in ds[data_var].coords
                ds.expand_dims(dim)
            except AssertionError:
                raise AssertionError(f"Cannot expand dims for {dim}, does not exist.")

        # create an internal expanded dataset to use from here on
        _dse = ds.expand_dims(extra_dims)

        if not trace_header_map:
            trace_header_map = dict()
        for th in trace_header_map:
            assert th in _dse

        header_vars = {
            byte: _dse[var]
            for var, byte in chain(dim_kwargs.items(), trace_header_map.items())
        }

        # combine and check header field lists
        self._create_byte_fields(trace_header_map, dim_kwargs)
        self._extract_attributes(_dse[data_var])
        #
        self.shape = _dse[data_var].shape

        # create the trace numbering and headers
        trace_headers = self._create_output_trace_headers(
            _dse[data_var],
            trace_order,
            header_vars,
            dead_trace_da=_dse[dead_trace_var] if dead_trace_var else None,
        )

        if dead_trace_var:
            n_dead_traces = np.sum(_dse["dead_traces"]).compute()
            trace_count = trace_headers["tracen"].size - n_dead_traces.item()
        else:
            trace_count = trace_headers["tracen"].size

        # get the spec of the segy file
        samples = _dse[data_var][vert_dimension].values
        spec = self._create_segyio_spec(samples, trace_count)  # the number of samples

        # add default header values if not specified by user
        for byte, val in zip(
            [segyio.su.offset, segyio.su.ns, segyio.su.delrt, segyio.su.scalco],
            [1, samples.size, samples[0], int(self.coord_scalar)],
        ):
            if byte not in list(trace_headers.var()):  # don't override user values
                trace_headers[byte] = xr.full_like(trace_headers["tracen"], val)

        trace_headers_variables = list(trace_headers.var())
        trace_headers_variables.remove("tracen")

        # calculate the number of chunks
        if _dse[data_var].chunks:
            n_chunks = np.prod(tuple(map(len, _dse[data_var].chunks)))
        else:
            n_chunks = 1

        with (
            segyio.create(self.segy_file, spec) as segyf,
            Progress(total=n_chunks, unit=" chunks", position=0) as pbar,
        ):
            segyf.bin.update(
                # tsort=segyio.TraceSortingFormat.INLINE_SORTING,
                # hdt=int(seisnc.sample_rate * 1000),
                hns=samples.size,
                # mfeet=msys,
                jobid=1,
                lino=1,
                reno=1,
                ntrpr=trace_count,
                nart=trace_count,
                fold=1,
            )

            # pre-allocate space for file 10_000 traces at a time
            pre_allocate_chunks = 10_000
            for start in range(0, trace_count, pre_allocate_chunks):
                end = start + pre_allocate_chunks
                end = end if end < trace_count else trace_count
                segyf.trace[start:end] = np.zeros(
                    (end - start, samples.size), dtype=np.float32
                )

            segyf.mmap()

            # keep trace of how many traces have been written using a segy-file
            # slice, this iterates as we iterate chunks and contiguous slices within
            # chunks (the contiguous slices may be different due to dead traces)
            for chunk_selectors in chunk_iterator(_dse[data_var]):
                # split the chunk selectors into labels for ds.sel
                # and index slices of ds.isel, this will help us keep trace of
                # where we are on a given axis when we sub-chunk the volume.
                chunk_labels = {
                    dim: labels for dim, (_, labels) in chunk_selectors.items()
                }
                chunk_slices = {
                    dim: slice for dim, (slice, _) in chunk_selectors.items()
                }

                # create the working chunk and stack the horizontal axis (not vert_dimension)
                data = (
                    _dse[data_var]
                    .sel(**chunk_labels)
                    .transpose(*trace_order, vert_dimension)
                    .stack({"ravel": trace_order})
                )
                if data.size < 1:
                    continue

                vert_chunk = chunk_labels.pop(vert_dimension)
                vert_slice = chunk_slices[vert_dimension]

                # create the work chunk headers
                head = (
                    trace_headers.sel(**chunk_labels)
                    .transpose(*trace_order)
                    .stack({"ravel": trace_order})
                )

                head_nd = head.sel(ravel=head.tracen >= 0)
                data_nd = data.sel(ravel=head.tracen >= 0)

                for chk_slice, trc_slice in contiguous_iterator(head_nd.tracen.values):
                    if vert_slice.start == 0:
                        segyf.header[trc_slice] = header_iterator(
                            head_nd.isel(ravel=chk_slice).drop_vars("tracen").compute(),
                            "ravel",
                        )
                    # read in trace_data, then modify in memory before write back out
                    # would be better to modify on disk but this isn't feasible with
                    # segyio, segfast might be better when it has a writer
                    trace_data = segyf.trace.raw[trc_slice]
                    trace_data[:, vert_slice] = (
                        data_nd.isel(ravel=chk_slice)
                        .transpose(..., vert_dimension)
                        .values
                    )
                    segyf.trace.raw[trc_slice] = trace_data

                pbar.update(1)

        # Write out a text header if available, else default header is used
        if self.use_text:
            text = _dse[data_var].attrs.get("text")
            if text is None:
                text = self._default_text_header(trace_header_map=trace_header_map)
            self.write_text_header(text, line_numbers=True)
