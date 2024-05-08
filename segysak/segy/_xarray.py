"""SEGY Backend for reading Seismic SEGY files."""

from typing import Dict, Tuple, Union
import os
import more_itertools

import numpy as np
import pandas as pd
import xarray as xr
from xarray import Dataset, Variable
from xarray.backends import BackendEntrypoint, BackendArray
from xarray.core import indexing

import segyio

from .._keyfield import AttrKeyField, VerticalKeyDim
from .._seismic_dataset import create_seismic_dataset


from . import (
    segy_header_scrape,
    segy_bin_scrape,
    header_as_dimensions,
    get_segy_texthead,
)
from ._segy_globals import _SEGY_MEASUREMENT_SYSTEM
from ._segy_core import sample_range, check_tracefield
from ..progress import Progress


class SgyBackendArray(BackendArray):
    def __init__(
        self,
        dnames: Tuple[str],
        shape: Tuple[int],
        lock,
        sgy_file: os.PathLike,
        trace_map: Dataset,
        segyio_kwargs: Dict = None,
    ):
        self.dnames = dnames
        self.shape = shape
        self.dtype = np.dtype(np.float32)
        self.sgy_file = str(sgy_file)
        self.lock = lock
        self.trace_map = trace_map
        self.segyio_kwargs = segyio_kwargs
        self.batch = 10_000

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:

        # xarray expects int axes to be squeezed
        squeeze_me = tuple(i for i, k in enumerate(key) if isinstance(k, int))

        # map int back to slice
        key = tuple(k if isinstance(k, slice) else slice(k, k + 1, 1) for k in key)

        # slices/ints to lengths
        out_sizes = tuple(
            len(range(*slc.indices(shp))) if isinstance(slc, slice) else 1
            for slc, shp in zip(key, self.shape)
        )

        # get the output sizes, last slice always samples
        hor_slices = key[:-1]
        samp_slice = key[-1]

        # convert the trace number map to a dataframe (i.e. flat format)
        # sort by trace number to access traces sequentially
        tracen_slice_df = (
            self.trace_map.tracen[hor_slices]
            .to_dataframe()
            .reset_index()
            .sort_values("tracen")
        )
        # put -1 in blank traces
        tracen_slice_df["tracen"] = tracen_slice_df["tracen"].fillna(-1).astype("int32")

        # calculate contiguous trace blocks from selection to enable fast selection
        # identify breaks in the trace numbering where the trace number increases
        # by more than 1
        breaks = tracen_slice_df.tracen[tracen_slice_df.tracen.diff(1).abs() > 1]
        # create a series from the breaks and add back to tracen_slice_df
        groups = pd.Series(
            name="blocks", index=breaks.index, data=range(1, len(breaks) + 1), dtype=int
        )
        tracen_slice_df = tracen_slice_df.join(groups).reset_index(drop=True)
        tracen_slice_df.loc[0, "blocks"] = 0
        tracen_slice_df = tracen_slice_df.ffill()

        # extract the data from the segyfile into a flat trace x sample format
        with (
            segyio.open(str(self.sgy_file), "r", **self.segyio_kwargs) as segyf,
            Progress(
                total=tracen_slice_df.shape[0],
                desc="Loading Traces",
                unit=" traces",
            ) as pb,
        ):
            segyf.mmap()
            volume = np.zeros(
                (tracen_slice_df.shape[0], out_sizes[-1]), dtype=self.dtype
            )
            idx = 0
            for _, traces in tracen_slice_df.groupby("blocks"):
                for batch in more_itertools.chunked(traces.tracen.values, self.batch):
                    t0 = batch[0]
                    tn = batch[-1]
                    nt = tn - t0 + 1
                    if t0 != -1:  # don't do anything for missing traces where t0 = -1
                        volume[idx : idx + nt, :] = segyf.trace.raw[t0 : tn + 1][
                            :, samp_slice
                        ]
                    idx += nt
                    pb.update(nt)

        # this creates a temporary dataset in the dimensions of a segy file
        # before using xarray to unstack the data into a block
        # this avoids complexities around missing traces in the segy data itself.
        ds = Dataset(
            coords={
                "tracen": tracen_slice_df["tracen"],
                VerticalKeyDim.samples: range(0, out_sizes[-1]),
            },
        )
        # important here to use all known dims except samples to get multi-index
        for dim in self.dnames[:-1]:
            ds[dim] = Variable(("tracen",), tracen_slice_df[dim].values)
        ds["sgy"] = Variable(("tracen", VerticalKeyDim.samples), volume)
        ds = ds.set_index(tracen=list(self.dnames[:-1]))
        data = (
            ds["sgy"]
            .unstack("tracen")
            .transpose(*self.dnames)
            .values.squeeze(axis=squeeze_me)
        )
        return data


class SgyBackendEntrypoint(BackendEntrypoint):

    def _tracefield_map(self, byte) -> str:
        return str(segyio.TraceField(byte))

    def _head_df_mapper(self, byte_fields: Dict[str, int]) -> Dict[str, str]:
        return {self._tracefield_map(byte): dim for dim, byte in byte_fields.items()}

    def _create_geometry_ds(self) -> Tuple[Tuple[str], Tuple[int], Dataset]:
        # creating dimensions and new dataset

        # create a mapper from segy_header_scrape which uses segyio column labels
        # to the users labels
        head_df_mapper = self._head_df_mapper(self.dim_byte_fields)
        head_df_mapper.update(self._head_df_mapper(self.extra_byte_fields))

        # scrape for headers if don't already exist
        if self._head_df is None:
            # Start by scraping the headers.
            self._head_df = segy_header_scrape(
                self.filename_or_obj,
                bytes_filter=(
                    list(self.dim_byte_fields.values())
                    + list(self.extra_byte_fields.values())
                    + [segyio.tracefield.TraceField.DelayRecordingTime]
                ),
                **self.segyio_kwargs,
            )
        else:
            # don't modify the external df
            self._head_df = self._head_df.copy()
        self._head_df.rename(columns=head_df_mapper, inplace=True)

        # get binary header and vertical sampling
        self._head_bin = segy_bin_scrape(self.filename_or_obj, **self.segyio_kwargs)
        nsamp = self._head_bin["Samples"]
        ns0 = self._head_df.DelayRecordingTime.min()
        sample_rate = self._head_bin["Interval"] / 1000.0
        samples = sample_range(ns0, sample_rate, nsamp)

        # calculate dimensions, the dims must be specified to create unique orthogonal
        # geometry for each trace
        dims = header_as_dimensions(self._head_df, list(self.dim_byte_fields))
        dims[VerticalKeyDim.samples] = samples
        ds = create_seismic_dataset(segysak_attr=False, **dims)

        # build a trace_index variable which maps dims to trace number in the segyfile
        self.trace_index = self._head_df[list(self.dim_byte_fields)].copy()
        for dim in self.dim_byte_fields:
            self.trace_index[dim] = self.trace_index[dim] - self.trace_index[dim].min()
        self.trace_index["tracen"] = self.trace_index.index.values
        self.trace_index.set_index(list(self.dim_byte_fields), inplace=True)
        self.trace_index = self.trace_index.to_xarray()

        # add extra byte fields
        if self.extra_byte_fields:
            dbf = list(self.dim_byte_fields)
            ebf = list(self.extra_byte_fields)
            extras = self._head_df[dbf + ebf].set_index(dbf).to_xarray()
            ds = xr.merge([ds, extras])
        else:
            ebf = []

        dnames = tuple(ds.sizes)
        shape = tuple(ds.sizes[dn] for dn in dnames)
        return dnames, shape, ds

    def _add_dataarray_attr(self, da):

        # get segy file attributes
        text = get_segy_texthead(self.filename_or_obj, **self.segyio_kwargs)
        msys = _SEGY_MEASUREMENT_SYSTEM[self._head_bin["MeasurementSystem"]]
        sample_rate = self._head_bin["Interval"] / 1000.0

        da.attrs[AttrKeyField.source_file] = self.filename_or_obj
        da.attrs[AttrKeyField.text] = text
        da.attrs[AttrKeyField.measurement_system] = msys
        da.attrs[AttrKeyField.sample_rate] = sample_rate

    def _determine_preferred_chunks(
        self, dnames: Tuple[str], dshape: Tuple[int]
    ) -> Dict[str, int]:
        # function to try and determine preferred chunks for data
        # this heuristically looks at the header dataframe and determines
        # the fast numbering dimension and its size
        dims = self._head_df[list(dnames[:-1])].astype("int32")

        # find axes where dim value repeats (i.e. the min abs diff is zero)
        diffs = dims.diff(axis=0).abs().min(axis=0)
        chunks = {
            d: 1 if s == 0.0 else dshape[i] for i, (d, s) in enumerate(diffs.items())
        }
        return chunks

    def open_dataset(
        self,
        filename_or_obj: Union[str, os.PathLike],
        dim_byte_fields: Dict[str, int],
        drop_variables: Union[tuple[str], None] = None,
        head_df: Union[pd.DataFrame, None] = None,
        extra_byte_fields: Union[Dict[str, int], None] = None,
        segyio_kwargs: Union[dict, None] = None,
    ):
        """Open a SEGY file with a native Xarray front end.

        Args:
            filename_or_obj: The SEG-Y file/path.
            dim_byte_fields: Dimension names and byte location pairs. This should at a minimum have the
                keys `cdp` or `iline` and `xline`.
            drop_variables: Ignored
            head_df: The DataFrame output from `segy_header_scrape`.
                This DataFrame can be filtered by the user
                to load select trace sets. Trace loading is based upon the DataFrame index.
            extra_byte_fields (dict, optional): Additional header information to
                load into the Dataset. Defaults to None.
            segyio_kwargs: Extra keyword arguments for segyio.open

        Returns
            Lazy segy file Dataset
        """
        self.filename_or_obj = str(filename_or_obj)
        self.dim_byte_fields = dim_byte_fields
        self._head_df = head_df
        self.extra_byte_fields = extra_byte_fields if extra_byte_fields else dict()
        self.segyio_kwargs = segyio_kwargs if segyio_kwargs else dict()
        self.segyio_kwargs.update({"ignore_geometry": True})

        check_tracefield(self.dim_byte_fields.values())
        check_tracefield(self.extra_byte_fields.values())

        dnames, dshape, ds = self._create_geometry_ds()
        encoding = {
            "preferred_chunks": self._determine_preferred_chunks(dnames, dshape)
        }

        backend_array = SgyBackendArray(
            dnames,
            dshape,
            None,
            filename_or_obj,
            self.trace_index,
            segyio_kwargs=self.segyio_kwargs,
        )
        data = indexing.LazilyIndexedArray(backend_array)
        ds["data"] = Variable(ds.dims, data, encoding=encoding)
        self._add_dataarray_attr(ds["data"])
        return ds

    def guess_can_open(self, filename_or_obj: Union[str, os.PathLike]):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".sgy", ".segy"}

    description = "Open SEGY files (.sgy, .segy) using segysak in Xarray"
    url = ""
