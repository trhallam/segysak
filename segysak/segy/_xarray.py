"""SEGY Backend for reading Seismic SEGY files."""

from typing import Dict, Tuple, Union
import os

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
from ._segy_core import tqdm, sample_range, check_tracefield

TQDM_ARGS = dict(unit_scale=True, unit=" traces")


class SgyBackendArray(BackendArray):
    def __init__(
        self,
        dnames: Tuple[str],
        shape: Tuple[int],
        lock,
        sgy_file: os.PathLike,
        trace_map: Dataset,
        silent: bool = False,
        segyio_kwargs: Dict = None,
    ):
        self.dnames = dnames
        self.shape = shape
        self.dtype = np.dtype(np.float32)
        self.sgy_file = str(sgy_file)
        self.lock = lock
        self.trace_map = trace_map
        self.segyio_kwargs = segyio_kwargs
        self._silent = silent

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:

        out_sizes = self.trace_map.tracen[key[:-1]].sizes
        out_dims = tuple(out_sizes)
        out_shape = tuple(out_sizes[d] for d in out_dims)
        tracen_slice_df = (
            self.trace_map.tracen[key[:-1]]
            .to_dataframe()
            .reset_index()
            .sort_values("tracen")
        )
        # calculate contiguous trace blocks from selection
        breaks = tracen_slice_df.tracen[tracen_slice_df.tracen.diff(1) > 1]
        groups = pd.Series(
            name="blocks", index=breaks.index, data=range(1, len(breaks) + 1), dtype=int
        )
        tracen_slice_df = tracen_slice_df.join(groups).reset_index(drop=True)
        tracen_slice_df.loc[0, "blocks"] = 0
        tracen_slice_df = tracen_slice_df.ffill()

        # extract the data from the segyfile into a flat trace x sample format
        with (
            segyio.open(str(self.sgy_file), "r", **self.segyio_kwargs) as segyf,
            tqdm(
                total=tracen_slice_df.shape[0],
                desc="Loading Traces",
                disable=self._silent,
                **TQDM_ARGS,
            ) as pb,
        ):
            segyf.mmap()
            volume = np.zeros(
                (tracen_slice_df.shape[0], self.shape[-1]), dtype=self.dtype
            )
            idx = 0
            for _, traces in tracen_slice_df.groupby("blocks"):
                t0 = traces.tracen.iloc[0]
                tn = traces.tracen.iloc[-1]
                nt = traces.shape[0]

                volume[idx : idx + nt] = segyf.trace.raw[t0 : tn + 1]
                idx += nt
                pb.update(nt)

        # this creates a temporary dataset in the dimensions of a segy file
        # before using xarray to unstack the data into a block
        # this avoids complexities around missing traces in the segy data itself.
        ds = Dataset(
            coords={
                "tracen": tracen_slice_df["tracen"],
                VerticalKeyDim.samples: range(0, self.shape[-1]),
            },
            data_vars={
                dim: (("tracen"), tracen_slice_df[dim].values) for dim in out_dims
            },
        )
        ds["data"] = (("tracen", VerticalKeyDim.samples), volume)
        ds = ds.set_index(tracen=list(out_dims))
        data = (
            ds.unstack("tracen")
            .data.transpose(*(out_dims + (VerticalKeyDim.samples,)))
            .values
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
                self.filename_or_obj, silent=self._silent, **self.segyio_kwargs
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
        msys = _SEGY_MEASUREMENT_SYSTEM[self._head_bin["MeasurementSystem"]]
        samples = sample_range(ns0, sample_rate, nsamp)

        # calculate dimensions, the dims must be specified to create unique orthogonal
        # geometry for each trace
        dims = header_as_dimensions(self._head_df, list(self.dim_byte_fields))
        dims[VerticalKeyDim.samples] = samples
        ds = create_seismic_dataset(**dims)

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

        # get segy file attributes
        text = get_segy_texthead(self.filename_or_obj, **self.segyio_kwargs)

        ds.attrs[AttrKeyField.source_file] = self.filename_or_obj
        ds.attrs[AttrKeyField.text] = text
        ds.attrs[AttrKeyField.measurement_system] = msys
        ds.attrs[AttrKeyField.sample_rate] = sample_rate

        dnames = tuple(ds.sizes)
        shape = tuple(ds.sizes[dn] for dn in dnames)
        return dnames, shape, ds

    def open_dataset(
        self,
        filename_or_obj: Union[str, os.PathLike],
        dim_byte_fields: Dict[str, int],
        drop_variables: Union[tuple[str], None] = None,
        head_df: Union[pd.DataFrame, None] = None,
        silent: bool = False,
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
            silent: Turn off progress bars. Defaults to False.
            extra_byte_fields (dict, optional): Additional header information to
                load into the Dataset. Defaults to None.
            segyio_kwargs: Extra keyword arguments for segyio.open

        Returns
            Lazy segy file Dataset
        """
        self.filename_or_obj = str(filename_or_obj)
        self.dim_byte_fields = dim_byte_fields
        self._head_df = head_df
        self._silent = silent
        self.extra_byte_fields = extra_byte_fields if extra_byte_fields else dict()
        self.segyio_kwargs = segyio_kwargs if segyio_kwargs else dict()

        check_tracefield(self.dim_byte_fields.values())
        check_tracefield(self.extra_byte_fields.values())

        dnames, dshape, ds = self._create_geometry_ds()

        backend_array = SgyBackendArray(
            dnames,
            dshape,
            None,
            filename_or_obj,
            self.trace_index,
            silent=self._silent,
            segyio_kwargs=self.segyio_kwargs,
        )
        data = indexing.LazilyIndexedArray(backend_array)
        ds["data"] = Variable(ds.dims, data)
        return ds

    def guess_can_open(self, filename_or_obj: Union[str, os.PathLike]):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".sgy", ".segy"}

    description = "Open SEGY files (.sgy, .segy) using segysak in Xarray"
    url = ""
