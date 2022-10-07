import xarray as xr
from xarray.backends import BackendEntrypoint, BackendArray
from xarray.core import indexing
import os
import pathlib
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
import segyio
import itertools

from segysak._keyfield import (
    AttrKeyField,
    VerticalKeyField,
    VariableKeyField
)

from ._segy_headers import segy_header_scrape, segy_bin_scrape
from ._segy_text import get_segy_texthead
from ._segy_globals import _SEGY_MEASUREMENT_SYSTEM

PERCENTILES = [0, 0.1, 10, 50, 90, 99.9, 100]

class SegyBackendArray(BackendArray):
    def __init__(
        self,
        shape,
        dtype,
        segyio_handle,
        tracen
    ):
        self.shape = shape
        self.dtype = dtype
        self.segyio_handle = segyio_handle
        self.tracen = tracen
        # self.lock = lock

    def __getitem__(
        self, key: indexing.ExplicitIndexer
    ) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
        shape = []
        out_shape = []
        for dim, end in zip(key, self.shape):
            if isinstance(dim, int):
                shape.append(1)
            if isinstance(dim, slice):
                start = 0 if dim.start is None else dim.start
                end = end if dim.stop is None else dim.stop
                step = 1 if dim.step is None else dim.step
                shape.append(len(
                    tuple(i for i in itertools.islice(range(end), start, end, step))
                ))
                out_shape.append(shape[-1])

        flat_length = np.product(shape[1:])

        volume = np.zeros((shape[0], flat_length), dtype=self.dtype)
        trace_indexes = self.tracen[key[1:]].flatten()
        nan_trace_mask = ~np.isnan(trace_indexes.astype(np.float64))
        
        for i, ind in zip(np.arange(flat_length)[nan_trace_mask], trace_indexes[nan_trace_mask]):
            volume[:, i] = self.segyio_handle.trace.raw[ind][key[0]]

        return volume.reshape(out_shape)

class SegyBackendEntrypoint(BackendEntrypoint):

    @staticmethod
    def _get_field_mappings(fields: Dict[str, int] | None) -> Dict:
        # When no fields are supplied, return an empty dict
        if fields is None:
            return dict()

        mapped_fields = dict()
        for field, byte_loc in fields.items():
            trace_field = str(segyio.TraceField(byte_loc))
            if trace_field == "Unknown Enum":
                raise ValueError(f"{field}:{byte_loc} was not a valid byte header")
            mapped_fields[field] = (byte_loc, trace_field)
        return mapped_fields

    def _get_head_df(self, filename_or_obj):

        bytes_filter = [
            byte for (byte, _) in self._dim_byte_fields.values()
        ] + [
            byte for (byte, _) in self._extra_byte_fields.values()
        ] + [self._delay_byte]

        # scrape headers
        head_df = segy_header_scrape(
            filename_or_obj, 
            silent=self._silent, 
            bytes_filter=bytes_filter, 
            **self._segyio_kwargs
        )

        # change names from segyio standard
        if self._dim_byte_fields:
            mapping = {trace_field:key for key, (_, trace_field) in self._dim_byte_fields.items()}
        else:
            mapping = dict()
        mapping.update({str(segyio.TraceField(self._delay_byte)):"delay"})
        head_df = head_df.rename(columns=mapping)
        
        # add a tracen
        head_df["tracen"] = head_df.index.astype("Int64")
        
        # reindex and check if dims supplied
        if self._dim_byte_fields:
            head_df.set_index([key for key in self._dim_byte_fields], inplace=True)
            if len(set(head_df.index)) != head_df.shape[0]:
                raise ValueError(
                    "The selected dimensions results in multiple traces per "
                    "dimension location, add additional dimensions or use "
                    "trace numbering to load as 2D."
                )
        else:
            # when no dims are supplied, load the file as a sequence of traces.
            head_df.set_index("tracen", inplace=True)
        return head_df

    def _get_trace_sampling(self, filename_or_obj, delay=None):
        head_bin = segy_bin_scrape(filename_or_obj, **self._segyio_kwargs)
        ns0 = delay if delay is not None else 0

        # binary header translation
        n_samples = head_bin["Samples"]
        sample_rate = head_bin["Interval"] / 1000.0
        msys = _SEGY_MEASUREMENT_SYSTEM[head_bin["MeasurementSystem"]]
        samples = np.arange(ns0, ns0 + sample_rate * n_samples, sample_rate, dtype=int)
        return dict(
            n_samples=n_samples, # number of samples
            ns0=ns0, # first sample (delay)
            sample_rate = sample_rate,
            msys = msys, # measurement system
            samples = samples
        )

    def _create_geometry_ds(self,
        head_df, sampling):
        # creating dimensions and new dataset
        ds = head_df[[]].to_xarray()
        vert_domain = sampling["vert_domain"]

        if vert_domain == "TWT":
            domain = "twt"
        elif vert_domain == "DEPTH":
            domain = "depth"
        else:
            raise ValueError(
                f"Unknown vert_domain {vert_domain}, expected one of TWT or DEPTH"
            )
        
        ds = xr.Dataset(coords={domain:sampling["samples"]}).merge(ds)
        return ds

    def _update_ds_attributes(self, filename_or_obj, ds, sampling):
        # getting attributes
        text = get_segy_texthead(filename_or_obj, **self._segyio_kwargs)
        ds.attrs[AttrKeyField.text] = text
        ds.attrs[AttrKeyField.source_file] = pathlib.Path(filename_or_obj).name
        ds.attrs[AttrKeyField.measurement_system] = sampling["msys"]
        ds.attrs[AttrKeyField.sample_rate] = sampling["sample_rate"]

    def _add_var_fields(self, ds, head_df, fields):
        # map extra byte fields into ds
        to_add = [trace_field for _, trace_field in fields.values()]
        if head_df.index.name != "tracen":
            to_add += ["tracen"]
        extras = head_df[to_add].to_xarray()
        ds.update(extras)

    def open_dataset(
        self, 
        filename_or_obj: str | os.PathLike, 
        drop_variables: tuple[str] | None = None,
        vert_domain: str = "TWT",
        delay_byte: int = 109,
        data_type: str = "AMP",
        silent: bool = False,
        dim_byte_fields: dict | None = None,
        extra_byte_fields: dict | None = None,
        head_df: pd.DataFrame | None = None,
        segyio_kwargs: dict | None = None,
    ):
        self._silent = silent
        self._segyio_kwargs = segyio_kwargs if segyio_kwargs is not None else dict()
        self._delay_byte = delay_byte

        self._dim_byte_fields = self._get_field_mappings(dim_byte_fields)
        self._extra_byte_fields = self._get_field_mappings(extra_byte_fields)

        head_df = head_df if head_df is not None else self._get_head_df(filename_or_obj)
        sampling = self._get_trace_sampling(
            filename_or_obj,
            delay=head_df.delay.min()
        )
        sampling.update({"vert_domain":vert_domain})
            
        ds = self._create_geometry_ds(head_df, sampling)
        self._update_ds_attributes(filename_or_obj, ds, sampling)
        self._add_var_fields(ds, head_df, self._extra_byte_fields)

        self._segyio_kwargs.update(dict(ignore_geometry=True))
        self._segyfile = segyio.open(filename_or_obj, "r", **self._segyio_kwargs)
        self._segyfile.mmap()

        backend_array = SegyBackendArray(
            list(ds.dims.values()),
            dtype=np.float32,
            segyio_handle=self._segyfile,
            tracen=ds.tracen.values
        )
        data = indexing.LazilyIndexedArray(backend_array)
        ds[VariableKeyField.data] = xr.Variable(ds.dims, data)
        return ds.drop("tracen", errors="ignore")

    def guess_can_open(self, filename_or_obj: str | os.PathLike):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".segy", ".sgy"}

    def __del__(self):
        self._segyfile.close()

