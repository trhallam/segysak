import xarray as xr
from xarray.backends import BackendEntrypoint, BackendArray
from xarray.core import indexing
import os
import pathlib
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np
import itertools

from openzgy.api import ZgyReader, ZgyWriter, SampleDataType, UnitDimension
import pyzgy

from segysak._keyfield import (
    AttrKeyField,
    CoordKeyField,
    VariableKeyField
)
from .._seismic_dataset import create3d_dataset

PERCENTILES = [0, 0.1, 10, 50, 90, 99.9, 100]

class ZgyBackendArray(BackendArray):
    def __init__(
        self,
        shape,
        dtype,
        zgy_file,
    ):
        self.shape = shape
        self.dtype = dtype
        self.zgy_file = zgy_file
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
        block_program = []

        for dim, end in zip(key, self.shape):
            dim_bp = []
            if isinstance(dim, int):
                shape.append(1)
                dim_bp.append((dim, slice(None, None, None)))
            else:
                start = 0 if dim.start is None else dim.start
                end = end if dim.stop is None else dim.stop
                step = 1 if dim.step is None else dim.step
                indexes = tuple(i for i in itertools.islice(range(end), start, end, step))
                shape.append(len(indexes))
                out_shape.append(shape[-1])

                if step == 1:
                    dim_bp.append((start, slice(None, None, None)))
                else:
                    for out_ind, ind in enumerate(indexes):
                        dim_bp.append(
                            (ind, slice(out_ind, out_ind+1, None))
                        )

            block_program.append(dim_bp)

        with ZgyReader(str(self.zgy_file)) as reader:
            volume = np.zeros(shape, dtype=self.dtype)
            
            # Zgy is Always 3D
            for iline, xline, vert in itertools.product(*tuple(block_program)):
                block_start = (iline[0], xline[0], vert[0])
                reader.read(block_start, volume[iline[1], xline[1], vert[1]])

        return volume.reshape(out_shape)

class ZgyBackendEntrypoint(BackendEntrypoint):

    def _create_geometry_ds(self,
        filename_or_obj):
        # creating dimensions and new dataset
        with ZgyReader(str(filename_or_obj)) as reader:
            header = reader.meta

            data_shape = header["size"]
            zstart = header["zstart"]
            zinc = header["zinc"]
            # zunit_fact = header["zunitfactor"] may need this later
            ilstep, xlstep = header["annotinc"]
            ilstart, xlstart = header["annotstart"]
            dtype = header["datatype"]

            if dtype == SampleDataType.float:
                self._dtype = np.float32
            elif dtype == SampleDataType.int16:
                self._dtype = np.int16
            elif dtype == SampleDataType.int8:
                self._dtype = np.int8
            else:
                self._dtype = np.float32

            corners = header["corners"]

            ds = create3d_dataset(
                data_shape,
                first_sample=zstart,
                sample_rate=zinc,
                iline_step=ilstep,
                xline_step=xlstep,
                first_iline=ilstart,
                first_xline=xlstart,
            )

            # load attributes
            ds.attrs[AttrKeyField.corner_points_xy] = [c for c in corners]
            ds.attrs[AttrKeyField.source_file] = str(filename_or_obj)
            ds.attrs[AttrKeyField.text] = "SEGY-SAK Created from ZGY Loader"
            ds.attrs[AttrKeyField.ns] = data_shape[2]
            ds.attrs[AttrKeyField.coord_scalar] = header["hunitfactor"]

            # create XY world coordinates.
            ds[CoordKeyField.cdp_x] = (
                (CoordKeyField.iline, CoordKeyField.xline),
                np.zeros(data_shape[0:2]),
            )
            ds[CoordKeyField.cdp_y] = ds[CoordKeyField.cdp_x].copy()

            pos = np.dstack(
                [
                    ds.iline.broadcast_like(ds[CoordKeyField.cdp_x]).values,
                    ds.xline.broadcast_like(ds[CoordKeyField.cdp_x]).values,
                ]
            )

            pos = np.apply_along_axis(reader.annotToWorld, -1, pos)

            ds[CoordKeyField.cdp_x] = (
                (CoordKeyField.iline, CoordKeyField.xline),
                pos[:, :, 0],
            )
            ds[CoordKeyField.cdp_y] = (
                (CoordKeyField.iline, CoordKeyField.xline),
                pos[:, :, 1],
            )

            ds.seis.calc_corner_points()

        return ds

    def _add_var_fields(self, ds):
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
    ):
        ds = self._create_geometry_ds(filename_or_obj)
        backend_array = ZgyBackendArray(
            list(ds.dims.values()),
            dtype=self._dtype,
            zgy_file=filename_or_obj,
        )
        data = indexing.LazilyIndexedArray(backend_array)
        ds[VariableKeyField.data] = xr.Variable(ds.dims, data)
        return ds

    def guess_can_open(self, filename_or_obj: str | os.PathLike):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".zgy"}

