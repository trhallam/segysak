# pylint: disable=invalid-name,no-member
"""Xarray data accessor methods and functions to help with seismic analysis and
data manipulation.

"""
from typing import Union, Dict, Tuple, Any, Type, List
from warnings import warn
import os
from collections.abc import Iterable
from more_itertools import windowed
import json

import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, OptimizeWarning
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from ._keyfield import (
    HorDimKeyField,
    DimKeyField,
    AttrKeyField,
    DimensionKeyField,
    CoordKeyField,
    _CoordKeyField,
    VariableKeyField,
    VerticalKeyField,
    _VerticalKeyField,
)
from ._richstr import _upgrade_txt_richstr
from .tools import get_uniform_spacing, halfsample, plane


@xr.register_dataset_accessor("seisio")
class SeisIO:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_segy(
        self,
        segy_file: Union[str, os.PathLike],
        use_text: bool = True,
        write_dead_traces: bool = False,
        coord_scalar: Union[float, None] = None,
        silent: bool = False,
        data_var: str = "data",
        vert_dimension: str = "samples",
        dead_trace_var: Union[str, None] = None,
        trace_header_map: Dict[str, int] = None,
        **dim_kwargs,
    ):
        from .segy._xarray_writer import SegyWriter

        with SegyWriter(
            segy_file,
            use_text=use_text,
            write_dead_traces=write_dead_traces,
            coord_scalar=coord_scalar,
            silent=silent,
        ) as writer:
            writer.to_segy(
                self._obj,
                data_var=data_var,
                vert_dimension=vert_dimension,
                dead_trace_var=dead_trace_var,
                trace_header_map=trace_header_map,
                **dim_kwargs,
            )

    def to_netcdf(self, file_path: Union[str, os.PathLike], **kwargs):
        """Output to netcdf4 with specs for seisnc.

        Args:
            file_path: The output file path.
            **kwargs: As per xarray function to_netcdf.
        """
        # First remove all None attr.
        remove_keys = list()
        for key, val in self._obj.attrs.items():
            if val is None:
                remove_keys.append(key)
        for key in remove_keys:
            _ = self._obj.attrs.pop(key)

        # Convert richstring back to normal string
        try:
            self._obj.attrs["text"] = str(self._obj.attrs["text"])
        except:
            self._obj.attrs["text"] = ""

        kwargs["engine"] = "h5netcdf"

        self._obj.to_netcdf(file_path, **kwargs)

    def to_subsurface(self):
        """Convert seismic data to a subsurface StructuredData Object

        Raises:
            err: [description]
            NotImplementedError: [description]

        Returns:
            subsurface.structs.base_structures.StructuredData: subsurface struct
        """
        warn(
            "to_subsurface will be removed in v0.6.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from subsurface.structs.base_structures import StructuredData
        except ImportError as err:
            print("subsurface optional dependency required")
            raise err

        if self._obj.seis.is_twt():
            vdom = "twt"
        else:
            vdom = "depth"

        if self._obj.seis.is_3d():
            ds = self._obj.rename_dims({"iline": "x", "xline": "y", vdom: "z"})
            ds["z"] = ds["z"] * -1
            subsurface_struct = StructuredData(ds, "data")
        elif self._obj.seis.is_2d():
            subsurface_struct = StructuredData(self._obj, "data")
        else:
            raise NotImplementedError

        return subsurface_struct


def open_seisnc(seisnc, **kwargs):
    """Load from netcdf4 with seisnc specs.

    This all fills missing attributes required for output to other storage types.

    Args:
        seisnc (string/path-like): The input seisnc file.
        **kwargs: As per xarray function open_dataset.

    Returns:
        xarray dataset
    """
    kwargs["engine"] = "h5netcdf"
    ds = xr.open_dataset(seisnc, **kwargs)

    # Add back missing attr to remind people.
    for attr in AttrKeyField:
        if attr not in ds.attrs:
            ds.attrs[AttrKeyField[attr]] = None

    if ds.attrs["text"] is not None:
        ds.attrs["text"] = _upgrade_txt_richstr(ds.attrs["text"])

    promote_to_coord = [
        var for var in [CoordKeyField.cdp_x, CoordKeyField.cdp_y] if var in ds
    ]
    ds = ds.set_coords(promote_to_coord)

    return ds


def coordinate_df(
    ds: xr.Dataset,
    dims: Tuple[str, str],
    coords: Tuple[str, str],
    extra_vars: Union[Tuple[str], None] = None,
    linear_fillna: bool = True,
):
    """From a Dataset of 3d+ seismic, quickly create an dimensions DataFrame. `coords` are the X and Y UTM coordinate
    variable names.

    Filling blank cdp_x and cdp_y spaces is particularly useful for using the
    xarray.plot option with x=cdp_x and y=cdp_y for geographic plotting.

    Args:
        ds: The seismic dataset.
        dims: A length two tuple of the INLINE and CROSSLINE variable names in ds.
        coord: A length two tuple of the X and Y UTM coordinate variable names in ds.
        extra_vars: Any extra ds variables to include in the DataFrame.
        linear_fillna: Defaults to True. This will use a planar fitting function to fill blank cdp_x and cdp_y spaces.

    Returns:
        dataframe: The coordinate DataFrame

    """
    for dim in dims:
        assert dim in ds.dims
    iline, xline = dims

    for coord in coords:
        assert coord in ds
        for dim in dims:
            assert dim in ds[coord].dims
    cdpx, cdpy = coords

    for var in extra_vars:
        assert var in ds
        for dim in dims:
            assert dim in ds[var].dims

    req_vars = list(dims) + list(coords)
    if extra_vars is not None:
        req_atr += extra_vars

    df = ds[extra_vars].to_dataframe().reset_index()
    df_nona = df.dropna(subset=coords)

    if not linear_fillna:
        return df_nona

    # create fits to surface for target x from il/xl
    fitx = curve_fit(
        plane,
        (df_nona[iline], df_nona[xline]),
        df_nona[cdpx],
        p0=(0, 0, 0),
    )
    x_from_ix = lambda xy: plane(xy, *fitx[0])
    # create fits to surface for target y from il/xl
    fity = curve_fit(
        plane,
        (df_nona[iline], df_nona[xline]),
        df_nona[cdpy],
        p0=(0, 0, 0),
    )
    y_from_ix = lambda xy: plane(xy, *fity[0])

    filled_x = x_from_ix((df[iline], df[xline]))
    filled_y = y_from_ix((df[iline], df[xline]))

    df.loc[df[cdpx].isna(), cdpx] = filled_x[df[cdpx].isna()].astype(np.float32)
    df.loc[df[cdpy].isna(), cdpy] = filled_y[df[cdpy].isna()].astype(np.float32)

    return df


class TemplateAccessor:

    @property
    def humanbytes(self):
        """Prints Human Friendly size of Dataset to return the bytes as an
        int use ``xarray.Dataset.nbytes``

        Returns:
            str: Human readable size of dataset.

        """
        nbytes = self._obj.nbytes
        suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        while nbytes >= 1024 and i < len(suffixes) - 1:
            nbytes /= 1024.0
            i += 1
        f = f"{nbytes:.2f}".rstrip("0").rstrip(".")
        return "{} {}".format(f, suffixes[i])

    def store_attributes(self, **kwargs: Dict[str, Any]):
        """Store attributes in for seisnc. NetCDF attributes are a little limited, so we expand the capabilities by
        serializing and deserializing from JSON text.
        """
        if attrs := self.attrs:
            attrs.update(kwargs)
            seisnc_attrs_json = json.dumps(attrs)
        else:
            seisnc_attrs_json = json.dumps(kwargs)

        self._obj.attrs["seisnc"] = seisnc_attrs_json

    def __getitem__(self, key):
        """Return an attribute key"""
        try:
            return self.attrs[key]
        except KeyError:
            raise KeyError(f"{key} is not a key of the seisnc attributes.")

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        """Set an attribute i"""
        self.store_attributes(**{key: value})

    @property
    def attrs(self) -> Dict[str, Any]:
        """Return the seisnc attributes for this Xarray object"""
        seisnc_attrs_json = self._obj.attrs.get("seisnc")
        if seisnc_attrs_json is None:
            return {}
        else:
            return json.loads(seisnc_attrs_json)

    def set_dimensions(
        self,
        seisnc_dims: Dict[str, str] = None,
        seisnc_vert_dim: Union[str, None] = None,
    ):
        """Set the dimensions of the DataArray/Dataset. This is required to ensure that other
        functions in this module can interpret the DataArray correctly.

        Args:
            seisnc_dims: Pairs from segysak.CoordKeyField and dimension vars.
            seisnc_vert_dim: The variable name for the vertical dimension.
        """
        attrs = {}
        for key, value in seisnc_dims.items():
            assert key in DimKeyField
            assert value in self._obj.dims

        if seisnc_dims is not None:
            attrs[AttrKeyField.dimensions] = seisnc_dims

        if seisnc_vert_dim is not None:
            assert seisnc_vert_dim in self._obj.dims
            attrs[DimKeyField.samples] = seisnc_vert_dim

        self.store_attributes(**attrs)

    def get_dimensions(self):
        """Returns attrs['dimensions'] if available. Else use infer_dimensions."""
        dims = self.attrs.get(AttrKeyField.dimensions)
        if not dims:
            dims = self.infer_dimensions()
        return dims

    def infer_dimensions(self, ignore: List[str] = None):
        """Infer the dimensions from those available in the dataset/array. They
        should match good seisnc dimension names from CoordKeyField

        Args:
            ignore: A list of dimensions to ignore when considering mappings.
        """
        inferred_dims = {}
        unknown_dims = []
        for dim in HorDimKeyField:
            if dim in self._obj.dims:
                inferred_dims[dim] = dim
            else:
                unknown_dims.append(dim)
        return inferred_dims


@xr.register_dataarray_accessor("segysak")
class SegysakDataArrayAccessor(TemplateAccessor):
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def set_vertical_domain(self, vert_domain: _VerticalKeyField):
        """Set the vertical domain of the DataArray, usually twt or depth."""
        self.store_attributes({AttrKeyField.vert_domain: vert_domain})


@xr.register_dataset_accessor("segysak")
class SegysakDatasetAccessor(TemplateAccessor):
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def set_coords(self, seisnc_coords: Dict[Union[_CoordKeyField, str], str]):
        """Set the seisnc coordinate mapping. This maps coordinate variable names
        to known seisnc coordinates.
        """
        attrs = {}
        for key, value in seisnc_coords.items():
            assert key in CoordKeyField
            assert value in self._obj.data_vars

        self.store_attributes(**seisnc_coords)

    def get_coords(self):
        """Returns attrs['cdp_x'], attrs['cdp_y'] if exist else user infer_coords()"""
        cdp_x = self.attrs.get(CoordKeyField.cdp_x)
        cdp_y = self.attrs.get(CoordKeyField.cdp_y)

        if cdp_x is None and cdp_y is None:
            try:
                cdp_x, cdp_y = self.infer_coords()
            except ValueError:
                raise KeyError(
                    "Could not find or infer coordinates. Use set_coords() to set map coordinate name mapping for 'cdp_x' and 'cdp_y'"
                )

        return cdp_x, cdp_y

    def infer_coords(self, ignore: List[str] = None):
        """ """
        inferred_coords = {}
        unknown_coords = []
        for coord in CoordKeyField:
            if coord in self._obj:
                inferred_coords[coord] = coord
            else:
                unknown_coords.append(coord)
        return inferred_coords

    def scale_coords(self, coord_scalar: float = None):
        """Scale the coordinates using a SEG-Y coord_scalar

        The coordinate multiplier is given by:

            coord_scalar_mult = np.power(abs(coord_scalar), np.sign(coord_scalar))

        Or
        | scalar | multiplier |
        | ------ | ---------- |
        | 1000   | 1000       |
        | 100    | 100        |
        | 10     | 10         |
        | 1      | 1          |
        | 0      | 1          |
        | -1     | 1          |
        | -10    | 0.1        |
        | -100   | 0.01       |
        | -1000  | 0.001      |
        """
        if self.attrs.get(AttrKeyField.coord_scaled):
            raise UserWarning("Coordinates already scaled.")

        if coord_scalar is None:
            coord_scalar = self.attrs.get(AttrKeyField.coord_scalar, None)

        cdp_x, cdp_y = self.get_coords()

        # if no specified coord_scalar then set to 1
        if coord_scalar is None:
            coord_scalar = 1.0

        coord_scalar = float(coord_scalar)
        coord_multiplier = np.power(abs(coord_scalar), np.sign(coord_scalar))

        self._obj[cdp_x] = self._obj[cdp_x] * coord_multiplier
        self._obj[cdp_y] = self._obj[cdp_y] * coord_multiplier

        self.store_attributes(
            **{AttrKeyField.coord_scalar: coord_scalar, AttrKeyField.coord_scaled: True}
        )

    def calc_corner_points(self) -> List[Tuple[str, str]]:
        """Calculate the corner points of the 3d geometry or end points of a 2D line.

        Returns:
            corner_points: A list of cdp_x, cdp_y pairs for the corner points of the dataset.
        """
        cdp_x, cdp_y = self.get_coords()

        assert self._obj[cdp_x].dims == self._obj[cdp_y].dims
        dims = self._obj[cdp_x].dims
        assert len(dims) <= 2

        # 3d or 2d selection (polygon vs line)
        if len(dims) == 2:
            selection = [(0, 0), (0, -1), (-1, -1), (-1, 0), (0, 0)]
        else:
            selection = [(0,), (-1,)]

        corner_points = []
        for sel in selection:
            corner = self._obj.isel({d: s for d, s in zip(dims, sel)})
            corner_points.append(
                (
                    corner[cdp_x].item(),
                    corner[cdp_y].item(),
                )
            )

        return corner_points


@xr.register_dataset_accessor("seis")
class SeisGeom(TemplateAccessor):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _coord_as_dimension(self, points, drop):
        """Convert x and y points to iline and xline. If the affine transform
        cannot be found this function will use a gridding interpolation approach.

        Args:
            cdp_x (float/array-like)
            cdp_y (float/array-like)
            method (str): Same a methods for xarray.Dataset.interp

        Returns:
            xarray.Dataset: At selected coordinates.
        """
        affine = None
        keys = ("cdp_x", "cdp_y")

        try:
            affine = self.get_affine_transform().inverted()
        except:
            pass

        if affine is not None:
            ilxl = affine.transform(np.dstack(points)[0])
            ils, xls = ilxl[:, 0], ilxl[:, 1]

        else:
            # check drop keys actually on dims, might not be
            drop = set(drop).intersection(*[set(self._obj[key].sizes) for key in keys])

            grid = np.vstack(
                [
                    self._obj[key]
                    .mean(dim=drop, skipna=True)
                    .transpose("iline", "xline", transpose_coords=True)
                    .values.ravel()
                    for key in keys
                ]
            ).transpose()
            xlines_, ilines_ = np.meshgrid(
                self._obj["xline"],
                self._obj["iline"],
            )

            # these must all be the same length
            ils = np.atleast_1d(griddata(grid, ilines_.ravel(), points))
            xls = np.atleast_1d(griddata(grid, xlines_.ravel(), points))

        return ils, xls

    def xysel(self, cdp_x, cdp_y, method="nearest", sample_dim_name="cdp"):
        """Select data at x and y coordinates

        Args:
            cdp_x (float/array-like)
            cdp_y (float/array-like)
            method (str): Same as methods for xarray.Dataset.interp
            sample_dim_name (str, optional): The name to give the output sampling dimension.

        Returns:
            xarray.Dataset: At selected coordinates.
        """

        cdp_x = np.atleast_1d(cdp_x)
        cdp_y = np.atleast_1d(cdp_y)

        if self.is_twt():
            core_dims = DimensionKeyField.threed_twt
        elif self.is_depth():
            core_dims = DimensionKeyField.threed_depth
        else:
            raise AttributeError("Dataset required twt or depth coordinates.")

        dims = set(self._obj.sizes.keys())

        if self.is_3d() or self.is_3dgath():
            # get other dims
            other_dims = dims.difference(core_dims)
            il, xl = self._coord_as_dimension((cdp_x, cdp_y), other_dims)

            sampling_arrays = dict(
                iline=xr.DataArray(
                    il, dims=sample_dim_name, coords={sample_dim_name: range(il.size)}
                ),
                xline=xr.DataArray(
                    xl, dims=sample_dim_name, coords={sample_dim_name: range(xl.size)}
                ),
            )
            output_ds = self._obj.interp(**sampling_arrays, method=method)

            # # populate back to 2d
            # cdp_ds = [
            #     None
            #     if np.isnan(np.sum([i, x]))
            #     else self._obj.interp(iline=i, xline=x, method=method)
            #     for i, x in zip(il, xl)
            # ]

            # if all(map(lambda x: x is None, cdp_ds)):
            #     raise ValueError("No points intersected the dataset")

            # for ds in cdp_ds:
            #     if ds is not None:
            #         none_replace = ds.copy(deep=True)
            #         break

            # none_replace[VariableKeyField.data][:] = np.nan
            # none_replace[DimKeyField.iline] = np.nan
            # none_replace[DimKeyField.xline] = np.nan

            # for i, (point, xloc, yloc) in enumerate(zip(cdp_ds, cdp_x, cdp_y)):
            #     if point is None:
            #         point = none_replace
            #     point[CoordKeyField.cdp_x] = point[CoordKeyField.cdp_x] * 0 + xloc
            #     point[CoordKeyField.cdp_y] = point[CoordKeyField.cdp_y] * 0 + yloc
            #     point[CoordKeyField.cdp] = i + 1
            #     point.attrs = dict()
            #     cdp_ds[i] = point.copy()

        elif self.is_2d() or self.is_2dgath():
            raise NotImplementedError("Not yet implemented for 2D")
        else:
            raise AttributeError(
                "xysel not support for this volume, must be 3d, 3dgath, 2d or 2dgath"
            )

        # cdp_ds = xr.concat(cdp_ds, "cdp")
        # cdp_ds.attrs = self._obj.attrs.copy()

        # return cdp_ds
        output_ds.attrs = self._obj.attrs.copy()
        return output_ds

    def _has_dims(self, dimension_options, invalid_dimension_options=None):
        """Check dataset has one of dimension options."""
        current_dims = set(self._obj.sizes)

        # check for a match
        result = False
        for opt in dimension_options:
            if set(opt).issubset(current_dims):
                result = True
                break

        # check for things that would invalidate this e.g. this might be a subset
        if invalid_dimension_options is not None:
            for opt in invalid_dimension_options:
                if set(opt).issubset(current_dims):
                    result = False
                    break

        return result

    def _is_known_geometry(self):
        """Decide if known geometry this volume is and return the appropriate enum."""
        current_dims = set(self._obj.sizes)
        for key in DimensionKeyField:
            if set(key).issubset(current_dims):
                return True
        return False

    def is_2d(self):
        """Returns True if the dataset is 2D peformant else False"""
        dim_options = [
            DimensionKeyField.twod_twt,
            DimensionKeyField.twod_depth,
        ]
        invalid_dim_options = [
            DimensionKeyField.twod_ps_twt,
            DimensionKeyField.twod_ps_depth,
        ]
        return self._has_dims(dim_options, invalid_dim_options)

    def is_3d(self):
        """Returns True if the dataset is 3D peformant else False"""
        dim_options = [
            DimensionKeyField.threed_twt,
            DimensionKeyField.threed_depth,
            DimensionKeyField.threed_xline_twt,
            DimensionKeyField.threed_iline_twt,
            DimensionKeyField.threed_xline_depth,
            DimensionKeyField.threed_iline_depth,
        ]
        invalid_dim_options = [
            DimensionKeyField.threed_ps_twt,
            DimensionKeyField.threed_ps_depth,
        ]
        return self._has_dims(dim_options, invalid_dim_options)

    def is_2dgath(self):
        """Returns True if the dataset is 2D peformant and has offset or angle else False"""
        dim_options = [
            DimensionKeyField.twod_ps_twt,
            DimensionKeyField.twod_ps_depth,
        ]
        return self._has_dims(dim_options)

    def is_3dgath(self):
        """Returns True if the dataset is 3D peformant and has offset or angle else False"""
        dim_options = [
            DimensionKeyField.threed_ps_twt,
            DimensionKeyField.threed_ps_depth,
        ]
        return self._has_dims(dim_options)

    def _check_multi_z(self):
        if DimKeyField.depth in self._obj.sizes and DimKeyField.twt in self._obj.sizes:
            raise ValueError(
                "seisnc cannot determine domain both twt and depth dimensions found"
            )

    def is_twt(self):
        """Check if seisnc volume is in twt"""
        self._check_multi_z()
        return True if DimKeyField.twt in self._obj.sizes else False

    def is_depth(self):
        """Check if seisnc volume is in depth"""
        self._check_multi_z()
        return True if DimKeyField.depth in self._obj.sizes else False

    def is_empty(self):
        """Check if empty"""
        if VariableKeyField.data in self._obj.variables:
            # This could be smartened up to check logic make sure dimensions of data
            # are correct.
            return False
        else:
            return True

    def get_measurement_system(self):
        """Return measurement_system if present, else None"""
        if hasattr(self._obj, "measurement_system"):
            return self._obj.measurement_system
        else:
            return None

    def zeros_like(self):
        """Create a new dataset with the same attributes and coordinates and
        dimensions but with data filled by zeros.
        """
        current_dims = {dim: val for dim, val in self._obj.sizes.items()}
        dims = [dim for dim in current_dims.keys()]
        shp = [val for val in current_dims.values()]
        # TODO: Investigate copy options, might need to manually copy geometry
        # also don't want to load into memory if large large file.
        out = self._obj.copy()
        out[VariableKeyField.data] = (dims, np.zeros(shp))
        return out

    def surface_from_points(
        self,
        points,
        attr,
        left=("cdp_x", "cdp_y"),
        right=None,
        key=None,
        method="linear",
    ):
        """Sample a 2D point set with an attribute (like Z) to the seisnc
        geometry using interpolation.

        Args:
            points (array-like): Nx2 array-like of points with coordinates
                corresponding to coord1 and coord2.
            attr (array-like, str): If str points should be a DataFrame where
                attr is the column name of the attribute to be interpolated to
                the seisnc geometry. Else attr is a 1D array of length N.
            left (tuple): Length 2 tuple of coordinate dimensions to interpolate to.
            right (tuple, optional): If points is DataFrame right is a length 2
                tuple of column keys corresponding to coordinates in argument left.
            method (str): The interpolation method to use for griddata from scipy.
                Defaults to 'linear'

        Returns:
            xr.Dataset: Surface with geometry specified in left.

        """

        if len(left) != 2 and not isinstance(left, Iterable):
            raise ValueError("left must be tuple of length 2")
        for var in left:
            if var not in self._obj.variables:
                raise ValueError("left coordinate names not found in dataset")
        for var in left:
            if var in self._obj.sizes:
                raise ValueError(
                    "surface_from_points is designed for coordinates that "
                    "are not already Dataset Dimensions, use ... instead."
                )

        a, b = left
        if self._obj[a].sizes != self._obj[b].sizes:
            raise ValueError(
                f"left keys {a} and {b} should exist on the same dimensions"
            )

        if isinstance(points, pd.DataFrame):
            if len(right) != 2 and not isinstance(right, Iterable):
                raise ValueError(
                    "points is DataFrame, right must a tuple specified "
                    "with coordinate keys from column names, e.g right=('cdp_x', 'cdp_y')"
                )
            names = [attr] + list(right)
            if not set(names).issubset(points.columns):
                raise ValueError("could not find specified keys in points DataFrame")

            _points = points[list(right)].values
            _attr = points[attr].values

        else:
            points = np.asarray(points)
            attr = np.asarray(attr)
            if points.shape[1] != 2:
                raise ValueError("points must have shape Nx2")
            if attr.size != points.shape[0]:
                raise ValueError("First dimension of points must equal length of attr")
            _points = points
            _attr = attr

        temp_ds = self._obj.copy()
        org_dims = temp_ds[a].sizes
        org_shp = temp_ds[a].shape

        var = griddata(
            _points,
            _attr,
            np.dstack(
                [
                    temp_ds[a]
                    .transpose(*org_dims, transpose_coords=True)
                    .values.ravel(),
                    temp_ds[b]
                    .transpose(*org_dims, transpose_coords=True)
                    .values.ravel(),
                ]
            ),
            method=method,
        )

        out = temp_ds.drop_vars(temp_ds.var().keys())
        temp_ds.attrs.clear()
        if key is None and isinstance(attr, str):
            key = attr
        elif key is None:
            key = "data"
        out[key] = (org_dims, var.reshape(org_shp))
        return out

    # DRAFT
    # def get_survey_extents(seismic_3d):
    #     return SimpleNamespace(
    #         iline=(seismic_3d.coords["iline"].min().item(), seismic_3d.coords["iline"].max().item()),
    #         xline=(seismic_3d.coords["xline"].min().item(), seismic_3d.coords["xline"].max().item())
    #     )

    def calc_corner_points(self):
        """Calculate the corner points of the geometry or end points of a 2D line.

        This puts two properties in the seisnc attrs with the calculated il/xl and
        cdp_x and cdp_y if available.

        Attr:
            ds.attrs['corner_points']
            ds.attrs['corner_points_xy']
        """
        corner_points = False
        corner_points_xy = False

        if self.is_3d() or self.is_3dgath():
            il, xl = DimensionKeyField.cdp_3d
            ilines = self._obj[il].values
            xlines = self._obj[xl].values
            corner_points = (
                (ilines[0], xlines[0]),
                (ilines[0], xlines[-1]),
                (ilines[-1], xlines[-1]),
                (ilines[-1], xlines[0]),
            )

            cdp_x, cdp_y = CoordKeyField.cdp_x, CoordKeyField.cdp_y

            if set((cdp_x, cdp_y)).issubset(self._obj.variables):
                xs = self._obj[cdp_x].values
                ys = self._obj[cdp_y].values
                corner_points_xy = (
                    (xs[0, 0], ys[0, 0]),
                    (xs[0, -1], ys[0, -1]),
                    (xs[-1, -1], ys[-1, -1]),
                    (xs[-1, 0], ys[-1, 0]),
                )

        elif self.is_2d() or self.is_2dgath():
            cdp = DimensionKeyField.cdp_2d[0]
            cdps = self._obj[cdp].values
            corner_points = (cdps[0], cdps[-1])

            cdp_x, cdp_y = CoordKeyField.cdp_x, CoordKeyField.cdp_y

            if set((cdp_x, cdp_y)).issubset(self._obj.variables):
                xs = self._obj[cdp_x].values
                ys = self._obj[cdp_y].values
                corner_points_xy = (
                    (xs[0], ys[0]),
                    (xs[0], ys[0]),
                    (xs[-1], ys[1]),
                    (xs[-1], ys[1]),
                )

        if corner_points:
            self._obj.attrs[AttrKeyField.corner_points] = corner_points
        if corner_points_xy:
            self._obj.attrs[AttrKeyField.corner_points_xy] = corner_points_xy

    def interp_line(
        self,
        cdp_x,
        cdp_y,
        extra=None,
        bin_spacing_hint=10,
        line_method="slinear",
        xysel_method="linear",
    ):
        """Select data at x and y coordinates

        Args:
            cdp_x (float/array-like)
            cdp_y (float/array-like)
            bin_spacing_hint (number): a bin spacing to stay close to, in cdp world units. Default: 10
            line_method (string): valid values for the *kind* argument in scipy.interpolate.interp1d
            xysel_method (string): valid values for DataArray.interp

        Returns:
            xarray.Dataset: Interpolated traces along the arbitrary line
        """
        extra = dict() if extra is None else extra

        cdp_x_i, cdp_y_i, _, extra_i = get_uniform_spacing(
            cdp_x,
            cdp_y,
            bin_spacing_hint=bin_spacing_hint,
            extra=extra,
            method=line_method,
        )
        ds = self._obj.seis.xysel(cdp_x_i, cdp_y_i, method=xysel_method)

        for key in extra:
            ds[key] = (("cdp"), extra_i[key])

        return ds

    def plot_bounds(self, ax=None):
        """Plot survey bbounding box to a new or existing axis

        Args:
            ax: (optional) axis to plot to

        Returns:
            matplotlib axis used

        """
        self.calc_corner_points()
        xy = self._obj.attrs["corner_points_xy"]

        if xy is None:
            self._obj.seis.calc_corner_points()
            xy = self._obj.attrs["corner_points_xy"]

        if not ax:
            fig, ax = plt.subplots(1, 1)

        x = [x for x, _ in xy]
        y = [y for _, y in xy]
        ax.scatter(x, y)
        ax.plot(x + [x[0]], y + [y[0]], "k:")

        ax.set_xlabel("cdp x")
        ax.set_ylabel("cdp y")

        return ax

    def subsample_dims(self, **dim_kwargs):
        """Return a dictionary of subsampled dims suitable for xarray.interp.

        This tool halves

        Args:
            dim_kwargs: dimension names as keyword arguments with values of how
                many times we should divide the dimension by 2.
        """
        output = dict()
        for dim in dim_kwargs:
            ar = self._obj[dim].values
            while dim_kwargs[dim] > 0:  # only for 3.8
                ar = halfsample(ar)
                dim_kwargs[dim] = dim_kwargs[dim] - 1
            output[dim] = ar
        return output

    def fill_cdpna(self):
        """Fills NaN cdp locations by fitting known cdp x and y values to the
        local grid using a planar surface relationshipt.
        """
        coord_df = coordinate_df(self._obj)
        self._obj["cdp_x"] = (
            ("iline", "xline"),
            coord_df.cdp_x.values.reshape(self._obj.cdp_x.shape),
        )
        self._obj["cdp_y"] = (
            ("iline", "xline"),
            coord_df.cdp_y.values.reshape(self._obj.cdp_y.shape),
        )

    def grid_rotation(self):
        """Calculate the rotation of the grid using the sum under the curve method.

        (x2 − x1)(y2 + y1)

        Returns a value >0 if the rotation of points along inline is clockwise.
        """
        self.calc_corner_points()
        points = self._obj.corner_points_xy
        p2p_sum = np.sum(
            [(b[0] - a[0]) * (b[1] + a[1]) for a, b in windowed(points, 2)]
        )
        return p2p_sum

    def get_affine_transform(self):
        """Calculate the forward iline/xline -> cdp_x, cdp_y Affine transform
        for Matplotlib using corner point geometry.

        Returns:
            matplotlib.transforms.Affine2D:

        Raises:
            ValueError: If Dataset is not 3D
        """
        if not self.is_3d():
            raise ValueError()
        self.calc_corner_points()

        # direct solve for affine transform via equation substitution
        # https://cdn.sstatic.net/Sites/math/img/site-background-image.png?v=09a720444763
        # ints for iline xline will often overflow
        (x0p, y0p), (x1p, y1p), (x2p, y2p) = np.array(
            self._obj.corner_points_xy[:3], dtype=float
        )
        (x0, y0), (x1, y1), (x2, y2) = np.array(
            self._obj.corner_points[:3], dtype=float
        )

        xs = np.array([v[0] for v in self._obj.corner_points_xy])
        ys = np.array([v[0] for v in self._obj.corner_points_xy])
        if np.all(xs == xs[0]) or np.all(ys == ys[0]):
            raise ValueError(
                "The coordinates cannot be transformed, check self.seis.corner_points_xy"
            )
        a = (x1p * y0 - x2p * y0 - x0p * y1 + x2p * y1 + x0p * y2 - x1p * y2) / (
            x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2
        )
        c = (x1p * x0 - x2p * x0 - x0p * x1 + x2p * x1 + x0p * x2 - x1p * x2) / (
            -x1 * y0 + x2 * y0 + x0 * y1 - x2 * y1 - x0 * y2 + x1 * y2
        )
        b = (y1p * y0 - y2p * y0 - y0p * y1 + y2p * y1 + y0p * y2 - y1p * y2) / (
            x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2
        )
        d = (y1p * x0 - y2p * x0 - y0p * x1 + y2p * x1 + y0p * x2 - y1p * x2) / (
            -x1 * y0 + x2 * y0 + x0 * y1 - x2 * y1 - x0 * y2 + x1 * y2
        )
        e = (
            x2p * x1 * y0
            - x1p * x2 * y0
            - x2p * x0 * y1
            + x0p * x2 * y1
            + x1p * x0 * y2
            - x0p * x1 * y2
        ) / (x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2)
        f = (
            y2p * x1 * y0
            - y1p * x2 * y0
            - y2p * x0 * y1
            + y0p * x2 * y1
            + y1p * x0 * y2
            - y0p * x1 * y2
        ) / (x1 * y0 - x2 * y0 - x0 * y1 + x2 * y1 + x0 * y2 - x1 * y2)
        values = (v if ~np.isnan(v) else 0.0 for v in (a, b, c, d, e, f))
        affine_grid2loc = Affine2D.from_values(*values)
        return affine_grid2loc

    def get_dead_trace_map(self, scan=None, zeros_as_nan=False):
        """Scan the vertical axis of a volume to find traces that are all NaN
        and return an DataArray which maps the all dead traces.

        Faster scans can be performed by setting scan to an int or list of int
        representing horizontal slice indexes to use for the scan.

        Args:
            scan (int/list of int, optional): Horizontal indexes to scan.
                Defaults to None (scan full volume).
            zeros_as_nan (bool, optional): Treat zeros as NaN during scan.

        Returns:
            xarray.DataArray: boolean dead trace map.
        """

        if self.is_twt():
            vdom = DimKeyField.twt
        else:
            vdom = DimKeyField.depth

        if scan:
            nan_map = (
                self._obj.data.isel(**{vdom: scan}).isnull().reduce(np.all, dim=vdom)
            )
        else:
            nan_map = self._obj.data.isnull().reduce(np.all, dim=vdom)

        if zeros_as_nan:
            zero_map = np.abs(self._obj.data).sum(dim=vdom)
            nan_map = xr.where(zero_map == 0.0, 1, nan_map)

        return nan_map
