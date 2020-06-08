# pylint: disable=invalid-name,no-member
"""Xarray data accessor methods and functions to help with seismic analysis and
data manipulation.

"""
from collections.abc import Iterable

import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from ._keyfield import AttrKeyField, DimensionKeyField, CoordKeyField, VariableKeyField
from ._richstr import _upgrade_txt_richstr


@xr.register_dataset_accessor("seisio")
class SeisIO:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_netcdf(self, seisnc, **kwargs):
        """Output to netcdf4 with specs for seisnc.

        Args:
            seisnc (string/path-like): The output file path. Preferably with .seisnc extension.
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

        self._obj.to_netcdf(seisnc, **kwargs)


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
    for attr in AttrKeyField._member_names_:
        if attr not in ds.attrs:
            ds.attrs[AttrKeyField[attr].value] = None

    if ds.attrs["text"] is not None:
        ds.attrs["text"] = _upgrade_txt_richstr(ds.attrs["text"])

    return ds


@xr.register_dataset_accessor("seis")
class SeisGeom:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _coord_as_dimension(self, points):

        keys = ("cdp_x", "cdp_y")
        grid = np.vstack(
            [
                self._obj[key]
                .transpose("iline", "xline", transpose_coords=True)
                .values.ravel()
                for key in keys
            ]
        ).transpose()
        xlines_, ilines_ = np.meshgrid(self._obj["xline"], self._obj["iline"],)

        # these must all be the same length
        ils = griddata(grid, ilines_.ravel(), points)
        xls = griddata(grid, xlines_.ravel(), points)

        return ils, xls

    def xysel(self, cdp_x, cdp_y):
        """Select data at x and y coordinates

        Args:
            cdp_x (float/array-like)
            cdp_y (float/array-like)

        Returns:
            xarray.Dataset: At selected coordinates.
        """
        il, xl = self._coord_as_dimension((cdp_x, cdp_y))
        return self._obj.sel(iline=il, xline=xl)

    def _has_dims(self, dimension_options, invalid_dimension_options=None):
        """Check dataset has one of dimension options.
        """
        current_dims = set(self._obj.dims)

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
        """Decide if known geometry this volume is and return the appropriate enum.
        """
        current_dims = set(self._obj.dims)
        for enum in DimensionKeyField:
            if set(enum.value).issubset(current_dims):
                return True
        return False

    def is_2d(self):
        """Returns True if the dataset is 2D peformant else False
        """
        dim_options = [
            DimensionKeyField.twod_twt.value,
            DimensionKeyField.twod_depth.value,
        ]
        invalid_dim_options = [
            DimensionKeyField.twod_ps_twt.value,
            DimensionKeyField.twod_ps_depth.value,
        ]
        return self._has_dims(dim_options, invalid_dim_options)

    def is_3d(self):
        """Returns True if the dataset is 3D peformant else False
        """
        dim_options = [
            DimensionKeyField.threed_twt.value,
            DimensionKeyField.threed_depth.value,
        ]
        invalid_dim_options = [
            DimensionKeyField.threed_ps_twt.value,
            DimensionKeyField.threed_ps_depth.value,
        ]
        return self._has_dims(dim_options, invalid_dim_options)

    def is_2dgath(self):
        """Returns True if the dataset is 2D peformant and has offset or angle else False
        """
        dim_options = [
            DimensionKeyField.twod_ps_twt.value,
            DimensionKeyField.twod_ps_depth.value,
        ]
        return self._has_dims(dim_options)

    def is_3dgath(self):
        """Returns True if the dataset is 3D peformant and has offset or angle else False
        """
        dim_options = [
            DimensionKeyField.threed_ps_twt.value,
            DimensionKeyField.threed_ps_depth.value,
        ]
        return self._has_dims(dim_options)

    def _check_multi_z(self):
        if (
            CoordKeyField.depth.value in self._obj.dims
            and CoordKeyField.twt.value in self._obj.dims
        ):
            raise ValueError(
                "seisnc cannot determin domain both twt and depth dimensions found"
            )

    def is_twt(self):
        """Check if twt
        """
        self._check_multi_z()
        return True if CoordKeyField.twt.value in self._obj.dims else False

    def is_depth(self):
        """Check if twt
        """
        self._check_multi_z()
        return True if CoordKeyField.depth.value in self._obj.dims else False

    def is_empty(self):
        """Check if empty
        """
        if VariableKeyField.data.value in self._obj.variables:
            # This could be smartened up to check logic make sure dimensions of data
            # are correct.
            return False
        else:
            return True

    def zeros_like(self):
        """Create a new dataset with the same attributes and coordinates and
        dimensions but with data filled by zeros.
        """
        current_dims = {dim: val for dim, val in self._obj.dims.items()}
        dims = [dim for dim in current_dims.keys()]
        shp = [val for val in current_dims.values()]
        # TODO: Investigate copy options, might need to manually copy geometry
        # also don't want to load into memory if large large file.
        out = self._obj.copy()
        out[VariableKeyField.data.value] = (dims, np.zeros(shp))
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
            if var in self._obj.dims:
                raise ValueError(
                    "surface_from_points is designed for coordinates that "
                    "are not already Dataset Dimensions, use ... instead."
                )

        a, b = left
        if self._obj[a].dims != self._obj[b].dims:
            raise ValueError(
                f"left keys {a} and {b} should exist on the same dimensions"
            )

        if isinstance(points, pd.DataFrame):
            if len(right) != 2 and not isinstance(right, Iterable):
                raise ValueError(
                    "points is DataFrame, right must a tuple specified "
                    "with coordinate keys from column names, e.g right=('cdpx', 'cdpy')"
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
        org_dims = temp_ds[a].dims
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
