# pylint: disable=invalid-name,no-member
"""Xarray data accessor methods and functions to help with seismic analysis and
data manipulation.

"""
import xarray as xr
import numpy as np
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
