# SEISNC conventions for `xarray`

The `xarray` seismic specification termed `seisnc` can be used by SEGY-SAK to
output NETCDF4 files is more performant for Python operations than standard SEG-Y.
Unlike SEG-Y, `xarray` compatible files fit neatly into the Python scientific
stack providing more performant operations like lazy loading, easy slicing, 
compatibility with multi-core and multi-node operations using `dask` as well 
as important features such as labelled axes and coordinates.

This specification is not meant to be prescriptive but outlines some basic
requirements for `xarray` datasets to work effectively with *SEGY-SAK* functionality.


## 3D data dimensions

*SEGY-SAK* uses the convention labels of `iline`, `xline` to describe
the bins of 3D data. The vertical dimension is labelled `samples` and it is up to the
user to understand the domain (usually TWT or depth).
A typical `xarray` dataset created by *SEGY-SAK* will return for example

```python
>>> seisnc_3d = xr.open_dataset('test3d.sgy', dim_byte_fields={"iline":189, "xline":193})
>>> seisnc_3d.sizes
Frozen({'iline': 61, 'xline': 202, 'samples': 850})
```

Frozen standard names are supplied via the [`HorDimKeyField`](#segysak._keyfield._HorDimKeyField).

## 2D data dimensions

For 2D data SEGY-SAK uses the dimension label `cdp` for CDP locations. This allows
the package to distinguish between 2D and 3D data to allow automation on saving
and convenience wrappers. The same vertical dimensions apply as for 3D.
A typical `xarray` in 2D format would return

```python
>>> seisnc_2d = xr.open_dataset('test3d.sgy', dim_byte_fields={"cdp": 21})
>>> seisnc_2d.sizes
Frozen({'cdp': 61, 'twt': 850})
```

Frozen standard names are supplied via the [`HorDimKeyField`](#segysak._keyfield._HorDimKeyField).

## Gathers / Pre-stack data / Extra dimensions

SEGY-SAK supports an arbitrary number of extra dimensions which might be written to SEG-Y. The most
common types of data with extra dimensions are gathers also known pre-stack data. Often this data is
either exported as angle or offset dependent gathers. SEGY-SAK provides standard names for these
dimensions.

Frozen standard names are supplied via the [`DimKeyField`](#segysak._keyfield._DimKeyField).

## Map Coordinates

Map coordinates describe the physical location on earth of the seismic data. Rarely do these coordinates
align nicely with a regular grid due to rotation of the seismic bin grid relative to the map grid. Therefore it is
typically not feasible to use map coordinates for the Dataset dimensions.

Instead, X and Y map coordinates must be loading into the Dataset variables on regular dimensions such as
`iline`/`xline` or `cdp`. SEGY-SAK by convention labels the X and Y map coordinate variables `cdp_x` and `cdp_y`.
Other variable names can be used if users use the `ds.segysak.set_coords()` method.

```python
>>> seisnc_3d = xr.open_dataset(
   'test3d.sgy', dim_byte_fields={"iline":189, "xline":193}, extra_byte_fields={"cdp_x": 73, "cdp_y": 77}
)
>>> seisnc_3d.sizes
Frozen({'iline': 61, 'xline': 202, 'samples': 850})
>>> seisnc_3d.data_vars
Data variables:
    cdp_x    (iline, xline) float64 99kB 4.364e+05 4.364e+05 ... 4.341e+05
    cdp_y    (iline, xline) float64 99kB 6.477e+06 6.477e+06 ... 6.479e+06
    data     (iline, xline, samples) float32 42MB ...
```

Frozen standard names are supplied via the [`CoordKeyField`](#segysak._keyfield._CoordKeyField).

## Attributes

Xarray Datasets have an attributes dictionary available from [`ds.attrs`](https://docs.xarray.dev/en/latest/generated/xarray.Dataset.attrs.html).

This works well for simple attributes but SEGY-SAK found a need for more complex variable types not supported by standard
NetCDF4 as well as a need for a protected attribute area. To meet this need, SEGY-SAK implements it's own `attrs` as a
JSON encoded string under `ds.attrs['seisnc']`. Access to the variables in this string are via the `segysak` Dataset
accessor (an extension for Xarray) `ds.segysak.attrs`.

Any number of attributes can be added to a `siesnc` file. Currently the
following attributes are extracted or reserved for use by SEGY-SAK. Attributes marked by :cross_mark: are depreciated.

| Attribute Name       | type  | Purpose                                                                                                                                                                                                                                          |
| -------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ``ns``               | int   | number of samples per trace                                                                                                                                                                                                                      |
| ``ds``               | float | sample interval                                                                                                                                                                                                                                  |
| ``text``             | str   | ebcidc header as ascii text                                                                                                                                                                                                                      |
| ``measurement_sys``  |       | vertical units of the data                                                                                                                                                                                                                       |
| ``d3_domain``        |       | :cross_mark: vertical domain of the data                                                                                                                                                                                                         |
| ``epsg``             | str   | data epsg code                                                                                                                                                                                                                                   |
| ``corner_points``    | tuple | :cross_mark: corner points of the dataset in grid coordinates                                                                                                                                                                                    |
| ``corner_points_xy`` | tuple | :cross_mark:corner points of the dataset in xy                                                                                                                                                                                                   |
| ``source_file``      | str   | name of the file the dataset was created from                                                                                                                                                                                                    |
| ``srd``              | float | seismic reference datum of the data in vertical units ``measurement_sys`` and ``d3_domain``                                                                                                                                                      |
| ``datatype``         | str   | :cross_mark: the data type e.g. amplitude, velocity, attribute                                                                                                                                                                                   |
| ``percentiles``      | tuple | :cross_mark: this is an array of approximate percentile values created during scanning from SEG-Y. Primarily this is useful for plotting by limiting the dynamic range of the display. The percentiles are in percent 0, 0.1, 10, 50, 90, 99.9 & |
| ``coord_scalar``     | int   | the SEG-Y coordinate scalar                                                                                                                                                                                                                      |
| ``coord_scaled``     |       | A boolean that indicates if the coordinates have been scaled or not.                                                                                                                                                                             |
| ``dimensions``       |       | A mapping of `DimKeyField` and `HorKeyField` to Dataset dimensions.                                                                                                                                                                              |
| ``vert_domain``      |       | A member of ``VerticalKeyDim``.                                                                                                                                                                                                                  |

Frozen standard names are supplied via the [`AttrKeyField`](#segysak._keyfield._AttrKeyField).

## KeyFields

Keyfields represent standard names for variables, dimensions and attributes that SEGY-SAK uses internally
to access relevant data. It is helpful to stick to these conventions when working with SEGY-SAK. Alternatively
it is possible to set a mapping to your chosen names using the `set_dimensions`, and `set_coords` methods in
the [accessor module `segysak`](./api/xarray_new.md).

The frozen `dataclass` are prefixed by `_` while instantiated dataclasses for general use have no prefix.

### Frozen base classes

!!! note

   These classes are referenced here to show the standard seisnc variable names.
   Use the [instantiated classes](#instantiated-classes) if including these keyfields in your code.

::: segysak._keyfield
    handler: python
    options:
      members:
        - _HorDimKeyField
        - _VerDimKeyField
        - _DimKeyField
        - _CoordKeyField
        - _VariableKeyField
        - _AttrKeyField
        - _VerticalKeyDim
      heading_level: 4
      show_root_toc_entry: false
      show_source: true

### Instantiated classes

```python
from segysak import (
   HorDimKeyField,
   VerDimKeyField,
   DimKeyField,
   CoordKeyField,
   VariableKeyField,
   AttrKeyField,
   VerticalKeyDim,
)
```

::: segysak
    handler: python
    options:
      members:
        - HorDimKeyField
        - VerDimKeyField
        - DimKeyField
        - CoordKeyField
        - VariableKeyField
        - AttrKeyField
        - VerticalKeyDim
      heading_level: 4
      show_root_toc_entry: false
      show_source: false