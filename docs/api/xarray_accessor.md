
!!! warning

    These methods have been depreciated in favour of a new `segysak` Xarray accessor.
    Examples in this documentation have been updated to use the new accessor or consult the
    [reference](./xarray_new.md).

Accessor modules are accessed as namespaces within the Xarray.Dataset objects
created by SEGY-SAK. When ``segysak`` is imported, all ``xarray.Dataset`` objects will
contain the ``.seis`` and ``.seisio`` namespaces.

## segysak Xarray accessor modules

::: segysak._accessor
    handler: python
    options:
      members:
        - SeisIO
      filter: ['!.*', 'to_netcdf']
      heading_level: 3
      show_root_toc_entry: false
      show_source: false

::: segysak._accessor
    handler: python
    options:
      members:
        - SeisGeom
      heading_level: 3
      show_root_toc_entry: false
      show_source: false