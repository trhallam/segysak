
Accessor modules are accessed as namespaces within the Xarray.Dataset objects
created by SEGY-SAK. When ``segysak`` is imported, all ``xarray.Dataset`` objects will
contain the ``.seis`` and ``.seisio`` namespaces.

## segysak Xarray accessor modules

::: segysak
    handler: python
    options:
      members:
        - SeisIO
        - SeisGeom
      heading_level: 1
      show_root_toc_entry: false
