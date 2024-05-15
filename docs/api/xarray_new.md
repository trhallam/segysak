
Accessor modules are accessed as namespaces within the Xarray.Dataset objects
created by SEGY-SAK. When ``segysak`` is imported, all ``xarray.Dataset`` objects will
contain the ``.segysak`` and ``.seisio`` namespaces.

::: segysak._accessor
    handler: python
    options:
      members:
        - SeisIO
      filters: ["to_segy"]
      heading_level: 2
      show_source: false

::: segysak._accessor
    handler: python
    options:
      members:
        - SegysakDataArrayAccessor
        - SegysakDatasetAccessor
      heading_level: 2
      show_root_toc_entry: false
      show_source: false
      inherited_members: true



## Creating blank datasets

These methods help to create datasets with appropriate coordinates for seismic
data.

::: segysak
    handler: python
    options:
      members:
        - create3d_dataset
        - create2d_dataset
        - create_seismic_dataset
      heading_level: 3
      show_root_toc_entry: false

