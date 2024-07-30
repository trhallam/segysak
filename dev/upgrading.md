SEGY-SAK underwent major redevelopment from `v0.4.x` to `v0.5`. The changes were necessary to
introduce improvements to the overall SEGY-SAK experience and to bring SEGY-SAK closer to the Xarray
ecosystem. This section offers a guide for upgrading your existing use of SEGY-SAK to the new
API and discusses some of the advantages of the new approach.

!!! tip Depreciating the old implementation

    If you have feedback or encounter issues with the new functionality of SEGY-SAK, the old functionality remains in place but
    is depreciated ([with small changes](#changes-to-the-old-api))
    for backwards compatability. The old API is problematic to maintain, and will be removed in future releases.

    Please submit any problems to the [Github Issue Tracker](https://github.com/trhallam/segysak/issues).

## Replacing `segy_loader`

The `segy_loader` method SEGY-SAK implemented was designed prior to the implementation of custom 
[backends](https://docs.xarray.dev/en/latest/internals/how-to-add-new-backend.html) for Xarray. The introduction
of this new API from Xarray offered an opportunity to significantly improve the way SEGY-SAK handled and interacted with
SEG-Y files. The new approach introduces

 - lazy loading of SEG-Y trace data. (Headers still greedy for geometry).
 - Better integration with other Xarray methods including streaming output to NetCDF, Zarr, ZGY or Xarray other supported formats.
 - Better large file support (reduced memory footprint and lazy loading of data).
 - Speed improvements for header scanning and loading.
 - Greater maintainability and usability through a geometry agnostic loading approach.
 - Support `open_mfdataset` such as in cases where gathers are split into inline files.

=== "`v0.5`"

    ```python
    import xarray as xr
    segy = xr.open_dataset(
        segy_file,
        dim_byte_fields={"iline":189, "xline":193},
        extra_byte_fields={"cdp_x":181, "cdp_y":185}
    )
    ```

=== "`v0.4.x`"

    ```python
    from segysak.segy import segy_loader
    segy = segy_loader(
        segy_file,
        iline=189, xline=193,
        cdpx=181, cdpy=185
    )
    ```

Upon loading the SEG-Y data with the new loading tool you will notice some incompatibility with the
older SEISNC standard. Read more about replacing the [`.seis`](#replacing-dsseis) accessor and the changes to the 
[SEISNC reference](#changes-to-seisnc).

Coordinate scaling is no longer applied by default to loaded SEG-Y data. The user can scale the data (if the coordinate scalar was set in the SEG-Y headers) or provide the scalar manually.

```python
segy.segysak.scale_coords()
# or 
segy.segysak.scale_coords(coord_scalar=-100)
```

## Replacing `segy_writer`

The `segy_writer` functionality has similarly been updated to better align with Xarray. The new methodology introduces

 - Native support for streamed writing via Xarray chunks. Specifically, chunking is support for all dimensions including trace samples.
 - Access via the Xarray `.seisio` accessor.
 - Dimensionality agnostic.
 - Dead trace skipping on write.

A vertical dimension is required and it's key can be specified using the `vert_dimension` keyword argument. Arbitrary key names for the vertical dimension are possible. Each orthogonal dimension of the trace data to be exported is then specified as an additional keyword argument (e.g. `iline` and `xline` in this case), with 1 or more horizontal dimensions required. Additional variables can be exported for each trace to the headers using the `trace_header_map` diction and byte location mapping.

=== "`v0.5`"

    ```python
    dataset.seisio.to_segy(
        "out.segy",
        trace_header_map={"cdp_x":73, "cdp_y":77},
        iline=189, xline=193,
        vert_dimension='samples'
    )
    ```

=== "`v0.4.x`"

    ```python
    from segysak.segy import segy_header_scrape
    segy_writer(
        dataset,
        "out.segy",
        trace_header_map=dict(iline=5, xline=21)
    )
    ```

## Replacing `ds.seisio.to_netcdf` and `open_seisnc`

Changes to the SEISNC format have negated the need for a special implementation of `to_netcdf` and improves SEGY-SAKs compatibility with other Xarray supported backends.

=== "`v0.5`"

    ```python
    dataset.to_netcdf('outfile.nc')
    ```

=== "`v0.4.x`"

    ```python
    dataset.seisio.to_netcdf('outfile.nc')
    ```

Similarly, `open_seisnc` is now depreciated. Datasets saved with the standard `ds.to_netcdf` method can now be opened using Xarray directly.

=== "`v0.5`"

    ```python
    dataset = xr.open_dataset('outfile_new.nc')
    ```

=== "`v0.4.x`"

    ```python
    from segysak import open_seisnc
    dataset = open_seisnc('outfile_old.nc')
    ```

## Replacing `ds.seis`

SEGY-SAK implemented a custom accessor namespace in `<=0.4.x` called `seis`. This has been depreciated in favour of a `segysak` namespace.
The `segysak` interface removes a lot of legacy implementation that tied SEGY-SAK uses to specific dimension names and keywords. While SEGY-SAK still supports that approach, the user has flexibility and SEGY-SAK now strives to be geometry/convention agnostic, relying on the user to understand their data. Doing this simplifies development and improves the generalisability of SEGY-SAK. 

!!! tip
    Data loaded using [SEGY-SAK conventions](./seisnc-standard.md) 
    still requires minimal input from the user on dimension and 
    coordinate names.

Major changes:

 - The `segysak` accessor is available on DataArray and Dataset objects.
 - Set and get dimension and coordinate names using `ds.segysak.set_dimensions`, `ds.segysak.get_dimensions`, `ds.segysak.set_coords` and `ds.segysak.get_coords`.
 - Store seisnc attributes on DataArray and Dataset objects. This allows multiple DataArrays to be combined (e.g. time and depth data) into a single Dataset.
 - Affine transform updated to use LSQ solver.
 - Fixes/removes bugs with [`fill_cdpna`](https://github.com/trhallam/segysak/issues/62) and [`percentiles`](https://github.com/trhallam/segysak/issues/75).
 - The arguments and returned values have changed for some accessor methods. Refer to the [reference](./api/xarray_new.md) for more details.

Example with `calc_corner_points`

=== "`v0.5`"

    ```python
    corner_points = dataset.seysak.calc_corner_points()
    ```

=== "`v0.4.x`"

    ```python
    dataset.seis.calc_corner_points()
    corner_points = dataset.attrs['corner_points_xy']
    ```

## Changes to SEISNC

SEGY-SAK no longer assumes any details about the data units or vertical dimension. As such, the vertical dimension is now loaded as `samples` by default.

Previously, SEG-Y attributes would be stored in the Xarray `Dataset.attrs` dictionary. This caused ambiguity where attributes were more
suited to a sub-array of the Dataset and created unnecessary 
complexity for storing objects that were not compatible with NetCDF 
(e.g. lists and custom attributes like the text header). To overcome
these difficulties SEGY-SAK has introduced a custom attribute to 
Xarray called `seisnc` which is stored in `Dataset.attrs['seisnc']` 
or `DataArray.attrs['seisnc']` as a JSON string. The use of JSON 
allows for the storage of more complicated objects natively in NetCDF.

SEGY-SAK accessors take care or serial/deserialisation of the data to the JSON string. Any JSON compatible object is supported (lists, dictionaries, int, float, strings). The SEISNC attributes are set and accessed using the accessor namespace.

```python
> ds.seisnc['an_attribute'] = 'my attr'
> ds.seisnc['an_attribute']
'my attr' 
> ds.seisnc.attrs
{'an_attribute':'my_attr'}
> ds.attrs
{'seisnc':'{"an_attribute":"my_attr"}'}
```

Certain keys are still reserved for the [SEISNC standard](./seisnc-standard.md). However, some will be depreciated in future versions.

## Changes to the old API

The only breaking change to the old API in `v0.5` is the removal of the silent argument from all functions. The old API made use
of `tqdm` to post information about slow loading processes. This functionality is carried forward, but control of `tqdm` is
handled by a new library class [`Progress`](./api/utils.md#segysak.progress.Progress). This all `tqdm` arguments to be set
at a session level.

For instance, to disable progress reporting for all subsequent SEGY-SAK commands.

=== "`v0.5`"

    ```python
    from segysak.progress import Progress
    from segysak.segy import segy_header_scrape
    Progress.set_defaults(disable=True)
    # silent keyword argument replaced by Progress.set_defaults(disable=True) ^^^
    scrape = segy_header_scrape("data/volve10r12-full-twt-sub3d.sgy")
    ```

=== "`v0.4.x`"

    ```python
    from segysak.segy import segy_header_scrape
    scrape = segy_header_scrape(
        "data/volve10r12-full-twt-sub3d.sgy",
        silent=True
    )
    ```

## Removing ZGY

SEGY-SAK has removed ZGY tools which have been migrated to and improved in the 
[PyZGY](https://github.com/equinor/pyzgy/tree/master) package. Please use the ZGY tools and
new Xarray backend provided by PyZGY.

=== "`v0.5`"

    Until a new release is created, please install PyZgy from Github

    ```bash
    pip install git+https://github.com/equinor/pyzgy.git
    ```

    ```python
    # open a zgy file
    zgy = xr.open_dataset('zgyfile.zgy')

    # write to a zgy file
    zgy.pyzgy.to_zgy('out_zgyfile.zgy')
    ```

=== "`v0.4.x`"

    ```python
    from segysak.openzgy import zgy_loader, zgy_writer
    ds_zgy = zgy_loader("zgyfile.zgy")

    zgy_writer(ds, "out_zgyfile.zgy")
    ```