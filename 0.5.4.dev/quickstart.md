## Open a SEG-Y File

SEG-Y data is opened using a lazy backend with an Xarray backend engine. The limited subset of the trace headers are 
scanned and loaded into the Dataset eagerly to build the Dataset geometry but the trace data is accessed as required
to save memory. Additional trace headers can be included by specifying a name and byte position in the `extra_byte_fields`
argument.

!!! note

    Not all SEG-Y data writes trace headers to the same byte positions. Check your data to ensure you have specified the
    correct trace header byte locations for the dimensions and extra byte fields you want.

=== "3D"

    ```python
    import xarray as xr
    ds = xr.open_dataset(
        segyfile_path,
        dim_byte_fields={'iline':189, 'xline':193},
        extra_byte_fields={'cdp_x':181, 'cdp_y':185},
    )
    ds.segysak.scale_coords() # if required
    ```

=== "3D Gathers"

    ```python
    import xarray as xr
    ds = xr.open_dataset(
        segyfile_path,
        dim_byte_fields={'iline':189, 'xline':193, 'offset':37},
        extra_byte_fields={'cdp_x':181, 'cdp_y':185},
    )
    ds.segysak.scale_coords() # if required
    ```

=== "2D Line"

    ```python
    import xarray as xr
    ds = xr.open_dataset(
        segyfile_path,
        dim_byte_fields={'cdp':22},
        extra_byte_fields={'cdp_x':181, 'cdp_y':185},
    )
    ds.segysak.scale_coords() # if required
    ```

## Check the text header

The text header can be accessed directly without loading the data and often contains information about
which byte locations to specify to `open_dataset`.

```python
from segysak.segy import get_segy_texthead

get_segy_texthead(segyfile_path)
```

## Scan or extract the trace headers

The trace headers contain information about each SEG-Y file trace, including scalars, parameters and coordinates.
The information available in your trace headers depends upon the file and how or where it was originally created.

To discover what information exists in your trace headers you can either scan a few traces for a summary
or extract the entire content as a Pandas DataFrame.

=== "Scan"

    ```python
    from segysak.segy import segy_header_scan
    scan = segy_header_scan(segyfile_path)
    ```

=== "Scrape/Extract"

    ```python
    from segysak.segy import segy_header_scrape
    scrape = segy_header_scrape(segyfile_path)
    ```

## Disabling TQDM progress bars

If you wish to disable loading/progress bars for the package. This can be done globally for the session.

```python
from segysak.progress import Progress
Progress.set_defaults(disable=True)
```

## More Details

For more detailed information and examples. Check the [examples section](./examples_about.md) or
the [package reference](./api.md).