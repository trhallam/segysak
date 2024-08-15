SEGY-SAK implements a methodology for chaining multiple cli commands through pipelines.
Pipelines provide a simple interface to file conversion and volume manipulations (e.g. cropping) 
in a single step process.

### Pipeline format

The pipeline format consists of a data loading command followed by any volume manipulations, and finally a data output command.
Pipelines have the general form

```shell
segysak -f <input-file> <load> [options] <manipulate> [options] ... <output> [options]
```

This form is slightly different to Unix piping (`|`) operations because the program `segysak` does not terminate between options
or use the `stdin` and `stdout` unix conventions.
Instead the number of variables associated with each command and it's options are understood by the parser and the results
of each command are passed to subsequent one. This approach means that Python objects can be kept in memory between operations
enabling the transfer of data and state.

For example, to load a 3D SEG-Y file and crop the inline dimension before output to new SEG-Y file, we could use (breaking 
commands across lines)

```console
$ segysak -f input_file.sgy \
  sgy -d iline 189 -d xline 193 \
  crop -c iline 10 20 \
  sgy -o output_file.sgy
```

!!! note
    Notice that the `-d` options for trace header locations are not required for output. This is because SEGY-SAK
    conveniently remembers where the header locations were from the input. If a SEG-Y file was not the input, the 
    trace header byte locations must be specified for output, as in the [file conversion](#file-conversion) example.
    

### File conversion

Chaining of commands allows for more flexible use of input and output operators. For example, a SEG-Y file
can be converted to NetCDF format and back again.

```console
$ segysak -f in.segy \
  sgy -d iline 189 -d xline 193 -v cdpx 181 -v cdpy 185 \
  netcdf -o out.nc

$ segysak -f out.nc \
  netcdf \
  sgy -o out.segy -d iline 189 -d xline 193 -v cdpx 181 -v cdpy 185
```

### Printing the pipeline Dataset

A print command is included to quickly check the content of NetCDF files or to check loading operations into Xarray
format.

```console
$ segysak -f in.nc netcdf print
segysak:info_ds    pipeline.ds:
<xarray.Dataset> Size: 1GB
Dimensions:  (xline: 951, iline: 651, samples: 462)
Coordinates:
  * xline    (xline) int16 2kB 300 301 302 303 304 ... 1246 1247 1248 1249 1250
  * iline    (iline) int16 1kB 100 101 102 103 104 105 ... 746 747 748 749 750
  * samples  (samples) float32 2kB 4.0 8.0 12.0 ... 1.84e+03 1.844e+03 1.848e+03
Data variables:
    data     (iline, xline, samples) float32 1GB dask.array<chunksize=(651, 951, 462), meta=np.ndarray>
    cdpx     (iline, xline) int32 2MB dask.array<chunksize=(651, 951), meta=np.ndarray>
    cdpy     (iline, xline) int32 2MB dask.array<chunksize=(651, 951), meta=np.ndarray>
Attributes:
    seisnc:   {"coord_scalar": -100.0, "coord_scaled": false}
```

### Troubleshooting

Pipeline commands can be configured to provide additional output for troubleshooting/debugging
using `--debug-level DEBUG` option.

```console
$ segysak -f in.segy --debug-level DEBUG sgy -d iline 189 -d xline 193
segysak:cli        segysak v0.5.2.dev4
segysak:cli        in.sgy
segysak:pipeline   Begin process pipeline
segysak:sgy        PARAM: {'dimension': (('iline', 189), ('xline', 193)), 'variable': (), 'output': None}
segysak:debug_ds   pipeline.ds:
--------------
<xarray.Dataset> Size: 372kB
Dimensions:  (iline: 2, xline: 100, samples: 463)
Coordinates:
  * iline    (iline) int8 2B 105 106
  * xline    (xline) int16 200B 300 301 302 303 304 305 ... 395 396 397 398 399
  * samples  (samples) float32 2kB 0.0 4.0 8.0 ... 1.84e+03 1.844e+03 1.848e+03
Data variables:
    data     (iline, xline, samples) float32 370kB ...
Attributes:
    seisnc:   {"coord_scalar": -10.0, "coord_scaled": false}
--------------
```
