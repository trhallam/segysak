Overview: Why SEGY-SAK?
========================

The objective of SEGY-SAK was to bring together the usefulness of `segyio` and
`xarray` to improve accessibility of seismic data for geoscientists using Python.
Key objectives for this project are to make loading and exporting SEG-Y easier
by offering common simple interfaces to `segyio` and to then load data into an
`xarray` format either directly into memory or by streaming large files to disk.

Currently SEGY-SAK can load 2D, 2D gathers, 3D and 3D gathers into a common format
stable format called `seisnc` that you can use faithfully in your downstream
applications. The `seisnc` format is outlined in the standard section of the
documentation and is the basis for all of the other functionality provided by
SEGY-SAK.

Beyond that, the `segysak` package has enhancements to `xarray` that reduce
complexity for common seismic related tasks to extracting information from seismic
such as arbitrary line, well-deviation or horizon extractions. We also offer a
long list of notebook examples to help new and experienced Python users alike
traverse the intricacies of Xarray and demonstrate how to do common tasks with
`segysak`.

Finally, the use of Xarray and the NetCDF4 format allows you to scale your problems
using Dask. Dask can lazily load and process data across multiple cores and even
distributed memory allowing you to apply your Python code to large seismic volumes.
This process also means you can plot slices from your volumes in Python
without loading the full dataset into memory.
