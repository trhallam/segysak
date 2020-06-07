[![Actions Status](https://github.com/trhallam/segysak/workflows/python_build/badge.svg)](https://github.com/trhallam/segysak/actions)

# SEGY-SAK <img src="https://github.com/trhallam/segysak/blob/master/docs/figures/logo.png" alt="Logo" title="Logo" width="400" align="right" />

SEGY Swiss Army Knife for Seismic Data

SEGY-SAK is a took for manipulating SEGY format seismic data. It builds upon the work of `segyio` to provide a more
user friendly interface for manipulating and accessing information within SEG-Y data.

SEGY-SAK can be used as either a functional library within your own project to more easily access SEG-Y data or
alternatively, a command line interface (CLI) is offered that allows you to examine, manipulate and convert SEG-Y data
without writing Python scripts.

SEGY-SAK also offers a to-from converter for `xarray` and Python friendly formats such as NetCDF4 and Zarr*. These are
highly performant file formats designed for buffered and distributed processing, a necessity when it comes to large
seismic data volumes.

## Current Capabilities
 * Convert 3D SEGY to NETCDF4 and back.
    * Extract sub-volumes via cropping.

## Installation
SEGY-SAK can be installed by using python setuptools.

Clone the SEGY-SAK Github repository and in the top level directory

```
python setup.py install
```

## Quick Start

The command line interface (CLI) provides an easy tool to convert or manipulate SEG-Y data. In your Python command-line
environment it can be accessed by calling `segysak`.

See `segysak --help` for options.

SEG-Y files converted using `segysak convert test.segy` for example can be loaded into Python using `xarray`.
```
test = xarray.open_dataset('test.SEISNC')
```

## `xarray` seismic specification
The `xarray` seismic specification used by segysak to output NETCDF4 files is more performant for Python operations than
standard SEG-Y. Unlike SEG-Y, `xarray` compatable files fit neatly into the Python scientific stack providing operations
like lazy loading, easy slicing, compatability with multi-core and multi-node operations using `dask` as well as
important features such as labelled axes and coordinates.

This specification is not meant to be prescriptive but outlines some basic requirements for SEGYSAK to work.

SEGYSAK uses the convention `.SEISNC` for the suffix on NETCDF4 files it creates. These files are datasets with many 1D
and 2D variables and a single 3D variable called `data`. The data variable contains the seismic cube volume and other 
variables are used to describe the coordinate space. Attributes can be used to provide further metadata about the cube.

The basic dimensions are `i`, `j` and `k`. These are simply sequential indicies in the inline, crossline and vertical 
directions. The `iline`, `xline` and `vert` variables contain the associated coordinates. These 6 variables are mandatory.

Non-mandatory variables are `CDP_X`, `CDP_Y` and `CDP_TRACE` which can contain additional coordinate data.

Any number of attributes can be added to a SEISNC file. Currently the following attributes are extracted. 
 * `ns` Number of samples per trace.
 * `ds` Sample interval
 * `text` The ebcidc header
