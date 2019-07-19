# SEGY-SAK
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

## Installation
SEGY-SAK can be installed by using python setuptools.

Clone the SEGY-SAK Github repository and in the top level directory

```
python setup.py install
```

## Quick Start

## `xarray` seismic specification
