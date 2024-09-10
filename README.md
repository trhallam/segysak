# **SEGY-SAK** - [![Version](https://img.shields.io/pypi/v/segysak?color=2d5016&label=pypi_version&logo=Python&logoColor=white)](https://pypi.org/project/segysak/)

[![build-status](https://github.com/trhallam/segysak/actions/workflows/python-build-test-publish.yml/badge.svg)](https://github.com/trhallam/segysak/actions)
![python-version](https://img.shields.io/pypi/pyversions/segysak)
[![code-style](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/license-GPLv3-brightgreen)](https://github.com/trhallam/segysak/blob/main/LICENSE)
[![docs](https://img.shields.io/endpoint?url=https%3A%2F%2Ftrhallam.github.io%2Fsegysak%2Flatest%2Fbadge-mkdocs.json)](https://trhallam.github.io/segysak/)

<div align="center">
<img src="https://github.com/trhallam/segysak/raw/main/docs/figures/logo.png" alt="SEGY-SAK logo" width="400" role="img">
<br></br>
</div>

_**SEGY-SAK**_ aims to be your Python Swiss Army Knife for Seismic Data.

To do this _**SEGY-SAK**_ offers two things:
 
 - A friendly interface to SEG-Y standard files
which leverages Xarray and lazy loading to enable the access and processing of large data.
 - A command-line interface for quickly checking, interrogating and converting SEG-Y data
to other formats which support Xarray (e.g. NetCDF, Zarr, ZGY).

_**SEGY-SAK**_ has many helpful features and examples which are explained in the
full [documentation](https://trhallam.github.io/segysak/).

## Installation

*SEGY-SAK* can be installed by using pip. Dependencies are specified in the `pyproject.toml`.

### Python Package Index via ``pip``

From the command line run the ``pip`` package manager

```shell
python -m pip install segysak
```

### Install from source

Clone the SEGY-SAK Github repository and in the top level directory run

```shell
python -m pip install .
```

## CLI Quick Start

The command line interface (CLI) provides an easy tool to convert or
manipulate SEG-Y data. In your Python command-line environment it can be
accessed by calling `segysak`.

For a full list of options run

```shell
segysak --help
```

or consult the [documentation](https://trhallam.github.io/segysak/latest/cli/about.html)
for additional information and examples.
