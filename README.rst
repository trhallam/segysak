================================
**SEGY-SAK** - |latest-version|
================================

|build-status| |python-version| |code-style| |license| |docs|

Access the full documentation for SEGY-SAK from `Gh-pages <https://trhallam.github.io/segysak/>`__

.. image:: https://github.com/trhallam/segysak/raw/main/docs/figures/logo.png
  :alt: LOGO

*SEGY-SAK* aims to be your Python Swiss Army Knife for Seismic Data.

To do this *SEGY-SAK* offers two things; a commandline interface (CLI) for
inspecting and converting SEG-Y data to a more friendly format called
NETCDF4, and by providing convenience functions for the data using
`xarray <http://xarray.pydata.org/en/stable/>`_. This includes lazy loading and
streaming of data via the Xarray backend engine provided by SEGY-SAK.

We try hard to load the data the same way every time so your functions will
work no-matter which cube/line you load. The `xarray` conventions we use are
outlined in the documentation.

Why NETCDF4? Well, NETCDF4 is a fancy type of enclosed file binary format that
allows for faster indexing and data retrieval than SEGY. We try our best to
scan in the header information and to make it easy (or easier) to load SEGY
in different formats, different configuration (2D, 2D gathers, 3D, 3D gathers).
We do all this with the help of `segyio <https://github.com/equinor/segyio>`_
which is a lower level interface to SEGY. If you stick to our ``xarray`` format
of files we also offer utility functions to return to SEG-Y so you can export to
other software.

Current Capabilities
-----------------------

- **CLI**:

  - Convert 2D, 3D and gathers type SEG-Y to NETCDF4 and back. The NETCDF4 files
    are one line open with ``xarray.open_dataset``.

  - Extract sub-volumes via cropping operations.

  - Read the EBCIDC/text header of the seismic file.

  - Perform a limited header scan.

* **Xarray and Python API**:

  * Load SEG-Y data to a ``xarray.Dataset``, dimensions are created by analysing trace header byte
    locations provided by the user.

  * Access header information and text headers in Python with convenience
    functions.

  * Select traces by UTM X and Y coordinates.

Installation
-------------

*SEGY-SAK* can be installed by using pip. Dependencies are specified in the `pyproject.toml`.

Python Package Index via ``pip``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the command line run the ``pip`` package manager

.. code-block:: shell

   python -m pip install segysak

Install from source
^^^^^^^^^^^^^^^^^^^

Clone the SEGY-SAK Github repository and in the top level directory run

.. code-block:: shell

   python -m pip install


CLI Quick Start
-----------------

The command line interface (CLI) provides an easy tool to convert or
manipulate SEG-Y data. In your Python command-line environment it can be
accessed by calling `segysak`.

For a full list of options run

.. code-block:: shell

   segysak --help

Any SEG-Y files converted using the ``convert`` command. For example

.. code-block:: shell

   segysak convert test.segy

Can be loaded into *Python* using ``xarray``.

.. code-block:: python

   test = xarray.open_dataset('test.SEISNC')


Complete Documentation
----------------------

The complete documentation for *SEGY-SAK* can be found at
`Gh-Pages <https://trhallam.github.io/segysak/>`__

.. |latest-version| image:: https://img.shields.io/pypi/v/segysak?color=2d5016&label=pypi_version&logo=Python&logoColor=white
   :alt: Latest version on PyPi
   :target: https://pypi.org/project/segysak/

.. |build-status| image:: https://github.com/trhallam/segysak/actions/workflows/python-build-test-publish.yml/badge.svg
   :alt: Build status
   :target: https://github.com/trhallam/segysak/actions

.. |python-version| image:: https://img.shields.io/pypi/pyversions/segysak
   :alt: Python versions

.. |code-style| image:: https://img.shields.io/badge/code_style-black-000000.svg
   :alt: code style: black
   :target: https://github.com/psf/black

.. |license| image:: https://img.shields.io/badge/license-GPLv3-brightgreen
   :alt: license
   :target: https://github.com/trhallam/segysak/blob/main/LICENSE

.. |docs| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Ftrhallam.github.io%2Fsegysak%2Flatest%2Fbadge-mkdocs.json
   :target: https://trhallam.github.io/segysak/
   :alt: Documentation Status
