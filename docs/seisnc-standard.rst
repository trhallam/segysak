SEISNC Standard for `xarray`
============================

The ``xarray`` seismic specification termed ``seisnc`` can be used by segysak to
output NETCDF4 files is more performant for Python operations than standard SEG-Y.
Unlike SEG-Y, ``xarray`` compatable files fit neatly into the Python scientific
stack providing operations like lazy loading, easy slicing, compatability with
multi-core and multi-node operations using ``dask`` as well as important features
such as labelled axes and coordinates.

This specification is not meant to be prescriptive but outlines some basic
requirements for ``xarray`` datasets to work with *SEGY-SAK* functionality.

*SEGY-SAK* uses the convention ``.seisnc`` for the suffix on NETCDF4 files it
creates. These files are datasets with specific 1D and 2D coordiates and have a
single variable called ``data``.
The ``data`` variable contains the seismic cube volume or 2D line traces.
Attributes can be used to provide further metadata about the cube.

3D and 3D Gathers
^^^^^^^^^^^^^^^^^

*SEGY-SAK* uses the convention labels of ``iline``, ``xline`` and ``offset`` to
describe
the bins of 3D data. Vertical dimensions are ``twt`` and ``depth``. A typical
``xarray`` dataset created by *SEGY-SAK* will return for example

.. code-block:: python

   >>> seisnc_3d = segysak.segy_loader('test3d.sgy', iline=189, xline=193)
   >>> seisnc_3d.dims

   Frozen(SortedKeysDict({'iline': 61, 'xline': 202, 'twt': 850}))


2D and 2D Gathers
^^^^^^^^^^^^^^^^^

For 2D data SEGY-SAK uses the dimension labels ``cdp`` and ``offset``. This allows
the package to distinguish between 2D and 3D data to allow automation on saving
and convience wrappers. The same vertical dimensions apply as for 3D.
A typical ``xarray`` in 2D format would return

.. code-block:: python

   >>> seisnc_2d = segysak.segy_loader('test2d.sgy', cdp=21)
   >>> seisnc_2d.dims

   Frozen(SortedKeysDict({'cdp': 61, 'twt': 850}))

Coordinates
^^^^^^^^^^^^

If the ``cdpx`` and ``cdpy`` byte locations are specified during loading the
SEG-Y the coordinates will be populated from the headers with the variable names
``cdp_x`` and ``cdp_y``. These will have dimensions equivalent to the horizontal
dimensions of the data (``iline``, ``xline`` for 3D and ``cdp`` for 2D).

Attributes
^^^^^^^^^^^
Any number of attributes can be added to a ``siesnc`` file. Currently the
following attributes are extracted or reserved for use by ``SEGY-SAK``.

 * ``ns`` number of samples per trace
 * ``ds`` sample interval
 * ``text`` ebcidc header as ascii text
 * ``measurement_sys`` vertical units of the data
 * ``d3_domain`` vertical domain of the data
 * ``epsg`` data epsg code
 * ``corner_points`` corner points of the dataset in grid coordinates
 * ``corner_points_xy`` corner points of the dataset in xy
 * ``source_file`` name of the file the dataset was created from
 * ``srd`` seismic reference datum of the data in vertical units ``measurement_sys``
   and ``d3_domain``
 * ``datatype`` the data type e.g. amplitude, velocity, attribute
 * ``percentiles`` this is an array of approximate percentile values created during
   scanning from SEG-Y. Primarily this is useful for plotting by limiting the dynamic
   range of the display. The percentiles are in percent 0, 0.1, 10, 50, 90, 99.9 & 100.
