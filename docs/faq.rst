FAQ
-------------

I have a HDF5 version conflict error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are in a conda environment this can occur when conflicts arrise from the
installed netCDF4 binaries and your system binaries. We suggest you try updating
the library with your distribution package manager. Re-creating your conda
environment or trying to reinstall the netCDF4 related packages.

How big can my input SEG-Y file be
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For practical purposes SEGY-SAK is a designed as a desktop focussed tool for files
that fit into memory. Files on the order of 10s of Gb
can be reliably loaded into memory these days.
For files greater than the amount
of memory available the `segy_convert` function should be used to convert directly
to NETCDF4. Conversion of very large files can be slow but they can then be lazily
loaded using `Xarray` and `dask`.

Why don't you use the global coordinates for dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Xarray requires that our dimensions be orthogonal to each other. Often seismic
data is rotated relative to the global cartesian grid and therefore it is not
orthogonal any more. To get around this users of seismic data regularly work with
the local seismic grid defined by the inline, crossline and vertical directions.
This local grid is linked to the global coordinate system through an affine transform.

