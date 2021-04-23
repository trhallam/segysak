
#############
API reference
#############

This page provides an auto-generated summary of the ``segysak`` package API. 
For more details
and examples, refer to the relevant chapters in the main part of the
documentation.

SEG-Y Files
===========

.. currentmodule:: segysak.segy

Inspecting SEG-Y data files
---------------------------

.. autosummary::
   :toctree: generated/

   segy_header_scrape
   segy_bin_scrape
   segy_header_scan
   get_segy_texthead


Loading & Converting SEG-Y data files
-------------------------------------

.. autosummary::
   :toctree: generated/

   segy_loader
   segy_freeloader
   segy_converter
   well_known_byte_locs


Modifying and Creating Text Headers
-----------------------------------

.. autosummary::
   :toctree: generated/

   put_segy_texthead
   create_default_texthead


Writing SEG-Y files
-------------------

.. autosummary::
   :toctree: generated/

   segy_writer
   segy_freewriter
   output_byte_locs

.. currentmodule:: segysak

SEISNC Operations
=================

Opening and Saving seisnc Files
--------------------------------

.. autosummary::
   :toctree: generated/

   open_seisnc

Functions which operate on seisnc objects
-----------------------------------------

.. autosummary::
   :toctree: generated/

   coordinate_df

Creating empty seisnc volumes
------------------------------

.. autosummary::
   :toctree: generated/

   create3d_dataset
   create2d_dataset
   create_seismic_dataset

Xarray Accessor modules
=========================

Accessor modules are accessed as namespaces within the Xarray.Dataset objects
created by SEGY-SAK. When ``segysak`` is imported, all ``xarray.Dataset`` objects will
contain the ``.seis`` and ``.seisio`` namespaces.

segysak xarray accessor modules
--------------------------------

.. autosummary::
  :toctree: generated/

  SeisIO
  SeisGeom

``SeisIO``
^^^^^^^^^^^^

Access via ``xarray.Dataset.seisio``

.. autosummary::
  :toctree: generated/

  SeisIO.to_netcdf

``SeisGeom``
^^^^^^^^^^^^

Access via ``xarray.Dataset.seis``

.. autosummary::
  :toctree: generated/

  SeisGeom.is_2d
  SeisGeom.is_2dgath
  SeisGeom.is_3d
  SeisGeom.is_3dgath
  SeisGeom.is_twt
  SeisGeom.is_depth
  SeisGeom.is_empty
  SeisGeom.zeros_like
  SeisGeom.get_measurement_system
  SeisGeom.fill_cdpna

``seisnc`` data tools

.. autosummary::
  :toctree: generated/

   SeisGeom.zeros_like
   SeisGeom.calc_corner_points

``seisnc`` sampling operations

.. autosummary::
  :toctree: generated/

  SeisGeom.xysel
  SeisGeom.interp_line
  SeisGeom.surface_from_points

``seisnc`` plotting functions

.. autosummary::
  :toctree: generated/

  SeisGeom.plot_bounds
  SeisGeom.get_affine_transform

ZGY Files
==========

.. currentmodule:: segysak.openzgy

Reading and Writing from ZGY
-----------------------------

**Experimental**

.. autosummary::
   :toctree: generated/

   zgy_loader
   zgy_writer