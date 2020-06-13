
#############
API reference
#############

This page provides an auto-generated summary of segysak's API. For more details
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


.. currentmodule:: segysak

SEISNC Operations
=================

Opening and Saving seisnc Files
--------------------------------

.. autosummary::
   :toctree: generated/

   open_seisnc


Creating emptry seisnc volumes
------------------------------

.. autosummary::
   :toctree: generated/

   create3d_dataset
   create2d_dataset
   create_seismic_dataset

segysak xarray accessor modules
--------------------------------

.. autosummary::
  :toctree: generated/

  SeisIO
  SeisGeom

``SeisIO``
^^^^^^^^^^^^

``xarray.Dataset.seisio`` queries

.. autosummary::
  :toctree: generated/

  SeisIO.to_netcdf

``SeisGeom``
^^^^^^^^^^^^

``xarray.Dataset.seis``

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

``seisnc`` data tools

.. autosummary::
  :toctree: generated/

   SeisGeom.zeros_like
   SeisGeom.calc_corner_points

``seisnc`` sampling operations

.. autosummary::
  :toctree: generated/

  SeisGeom.xysel
  SeisGeom.surface_from_points
