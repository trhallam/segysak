.. currentmodule:: segysak

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


Loading SEG-Y data files
------------------------

.. autosummary::
   :toctree: generated/

   segysak.segy.segy_loader
   segysak.segy.well_known_byte_locs


Modifying and Creating Text Headers
-----------------------------------

.. autosummary::
   :toctree: generated/

   segysak.segy.put_segy_texthead
   segysak.segy.create_default_texthead


Writing SEG-Y files
-------------------

