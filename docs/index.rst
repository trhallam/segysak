segysak: A library for loading and manipulating SEG-Y data with Python using **xarray**
========================================================================================

**segysak** is a ...

**segysak** leverages Xarray_ and Pandas_, to simplify handling of seismic data with
coordinates and meta data. It also offered a large number of utilities for common
seismic operations. To handle large data files it uses netCDF_ files, which were 
also the source of xarray's data model.

.. _NumPy: http://www.numpy.org
.. _Pandas: http://pandas.pydata.org
.. _dask: http://dask.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _Xarray: http://xarray.pydata.org/en/stable/


Documentation
-------------

**Getting Started**

TODO:
 - quick-overview

* :doc:`why-segysak`
* :doc:`faq`

.. * :doc:`quick-overview`

* :doc:`examples`
* :doc:`installation`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   why-segysak
   installation
   examples
   faq

..    quick-overview

**User Guide**

TODO:
  - seisnc-standard
  - terminology
  - common-tasks

* :doc:`seisnc-standard`
* :doc:`command-line-tool`

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :hidden:

   seisnc-standard
   command-line-tool
..    terminology
..    common-tasks

**Help & Reference**

TODO:
  - contributing
  - related-projects

* :doc:`api`

.. * :doc:`contributing`
.. * :doc:`related-projects`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & reference

   api

See also
--------

Fundamental Python libraries to segysak are segyio_ and xarray_.

.. _segyio: https://segyio.readthedocs.io/en/latest/
.. _xarray: http://xarray.pydata.org/en/stable/

The full Volve dataset was published by Equinor and you can read_ about it 
or get a copy of it here_.

.. _read: https://www.equinor.com/en/news/14jun2018-disclosing-volve-data.html
.. _here: https://data.equinor.com/authenticate

Get in touch
------------

Contact through swung channel



History
-------
segysak was original conceived out of a need for a better interface to SEG-Y data
in Python. The ground work was layed by Tony Hallam but development really began
during the Transform 2020 Software Underground Hackathon held online across
the world due to the cancellation of of the EAGE Annual in June of that year.
Significant contributions during the hackathon
were made by Steve Purves, Gijs Straathof and Fabio ... .

License
-------

segysak use the GPL-3 license.

The GNU General Public License is a free, copyleft license for
software and other kinds of works.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
