# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quick Overview
#
# Here you can find some quick examples of what you can do with SEGY-SAK. For more details refer to the [examples](../examples.html).

# %% [markdown]
# The library is imported as *segysak* and the loaded `xarray` objects are compatible with *numpy* and *matplotlib*.
#
# The cropped volume from the Volve field in the North Sea (made available by Equinor) is used for this example, and
# all the examples and data in this documentation are available from the `examples` folder of the
# [Github](https://github.com/trhallam/segysak/examples/) respository.

# %% editable=true slideshow={"slide_type": ""}
import matplotlib.pyplot as plt
import pathlib

# %% editable=true slideshow={"slide_type": ""}
V3D_path = pathlib.Path("data/volve10r12-full-twt-sub3d.sgy")
print("3D", V3D_path, V3D_path.exists())

# %% editable=true slideshow={"slide_type": ""} tags=["hide-code"]
# Disable progress bars for small examples
from segysak.progress import Progress

Progress.set_defaults(disable=True)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Scan SEG-Y headers
#

# %% [markdown] editable=true slideshow={"slide_type": ""}
# A basic operation would be to check the text header included in the SEG-Y file. The *get_segy_texthead*
# function accounts for common encoding issues and returns the header as a text string. It is important to check the text header if you are unfamiliar with your data, it may provide important information about the contents/location of trace header values which are needed by SEGY-SAK to successfully load your data into a `xarray.Dataset`.

# %% editable=true slideshow={"slide_type": ""}
from segysak.segy import get_segy_texthead

get_segy_texthead(V3D_path)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# If you need to investigate the trace header data more deeply, then *segy_header_scan* can be used to report
# basic statistics of each byte position for a limited number of traces.
#
# *segy_header_scan* returns a `pandas.DataFrame`. To see the full DataFrame use the `pandas` option_context manager.

# %% editable=true slideshow={"slide_type": ""}
from segysak.segy import segy_header_scan
import pandas as pd

scan = segy_header_scan(V3D_path)

with pd.option_context("display.max_rows", 100, "display.max_columns", 10):
    # drop byte locations where the mean is zero, these are likely empty.
    display(scan)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The header report can also be reduced by filtering blank byte locations. Here we use the standard deviation `std`
# to filter away blank values which can help us to understand the composition of the data.
#
# For instance, key values like **trace UTM coordinates** are located in bytes *73* for X & *77* for Y. We
# can also see the byte positions of the **local grid** for INLINE_3D in byte *189* and for CROSSLINE_3D in byte *193*.

# %% editable=true slideshow={"slide_type": ""}
scan[scan["mean"] > 0]

# %% [markdown] editable=true slideshow={"slide_type": ""}
# To retreive the raw header content (values) use `segy_header_scrape`. Setting `partial_scan=None` will return the
# full dataframe of trace header information.

# %% editable=true slideshow={"slide_type": ""}
from segysak.segy import segy_header_scrape

scrape = segy_header_scrape(V3D_path, partial_scan=1000)
scrape

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load SEG-Y data

# %% [markdown] editable=true slideshow={"slide_type": ""}
# All SEG-Y (2D, 2D gathers, 3D & 3D gathers) are ingested into `xarray.Dataset` using the Xarray SEGY engine provided by
# segysak. The engine recognizes the '.segy' and `.sgy' extensions automatically when passed to `xr.open_dataset()`.
# It is important to provide the `dim_bytes_fields` and if required the `extra_bytes_fields` variables.
#
# `dim_byte_fields` specifies the byte locations which determine the orthogonal geometry of your data. This could be CDP for 2D data,
# iline and xline for 3D data, or iline, xline and offset for pre-stack gathers over 3D data. UTM coordinates are not usually valid dimensions,
# because if a survey is rotated relative to the UTM grid, you do not get orthogonal dimensions.
#
# UTM coordinates and other trace header information you require can be loaded via the `extra_bytes_fields` argument.
#
# > Note that the keys you use to label byte fields will be the keys of the dimensions and variables loaded into the dataset.
#
# > Since v0.5 of SEGY-SAK, the vertical dimension is always labelled as `samples`.

# %% editable=true slideshow={"slide_type": ""}
import xarray as xr

V3D = xr.open_dataset(
    V3D_path,
    dim_byte_fields={"iline": 189, "xline": 193},
    extra_byte_fields={"cdp_x": 73, "cdp_y": 77},
)
V3D

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Visualising data
#
# `xarray` objects use smart label based indexing techniques to retreive subsets of data. More
# details on `xarray` techniques for *segysak* are covered in the examples, but this demonstrates
# a general syntax for selecting data by label with `xarray`. Plotting is done by `matploblib` and
# `xarray` selections can be passed to normal `matplotlib.pyplot` functions.

# %% editable=true slideshow={"slide_type": ""}
fig, ax1 = plt.subplots(ncols=1, figsize=(15, 8))
iline_sel = 10093
V3D.data.transpose("samples", "iline", "xline", transpose_coords=True).sel(
    iline=iline_sel
).plot(yincrease=False, cmap="seismic_r")
plt.grid("grey")
plt.ylabel("TWT")
plt.xlabel("XLINE")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Saving data to NetCDF4
#
# SEGYSAK now loads data in a way that is compatable with the standard `to_netcdf` method of an `xarray.Dataset`. To output data to NetCDF4, simply specify the output file and Xarray should take care of the rest. If you have a particularly large SEG-Y file, that will not fit into memory, then you may need to chunk the dataset first prior to output. This will ensure lazy loading and output of data.

# %% editable=true slideshow={"slide_type": ""}
V3D.to_netcdf("data/V3D.nc")

# %% editable=true slideshow={"slide_type": ""}
# Here the seismic volume to process one INLINE at a time to the NetCDF4 file.
V3D.chunk({"iline": 1, "xline": -1, "samples": -1}).to_netcdf("data/V3D_chks.nc")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Saving data to SEG-Y
#
# To return data to SEG-Y after modification use the `seisio` accessor provided by SEGY-SAK. Unlike the `to_netcdf` method. Additional arguments are required by `to_segy` to successfuly write a SEG-Y file with appropriate header information. At a minimum, this should include byte locations for each dimension in the Dataset, but additional variables can be written to trace headers as required.

# %% editable=true slideshow={"slide_type": ""}
V3D.seisio.to_segy(
    "data/V3D-out.segy",
    trace_header_map={"cdp_x": 73, "cdp_y": 77},
    iline=189,
    xline=193,
)
