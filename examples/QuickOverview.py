# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quick Overview
#
# Here you can find some quick examples of what you can do with segysak. For more details refer to the [examples](../examples.html).

# %% [markdown]
# The library is imported as *segysak* and the loaded `xarray` objects are compatible with *numpy* and *matplotlib*.
#
# The cropped volume from the Volve field in the North Sea (made available by Equinor) is used for this example, and
# all the examples and data in this documentation are available from the `examples` folder of the
# [Github](https://github.com/trhallam/segysak) respository.

# %% nbsphinx="hidden"
import warnings

warnings.filterwarnings("ignore")

# %%
import matplotlib.pyplot as plt
import pathlib

# %%
V3D_path = pathlib.Path("data/volve10r12-full-twt-sub3d.sgy")
print("3D", V3D_path, V3D_path.exists())

# %% [markdown]
# ## Scan SEG-Y headers
#

# %% [markdown]
# A basic operation would be to check the text header included in the SEG-Y file. The *get_segy_texthead*
# function accounts for common encoding issues and returns the header as a text string.

# %%
from segysak.segy import get_segy_texthead

get_segy_texthead(V3D_path)

# %% [markdown]
# If you need to investigate the trace header data more deeply, then *segy_header_scan* can be used to report
# basic statistics of each byte position for a limited number of traces.
#
# *segy_header_scan* returns a `pandas.DataFrame`. To see the full DataFrame use the `pandas` option_context manager.

# %%
from segysak.segy import segy_header_scan

scan = segy_header_scan(V3D_path)
scan

# %% [markdown]
# The header report can also be reduced by filtering blank byte locations. Here we use the standard deviation `std`
# to filter away blank values which can help us to understand the composition of the data.
#
# For instance, key values like **trace UTM coordinates** are located in bytes *73* for X & *77* for Y. We
# can also see the byte positions of the **local grid** for INLINE_3D in byte *189* and for CROSSLINE_3D in byte *193*.

# %%
scan[scan["std"] > 0]

# %% [markdown]
# To retreive the raw header content use `segy_header_scrape`. Setting `partial_scan=None` will return the
# full dataframe of trace header information.

# %%
from segysak.segy import segy_header_scrape

scrape = segy_header_scrape(V3D_path, partial_scan=1000)
scrape

# %% [markdown]
# ## Load SEG-Y data

# %% [markdown]
# All SEG-Y (2D, 2D gathers, 3D & 3D gathers) are ingested into `xarray.Dataset` objects through the
# `segy_loader` function. It is best to be explicit about the byte locations of key information but
# `segy_loader` can attempt to guess the shape of your dataset. Some standard byte positions are
# defined in the `well_known_bytes` function and others can be added via pull requests to the Github
# repository if desired.

# %%
from segysak.segy import segy_loader, well_known_byte_locs

V3D = segy_loader(V3D_path, iline=189, xline=193, cdpx=73, cdpy=77, vert_domain="TWT")
V3D

# %% [markdown]
# ## Visualising data
#
# `xarray` objects use smart label based indexing techniques to retreive subsets of data. More
# details on `xarray` techniques for *segysak* are covered in the examples, but this demonstrates
# a general syntax for selecting data by label with `xarray`. Plotting is done by `matploblib` and
# `xarray` selections can be passed to normal `matplotlib.pyplot` functions.

# %%
fig, ax1 = plt.subplots(ncols=1, figsize=(15, 8))
iline_sel = 10093
V3D.data.transpose("twt", "iline", "xline", transpose_coords=True).sel(
    iline=iline_sel
).plot(yincrease=False, cmap="seismic_r")
plt.grid("grey")
plt.ylabel("TWT")
plt.xlabel("XLINE")

# %% [markdown]
# ## Saving data to NetCDF4
#
# SEGYSAK offers a convenience utility to make saving to NetCDF4 simple. This is accesssed through the `seisio` accessor on the loaded
# SEG-Y or SEISNC volume. The `to_netcdf` method accepts the same arguments as the `xarray` version.

# %%
V3D.seisio.to_netcdf("V3D.SEISNC")

# %% [markdown]
# ## Saving data to SEG-Y
#
# To return data to SEG-Y after modification use the `segy_writer` function. `segy_writer` takes as argument a SEISNC dataset which
# requires certain attributes be set. You can also specify the byte locations to write header information.

# %%
from segysak.segy import segy_writer

segy_writer(
    V3D, "V3D.segy", trace_header_map=dict(iline=5, xline=21)
)  # Petrel Locations
