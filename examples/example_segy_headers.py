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
# # Working with SEG-Y headers
#
# Headers in SEG-Y data are additional meta information associated with each trace. In SEG-Y these are not pooled in a common data block but interleaved with the seismic trace data so we need to do some work to extract it. **segysak** has two helper methods for extracting information from a SEG-Y file. These are `segy_header_scan` and `segy_header_scrape`. Both of these functions return `pandas.DataFrame` objects containing header or header related information which can be used in QC, analysis and plotting.


# %% editable=true slideshow={"slide_type": ""} tags=["hide-code"]
# Disable progress bars for small examples
from segysak.progress import Progress
Progress.set_defaults(disable=True)

# %% [markdown]
# ## Scanning the headers
#
# `segy_header_scan` is primarily designed to help quickly asscertain the byte locations of key header information for loading or converting the full SEG-Y file. It does this by just looking at the first *N* traces (1000 by default) and returns the byte location and statistics related to the file.

# %%
from segysak.segy import segy_header_scan

# default just needs the file name
scan = segy_header_scan("data/volve10r12-full-twt-sub3d.sgy")
scan

# %% [markdown]
# If you want to see the full DataFrame in a notebook, use the `pandas` options context manager.
#

# %%
import pandas as pd
from IPython.display import display

with pd.option_context("display.max_rows", 91):
    display(scan)

# %% [markdown]
# Often lots of header fields don't get filled, so lets filter by the standard deviation column `std`. In fact, there are so few here we don't need the context manager. As you can see, for `segy_loader` or `segy_converter` we will need to tell those functions that the byte location for **iline** and **xline** are *189* and *193* respectively, and the byte locations for **cdp_x** and **cdp_y** are either *73* and *77* or *181* and *185* which are identical pairs.

# %%
# NIIIICCCEEEE...
scan[scan["std"] > 0]

# %% [markdown]
# ## Scraping Headers
#
# Scraping the header works like a scan but instead of statistics we get a DataFrame of actual trace header values. You can reduce the size of the scan by using the *partial_scan* keyword if required. The index of the DataFrame is the trace index and the columns are the header fields.

# %%
from segysak.segy import segy_header_scrape

scrape = segy_header_scrape("data/volve10r12-full-twt-sub3d.sgy", partial_scan=10000)
scrape

# %% [markdown]
# We know from the scan that many of these fields were empty so lets go ahead and filter our *scrape* by using the standard deviation again and passing the index which is the same as our column names.

# %%
scrape = scrape[scan[scan["std"] > 0].index]
scrape

# %% [markdown]
# We know from the scan that many of these fields were empty so lets go ahead and filter our *scrape* by using the standard deviation again and passing the index which is the same as our column names.

# %%
import matplotlib.pyplot as plt

# %matplotlib inline


# %%
plot = scrape.hist(bins=25, figsize=(20, 10))

# %% [markdown]
# We can also just plot up the geometry to check that everything looks ok, here the line numbering and coordinates seem to match up, great!

# %%
fig, axs = plt.subplots(nrows=2, figsize=(12, 10), sharex=True, sharey=True)

scrape.plot(
    kind="scatter", x="CDP_X", y="CDP_Y", c="INLINE_3D", ax=axs[0], cmap="gist_ncar"
)
scrape.plot(
    kind="scatter", x="CDP_X", y="CDP_Y", c="CROSSLINE_3D", ax=axs[1], cmap="gist_ncar"
)
for aa in axs:
    aa.set_aspect("equal", "box")
