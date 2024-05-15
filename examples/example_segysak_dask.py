# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using **dask**
#
# [dask](https://dask.org/) is a Python package built upon the scientific stack to enable scalling of Python through interactive sessions to multi-core and multi-node.
#
# Of particular relevance to **SEGY-SAK** is that `xrray.Dataset` loads naturally into `dask`.

# %% [markdown]
# ## Imports and Setup
#
# Here we import the plotting tools, `numpy` and setup the `dask.Client` which will auto start a `localcluster`. Printing the client returns details about the dashboard link and resources.

# %% nbsphinx="hidden"
import warnings

warnings.filterwarnings("ignore")

# %%
import numpy as np
from segysak import open_seisnc, segy

import matplotlib.pyplot as plt

# %matplotlib inline

# %%
from dask.distributed import Client

client = Client()
client

# %% [markdown]
# We can also scale the cluster to be a bit smaller.

# %%
client.cluster.scale(2, memory="0.5gb")
client

# %% [markdown]
# ## Lazy loading from SEISNC using chunking
#
# If your data is in SEG-Y to use dask it must be converted to SEISNC. If you do this with the CLI it only need happen once.

# %%
segy_file = "data/volve10r12-full-twt-sub3d.sgy"
seisnc_file = "data/volve10r12-full-twt-sub3d.seisnc"
segy.segy_converter(segy_file, seisnc_file, iline=189, xline=193, cdp_x=181, cdp_y=185)

# %% [markdown]
# By specifying the chunks argument to the `open_seisnc` command we can ask dask to fetch the data in chunks of size *n*. In this example the `iline` dimension will be chunked in groups of 100. The valid arguments to chunks depends on the dataset but any dimension can be used.
#
# Even though the seis of the dataset is `2.14GB` it hasn't yet been loaded into memory, not will `dask` load it entirely unless the operation demands it.

# %%
seisnc = open_seisnc("data/volve10r12-full-twt-sub3d.seisnc", chunks={"iline": 100})
seisnc.seis.humanbytes

# %% [markdown]
# Lets see what our dataset looks like. See that the variables are `dask.array`. This means they are references to the on disk data. The dimensions must be loaded so `dask` knows how to manage your dataset.

# %%
seisnc

# %% [markdown]
# ## Operations on SEISNC using `dask`
#
# In this simple example we calculate the mean, of the entire cube. If you check the dashboard (when running this example yourself). You can see the task graph and task stream execution.

# %%
mean = seisnc.data.mean()
mean

# %% [markdown]
# Whoa-oh, the mean is what? Yeah, `dask` won't calculate anything until you ask it to. This means you can string computations together into a task graph for lazy evaluation. To get the mean try this

# %%
mean.compute().values

# %% [markdown]
# ## Plotting with `dask`
#
# The lazy loading of data means we can plot what we want using `xarray` style slicing and `dask` will fetch only the data we need.

# %% tags=[]
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

iline = seisnc.sel(iline=10100).transpose("twt", "xline").data
xline = seisnc.sel(xline=2349).transpose("twt", "iline").data
zslice = seisnc.sel(twt=2900, method="nearest").transpose("iline", "xline").data

q = iline.quantile([0, 0.001, 0.5, 0.999, 1]).values
rq = np.max(np.abs([q[1], q[-2]]))

iline.plot(robust=True, ax=axs[0, 0], yincrease=False)
xline.plot(robust=True, ax=axs[0, 1], yincrease=False)
zslice.plot(robust=True, ax=axs[0, 2])

imshow_kwargs = dict(
    cmap="seismic", aspect="auto", vmin=-rq, vmax=rq, interpolation="bicubic"
)

axs[1, 0].imshow(iline.values, **imshow_kwargs)
axs[1, 0].set_title("iline")
axs[1, 1].imshow(xline.values, **imshow_kwargs)
axs[1, 1].set_title("xline")
axs[1, 2].imshow(zslice.values, origin="lower", **imshow_kwargs)
axs[1, 2].set_title("twt")
