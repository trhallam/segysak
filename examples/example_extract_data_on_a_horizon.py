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
# # Extract data at the intersection of a horizon and 3D volume

# %% nbsphinx="hidden"
import warnings

warnings.filterwarnings("ignore")

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt

# %matplotlib inline

from segysak.segy import (
    segy_loader,
    get_segy_texthead,
    segy_header_scan,
    segy_header_scrape,
)

from os import path

# %% [markdown]
# ## Load Small 3D Volume from Volve

# %%
volve_3d_path = path.join("data", "volve10r12-full-twt-sub3d.sgy")
print("3D", volve_3d_path, path.exists(volve_3d_path))

# %%
get_segy_texthead(volve_3d_path)

# %%
from segysak.segy import well_known_byte_locs

volve_3d = segy_loader(volve_3d_path, **well_known_byte_locs("petrel_3d"))
volve_3d.data

# %% [markdown]
# ## Load up horizon data

# %%
top_hugin_path = path.join("data", "hor_twt_hugin_fm_top.dat")
print("Top Hugin", top_hugin_path, path.exists(top_hugin_path))

# %%
import pandas as pd

top_hugin_df = pd.read_csv(top_hugin_path, names=["cdp_x", "cdp_y", "twt"], sep=" ")
top_hugin_df.head()

# %% [markdown]
# Would be good to plot a seismic (iline,xline) section in Pyvista as well

# %%
# import pyvista as pv

# point_cloud = pv.PolyData(-1*top_hugin_df.to_numpy(), cmap='viridis')
# point_cloud.plot(eye_dome_lighting=True)

# %% [markdown]
# Alternativey we can use the points to output a `xarray.Dataset` which comes with coordinates for plotting already gridded up for Pyvista.

# %%
top_hugin_ds = volve_3d.seis.surface_from_points(
    top_hugin_df, "twt", right=("cdp_x", "cdp_y")
)
top_hugin_ds

# %%
# the twt values from the points now in a form we can relate to the xarray cube.
plt.imshow(top_hugin_ds.twt)

# %%
# point_cloud = pv.StructuredGrid(
#     top_hugin_ds.cdp_x.values,
#     top_hugin_ds.cdp_y.values,top_hugin_ds.twt.values,
#     cmap='viridis')
# point_cloud.plot(eye_dome_lighting=True)

# %% [markdown]
# ## Horizon Amplitude Extraction

# %% [markdown]
# Extracting horizon amplitudes requires us to interpolate the cube onto the 3D horizon.

# %%
top_hugin_amp = volve_3d.data.interp(
    {"iline": top_hugin_ds.iline, "xline": top_hugin_ds.xline, "twt": top_hugin_ds.twt}
)

# %%
#
fig = plt.figure(figsize=(15, 5))
top_hugin_amp.plot(cmap="bwr")
cs = plt.contour(
    top_hugin_amp.xline, top_hugin_amp.iline, top_hugin_ds.twt, levels=20, colors="grey"
)
plt.clabel(cs, fontsize=14, fmt="%.1f")
