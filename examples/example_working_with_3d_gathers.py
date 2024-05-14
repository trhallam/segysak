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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Working with 3D Gathers
#
# Gathers or pre-stack data are common in seismic interpretation and processing. In this example we will load gather data by finding and specifying the offset byte location. Learn different ways to plot and select offset data. As well as perform a simple mute and trace stack to reduce the offset dimension.

# %% editable=true slideshow={"slide_type": ""} tags=["hide-code"]
# Disable progress bars for small examples
from segysak.progress import Progress
Progress.set_defaults(disable=True)

# %% editable=true slideshow={"slide_type": ""}
import pathlib
from IPython.display import display
import pandas as pd
import xarray as xr
import numpy as np
from segysak.segy import segy_loader, segy_header_scan
import matplotlib.pyplot as plt

# %% [markdown] editable=true slideshow={"slide_type": ""}
# This example uses a subset of the Penobscot 3D with data exported from the OpendTect [project](https://terranubis.com/datainfo/Penobscot).

# %% [markdown] editable=true slideshow={"slide_type": ""}
# First we scan the data to determine which byte locations contain the relevant information. We will need to provide a byte location for the offset variable so a 4th dimension can be created when loaded the data.

# %% editable=true slideshow={"slide_type": ""}
segy_file = pathlib.Path("data/3D_gathers_pstm_nmo.sgy")
with pd.option_context("display.max_rows", 100):
    display(segy_header_scan(segy_file))

# %% editable=true slideshow={"slide_type": ""}
penobscot_3d_gath = xr.open_dataset(
    segy_file, dim_byte_fields={"iline":189, "xline":193, "offset":37}, extra_byte_fields={"cdp_x":181, "cdp_y":185}
)

# coordinates are not scaled automatically, to use the coordinate scalar found in the trace headers, we run scale_coords()
penobscot_3d_gath.segysak.scale_coords()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Note that the loaded Dataset has four dimensions with the additional dimension labeled offset. There are 61 offsets in this dataset or 61 traces per inline and xline location.

# %% editable=true slideshow={"slide_type": ""}
display(penobscot_3d_gath)
print(penobscot_3d_gath.offset.values)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Lets check that the data looks OK for a couple of offsets. We've only got a small dataset of 11x11 traces so the seismic will look at little odd at this scale.

# %% editable=true slideshow={"slide_type": ""}
fig, axs = plt.subplots(ncols=2, figsize=(20, 10))

penobscot_3d_gath.isel(iline=5, offset=0).data.T.plot(
    yincrease=False, ax=axs[0], vmax=5000
)
penobscot_3d_gath.isel(xline=5, offset=0).data.T.plot(
    yincrease=False, ax=axs[1], vmax=5000
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Plotting Gathers Sequentially
#
# Plotting of gathers is often done in a stacked way, displaying sequential gathers along a common dimension, usually inline or crossline. Xarray provides the `stack` method which can be used to stack labelled dimensions together.

# %% editable=true slideshow={"slide_type": ""}
fig, axs = plt.subplots(figsize=(20, 10))
axs.imshow(
    penobscot_3d_gath.isel(iline=0)
    .data.stack(stacked_offset=("xline", "offset"))
    .values,
    vmin=-5000,
    vmax=5000,
    cmap="seismic",
    aspect="auto",
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# One can easily create a common offset stack by reversing the stacked dimension arguments `"offset"` and `"xline"`.

# %% editable=true slideshow={"slide_type": ""}
fig, axs = plt.subplots(figsize=(20, 10))
axs.imshow(
    penobscot_3d_gath.isel(iline=0)
    .data.stack(stacked_offset=("offset", "xline"))
    .values,
    vmin=-5000,
    vmax=5000,
    cmap="seismic",
    aspect="auto",
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Arbitrary line extraction on Gathers
#
# Arbitrary line slicing of gathers based upon coordinates is also possible. Lets create a line that crosses the 3D.

# %% tags=[]
penobscot_3d_gath

# %% tags=[]
penobscot_3d_gath.drop_dims('offset')

# %% tags=[]
penobscot_3d_gath.isel(offset=0).segysak.calc_corner_points()

# %% editable=true slideshow={"slide_type": ""}
arb_line = np.array([(733600, 4895180.0), (733850, 4895180.0)])

# we must drop the offset axis somehow, xy locations were loaded for each header position but
# they must be reduced, we could take the mean over the dimension but here we just select the 
# first value using `offset=0`
ax = penobscot_3d_gath.isel(offset=0).segysak.plot_bounds()
ax.plot(arb_line[:, 0], arb_line[:, 1], label="arb_line")
plt.legend()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Here we need to think carefully about the `bin_spacing_hint`. We also don't want to interpolate the gathers, so we use `xysel_method="nearest"`.

# %% editable=true slideshow={"slide_type": ""}
penobscot_3d_gath_arb = penobscot_3d_gath.segysak.xysel(
    arb_line, method="nearest"
)

# %%
fig, axs = plt.subplots(figsize=(20, 10))
axs.imshow(
    penobscot_3d_gath_arb.data.stack(
        stacked_offset=(
            "cdp",
            "offset",
        )
    ).values,
    vmin=-5000,
    vmax=5000,
    cmap="seismic",
    aspect="auto",
)

# %% [markdown]
# ## Muting and Stacking Gathers
#
# Using one of our gathers let's define a mute function before we stack the data.

# %% editable=true slideshow={"slide_type": ""}
fig, axs = plt.subplots(ncols=2, figsize=(20, 10))

penobscot_3d_gath.isel(iline=5, xline=0).transpose('samples', ...).data.plot(
    yincrease=False, ax=axs[0], vmax=5000,
)

# the mute relates the offset to an expected twt, let's just use a linear mute for this example
mute = penobscot_3d_gath.offset * 0.6 + 300
# and then we can plot it up
mute.plot(ax=axs[0], color="k")
# apply the mute to the volume
penobscot_3d_gath_muted = penobscot_3d_gath.where(penobscot_3d_gath.samples > mute)

# muted
penobscot_3d_gath_muted.isel(iline=5, xline=0).data.transpose('samples', ...).plot(
    yincrease=False, ax=axs[1], vmax=5000
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Stacking is the process of averaging the gathers for constant time to create a single trace per inline and crossline location.

# %% editable=true slideshow={"slide_type": ""}
fig, axs = plt.subplots(ncols=3, figsize=(20, 10))

plot_kwargs = dict(vmax=5000, interpolation="bicubic", yincrease=False)

# compare with the zero offset trace and use imshow for interpolation
penobscot_3d_gath.isel(iline=5, offset=0).data.T.plot.imshow(ax=axs[0], **plot_kwargs)

# stack the no mute data
penobscot_3d_gath.isel(iline=5).data.mean("offset").T.plot.imshow(
    ax=axs[1], **plot_kwargs
)

# stack the muted data
penobscot_3d_gath_muted.isel(iline=5).data.mean("offset").T.plot.imshow(
    ax=axs[2], **plot_kwargs
)
