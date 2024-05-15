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
# # Working with seismic and interpreted horizons

# %% tags=["hide-code"]
# Disable progress bars for small examples
from segysak.progress import Progress

Progress.set_defaults(disable=True)

# %%
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# %%
from segysak.segy import (
    get_segy_texthead,
    segy_header_scan,
    segy_header_scrape,
)

# %%
segy_file = pathlib.Path("data/volve10r12-full-twt-sub3d.sgy")
cube = xr.open_dataset(
    segy_file,
    dim_byte_fields={"iline": 5, "xline": 21},
    extra_byte_fields={"cdp_x": 73, "cdp_y": 77},
)
cube.segysak.scale_coords()

# %%
print("Loaded cube size: {}".format(cube.segysak.humanbytes))

# %% [markdown]
# ## Load horizon data
#
# First we load the horizon data to a Pandas DataFrame and take a look at the first few lines:

# %%
hrz_file = pathlib.Path("data/hor_twt_hugin_fm_top.dat")
hrz = pd.read_csv(hrz_file, names=["cdp_x", "cdp_y", "twt"], sep="\s+")
hrz.head()

# %% [markdown]
# ## Display horizon and seismic extents
#
# We can build a plot to display the horizon in its entirety and overlay it with the extent of the seismic cube previously loaded.
#
# First we use **segysak** built-in `calc_corner_points()` method to calculate the corner points of the loaded cube and copy them to a numpy array to be used for the plot:

# %%
cp = cube.segysak.calc_corner_points()
corners = np.array(cp)
corners

# %% [markdown]
# To display the horizon we need to grid the raw data first:

# %%
from scipy.interpolate import griddata

xi = np.linspace(hrz.cdp_x.min(), hrz.cdp_x.max(), 250)
yi = np.linspace(hrz.cdp_y.min(), hrz.cdp_y.max(), 250)
X, Y = np.meshgrid(xi, yi)
Z = griddata((hrz.cdp_x, hrz.cdp_y), hrz.twt, (X, Y), rescale=True)

# %% [markdown]
# And this is the resulting plot with the extent of the loaded cube displayed as a thick red outline:

# %%
from matplotlib.patches import Polygon

survey_limits = Polygon(
    corners, fill=False, edgecolor="r", linewidth=2, label="3D survey extent"
)

f, ax = plt.subplots(figsize=(8, 6))
pp = ax.pcolormesh(X, Y, Z, cmap="terrain_r")
f.colorbar(pp, orientation="horizontal", label="TWT [ms]")
ax.add_patch(survey_limits)
ax.axis("equal")
ax.legend()
ax.set_title("Top Hugin fm.")

# %% [markdown]
# ## Extracting amplitudes along the horizon
#
# This is where the magic of segysak comes in. We use `surface_from_points` to map the loaded horizon imported in tabular format to each seismic bin. The input horizon in this case is defined in geographical coordinates but it would also have worked if it was defined in inlines and crosslines:

# %%
cube = cube.set_coords(('cdp_x', 'cdp_y'))
hrz_mapped = cube.seis.surface_from_points(hrz, "twt", right=("cdp_x", "cdp_y"))

# %% [markdown]
# And to extract seismic amplitudes along this horizon we use the magic of `xarray`:

# %%
amp = cube.data.interp(
    {"iline": hrz_mapped.iline, "xline": hrz_mapped.xline, "samples": hrz_mapped.twt}
)

# %% [markdown]
# For the next plot we use another attribute automatically calculated by **segysak** during loading to squeeze the colormap used when displaying amplitudes:

# %%
#minamp, maxamp = cube.attrs["percentiles"][1], cube.attrs["percentiles"][-2]

# %% [markdown]
# ...and we calculate the minimum and maximum X and Y coordinates using the `corners` array described above to set the figure extent and zoom in the area covered by the seismic cube:

# %%
xmin, xmax = corners[:, 0].min(), corners[:, 0].max()
ymin, ymax = corners[:, 1].min(), corners[:, 1].max()

# %% [markdown]
# We can plot `amp` now on top of the same twt grid we did above:

# %%
survey_limits = Polygon(
    corners, fill=False, edgecolor="r", linewidth=2, label="3D survey extent"
)

f, ax = plt.subplots(nrows=2, figsize=(8, 10))
ax[0].pcolormesh(X, Y, Z, cmap="terrain_r")
ax[0].add_patch(survey_limits)
for aa in ax:
    hh = aa.pcolormesh(
        amp.cdp_x, amp.cdp_y, amp.data, cmap="RdYlBu", #vmin=minamp, vmax=maxamp
    )
    aa.axis("equal")
ax[0].legend()
ax[1].set_xlim(xmin, xmax)
ax[1].set_ylim(ymin, ymax)
ax[0].set_title("Top Hugin fm. and amplitude extraction on loaded seismic")
ax[1].set_title("Amplitude extraction at Top Hugin (zoom)")
cax = f.add_axes([0.15, 0.16, 0.5, 0.02])
f.colorbar(hh, cax=cax, orientation="horizontal")

# %% [markdown]
# Another classic display is to superimpose the structure contours to the amplitudes. This is much faster and easier using survey coordinates (inlines and crosslines):

# %%
f, ax = plt.subplots(figsize=(12, 4))
amp.plot(cmap="RdYlBu")
cs = plt.contour(amp.xline, amp.iline, hrz_mapped.twt, levels=20, colors="grey")
plt.clabel(cs, fontsize=10, fmt="%.0f")
ax.invert_xaxis()

# %% [markdown]
# ## Display horizon in section view

# %%
opt = dict(
    x="xline",
    y="samples",
    add_colorbar=True,
    interpolation="spline16",
    robust=True,
    yincrease=False,
    cmap="Greys",
)

inl_sel = [10130, 10100]

f, ax = plt.subplots(nrows=2, figsize=(10, 6), sharey=True, constrained_layout=True)
for i, val in enumerate(inl_sel):
    cube.data.sel(iline=val, samples=slice(2300, 3000)).plot.imshow(ax=ax[i], **opt)
    x, t = hrz_mapped.sel(iline=val).xline, hrz_mapped.sel(iline=val).twt
    ax[i].plot(x, t, color="r")
    ax[i].invert_xaxis()

# %% [markdown]
# We can also show an overlay of amplitudes and two-way-times along the same inlines:

# %%
inl_sel = [10130, 10100]

f, ax = plt.subplots(nrows=2, figsize=(10, 6), sharey=True, constrained_layout=True)

for i, val in enumerate(inl_sel):
    axz = ax[i].twinx()
    x, t = amp.sel(iline=val).xline, amp.sel(iline=val).samples
    a = amp.sel(iline=val).data
    ax[i].plot(x, a, color="r")
    axz.plot(x, t, color="k")
    ax[i].invert_xaxis()
    axz.invert_yaxis()
    ax[i].set_ylabel("Amplitude", color="r")
    plt.setp(ax[i].yaxis.get_majorticklabels(), color="r")
    axz.set_ylabel("TWT [ms]")
    ax[i].set_title("Amplitude and two-way-time at inline {}".format(val))

# %% [markdown]
# ## Horizon Sculpting and Windowed map Extraction

# %% [markdown]
# To create windows map extraction it is necessary to first mask the seisnc cube and then to collapse it along the chosen axis using a prefered method. In this case we are just going to calculate the mean amplitude in a 100ms window below our horizon.

# %%
mask_below = cube.where(cube.samples < hrz_mapped.twt + 100)
mask_above = cube.where(cube.samples > hrz_mapped.twt)
masks = [mask_above, mask_below]

# %% [markdown]
# Lets see what our masks look like

# %%
opt = dict(
    x="xline",
    y="samples",
    add_colorbar=True,
    interpolation="spline16",
    robust=True,
    yincrease=False,
    cmap="Greys",
)

inl_sel = [10130, 10100]

f, ax = plt.subplots(nrows=2, figsize=(10, 6), sharey=True, constrained_layout=True)
for i, val in enumerate(
    inl_sel,
):
    masks[i].data.sel(iline=val, samples=slice(2300, 3000)).plot.imshow(ax=ax[i], **opt)
    x, t = hrz_mapped.sel(iline=val).xline, hrz_mapped.sel(iline=val).twt
    ax[i].plot(x, t, color="r")
    ax[i].invert_xaxis()

# %% [markdown]
# And if we combine the masks.

# %%
opt = dict(
    x="xline",
    y="samples",
    add_colorbar=True,
    interpolation="spline16",
    robust=True,
    yincrease=False,
    cmap="Greys",
)

inl_sel = [10130, 10100]

f, ax = plt.subplots(nrows=1, figsize=(10, 3), sharey=True, constrained_layout=True)

masked_data = 0.5 * (mask_below + mask_above)

masked_data.data.sel(iline=val, samples=slice(2300, 3000)).plot.imshow(ax=ax, **opt)
x, t = hrz_mapped.sel(iline=val).xline, hrz_mapped.sel(iline=val).twt
ax.plot(x, t, color="r")
ax.invert_xaxis()

# %% [markdown]
# To get the horizon window extraction for sum of amplitudes we now need to sum along the time axis. Or
# we can use the `np.apply_along_axis` function to apply a custom function to our masked cube.

# %% tags=[]
summed_amp = masked_data.sum(dim="samples")

f, ax = plt.subplots(figsize=(12, 4))
summed_amp.data.plot.imshow(
    cmap="RdYlBu",
    interpolation="spline16",
)
cs = plt.contour(amp.xline, amp.iline, hrz_mapped.twt, levels=20, colors="grey")
plt.clabel(cs, fontsize=10, fmt="%.0f")
ax.invert_xaxis()
ax.set_title("Sum of amplitudes Top Hugin 0 to +100ms")
