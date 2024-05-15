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

# %% nbsphinx="hidden" tags=["hide-code"]
# Disable progress bars for small examples
from segysak.progress import Progress

Progress.set_defaults(disable=True)

# %% [markdown]
# # Merging Seismic Data Cubes

# %% [markdown]
# Often we receive seismic data cubes which we wish to merge that have different geometries. This workflow will guide you through the process of merging two such cubes.

# %%
from segysak import create3d_dataset
from scipy.optimize import curve_fit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from segysak import open_seisnc, create_seismic_dataset

from dask.distributed import Client

# start a dask client for processing
client = Client()

# %% [markdown]
# Creating some synthetic seismic surveys for us to play with. The online documentation uses a smaller example, but you can stress dask by commenting the dimension lines for smaller data and uncommenting the large desktop size volumes.

# %%
# large desktop size merge
# i1, i2, iN = 100, 750, 651
# x1, x2, xN = 300, 1250, 951
# t1, t2, tN = 0, 1848, 463
# trans_x, trans_y = 10000, 2000

# online example size merge
i1, i2, iN = 650, 750, 101
x1, x2, xN = 1150, 1250, 101
t1, t2, tN = 0, 200, 201
trans_x, trans_y = 9350, 1680

iline = np.linspace(i1, i2, iN, dtype=int)
xline = np.linspace(x1, x2, xN, dtype=int)
twt = np.linspace(t1, t2, tN, dtype=int)

affine_survey1 = Affine2D()
affine_survey1.rotate_deg(20).translate(trans_x, trans_y)

# create an empty dataset and empty coordinates
survey_1 = create_seismic_dataset(twt=twt, iline=iline, xline=xline)
survey_1["cdp_x"] = (("iline", "xline"), np.empty((iN, xN)))
survey_1["cdp_y"] = (("iline", "xline"), np.empty((iN, xN)))

stacked_iline = survey_1.iline.broadcast_like(survey_1.cdp_x).stack(
    {"ravel": (..., "xline")}
)
stacked_xline = survey_1.xline.broadcast_like(survey_1.cdp_x).stack(
    {"ravel": (..., "xline")}
)

points = np.dstack([stacked_iline, stacked_xline])
points_xy = affine_survey1.transform(points[0])
# put xy back in stacked and unstack into survey_1
survey_1["cdp_x"] = stacked_iline.copy(data=points_xy[:, 0]).unstack()
survey_1["cdp_y"] = stacked_iline.copy(data=points_xy[:, 1]).unstack()

# and the same for survey_2 but with a different affine to survey 1
affine_survey2 = Affine2D()
affine_survey2.rotate_deg(120).translate(11000, 3000)

survey_2 = create_seismic_dataset(twt=twt, iline=iline, xline=xline)
survey_2["cdp_x"] = (("iline", "xline"), np.empty((iN, xN)))
survey_2["cdp_y"] = (("iline", "xline"), np.empty((iN, xN)))

stacked_iline = survey_1.iline.broadcast_like(survey_1.cdp_x).stack(
    {"ravel": (..., "xline")}
)
stacked_xline = survey_1.xline.broadcast_like(survey_1.cdp_x).stack(
    {"ravel": (..., "xline")}
)

points = np.dstack([stacked_iline, stacked_xline])
points_xy = affine_survey2.transform(points[0])
# put xy back in stacked and unstack into survey_1
survey_2["cdp_x"] = stacked_iline.copy(data=points_xy[:, 0]).unstack()
survey_2["cdp_y"] = stacked_iline.copy(data=points_xy[:, 1]).unstack()


# %% [markdown]
# Let's check that there is a bit of overlap between the two surveys.

# %%
# plotting every 10th line

plt.figure(figsize=(10, 10))
survey_1_plot = plt.plot(
    survey_1.cdp_x.values[::10, ::10], survey_1.cdp_y.values[::10, ::10], color="grey"
)
survey_2_plot = plt.plot(
    survey_2.cdp_x.values[::10, ::10], survey_2.cdp_y.values[::10, ::10], color="blue"
)
# plt.aspect("equal")
plt.legend([survey_1_plot[0], survey_2_plot[0]], ["survey_1", "survey_2"])

# %% [markdown]
# Let's output these two datasets to disk so we can use lazy loading from seisnc. If the files are large, it's good practice to do this, because you will probably run out of memory.
#
# We are going to fill survey one with a values of 1 and survey 2 with a value of 2, just for simplicity in this example.
# We also tell the dtype to be `np.float32` to reduce memory usage.

# %%
# save out with new geometry
survey_1["data"] = (
    ("iline", "xline", "twt"),
    np.full((iN, xN, tN), 1, dtype=np.float32),
)
survey_1.seisio.to_netcdf("data/survey_1.seisnc")
del survey_1
survey_2["data"] = (
    ("iline", "xline", "twt"),
    np.full((iN, xN, tN), 2, dtype=np.float32),
)
survey_2.seisio.to_netcdf("data/survey_2.seisnc")
del survey_2

# %% [markdown]
# ## How to merge two different geometries

# %% [markdown]
# Let us reimport the surveys we created but in a chunked (lazy) way.

# %%
survey_1 = open_seisnc("data/survey_1.seisnc", chunks=dict(iline=10, xline=10, twt=100))
survey_2 = open_seisnc("data/survey_2.seisnc", chunks=dict(iline=10, xline=10, twt=100))

# %% [markdown]
# Check that the survey is chunked by looking at the printout for our datasets. Lazy and chunked data will have a `dask.array<chunksize=` where the values are usually displayed.

# %%
survey_1

# %% [markdown]
# It will help with our other functions if we load the cdp_x and cdp_y locations into memory. This can be done using the `load()` method.

# %%
survey_1["cdp_x"] = survey_1["cdp_x"].load()
survey_1["cdp_y"] = survey_1["cdp_y"].load()
survey_2["cdp_x"] = survey_2["cdp_x"].load()
survey_2["cdp_y"] = survey_2["cdp_y"].load()


# %%
survey_1

# %% [markdown]
# We will also need an affine transform which converts from il/xl on survey 2 to il/xl on survey 1. This will relate the two grids to each other. Fortunately affine transforms can be added together which makes this pretty straight forward.

# %%
ilxl2_to_ilxl1 = (
    survey_2.seis.get_affine_transform()
    + survey_1.seis.get_affine_transform().inverted()
)

# %% [markdown]
# Let's check the transform to see if it makes sense. The colouring of our values by inline should be the same after the transform is applied. Cool.

# %%
# plotting every 50th line

n = 5

fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
survey_1_plot = axs[0].scatter(
    survey_1.cdp_x.values[::n, ::n],
    survey_1.cdp_y.values[::n, ::n],
    c=survey_1.iline.broadcast_like(survey_1.cdp_x).values[::n, ::n],
    s=20,
)
survey_2_plot = axs[0].scatter(
    survey_2.cdp_x.values[::n, ::n],
    survey_2.cdp_y.values[::n, ::n],
    c=survey_2.iline.broadcast_like(survey_2.cdp_x).values[::n, ::n],
    s=20,
)
plt.colorbar(survey_1_plot, ax=axs[0])
axs[0].set_aspect("equal")

axs[0].set_title("inline numbering before transform")

new_ilxl = ilxl2_to_ilxl1.transform(
    np.dstack(
        [
            survey_2.iline.broadcast_like(survey_2.cdp_x).values.ravel(),
            survey_2.xline.broadcast_like(survey_2.cdp_x).values.ravel(),
        ]
    )[0]
).reshape((survey_2.iline.size, survey_2.xline.size, 2))

survey_2.seis.calc_corner_points()
s2_corner_points_in_s1 = ilxl2_to_ilxl1.transform(survey_2.attrs["corner_points"])

print("Min Iline:", s2_corner_points_in_s1[:, 0].min(), survey_1.iline.min().values)
min_iline_combined = min(
    s2_corner_points_in_s1[:, 0].min(), survey_1.iline.min().values
)
print("Max Iline:", s2_corner_points_in_s1[:, 0].max(), survey_1.iline.max().values)
max_iline_combined = max(
    s2_corner_points_in_s1[:, 0].max(), survey_1.iline.max().values
)

print("Min Xline:", s2_corner_points_in_s1[:, 1].min(), survey_1.xline.min().values)
min_xline_combined = min(
    s2_corner_points_in_s1[:, 1].min(), survey_1.xline.min().values
)
print("Max Xline:", s2_corner_points_in_s1[:, 1].max(), survey_1.xline.max().values)
max_xline_combined = max(
    s2_corner_points_in_s1[:, 1].max(), survey_1.xline.max().values
)
print(min_iline_combined, max_iline_combined, min_xline_combined, max_xline_combined)

survey_1_plot = axs[1].scatter(
    survey_1.cdp_x.values[::n, ::n],
    survey_1.cdp_y.values[::n, ::n],
    c=survey_1.iline.broadcast_like(survey_1.cdp_x).values[::n, ::n],
    vmin=min_iline_combined,
    vmax=max_iline_combined,
    s=20,
    cmap="hsv",
)


survey_2_plot = axs[1].scatter(
    survey_2.cdp_x.values[::n, ::n],
    survey_2.cdp_y.values[::n, ::n],
    c=new_ilxl[::n, ::n, 0],
    vmin=min_iline_combined,
    vmax=max_iline_combined,
    s=20,
    cmap="hsv",
)
plt.colorbar(survey_1_plot, ax=axs[1])

axs[1].set_title("inline numbering after transform")
axs[1].set_aspect("equal")

# %% [markdown]
# So we now have a transform that can convert ilxl from `survey_2` to ilxl from `survey_1` and we need to create a combined geometry for the merge. The dims need to cover the maximum range (inline, crossline) of two surveys. We are also going to use survey 1 as the base as we don't want to have to resample two cubes, although if you have a prefered geometry you could do that (custom affine transforms per survey required).

# %%
# create new dims
iline_step = survey_1.iline.diff("iline").mean().values
xline_step = survey_1.xline.diff("xline").mean().values

# create the new dimensions
new_iline_dim = np.arange(
    int(min_iline_combined // iline_step) * iline_step,
    int(max_iline_combined) + iline_step * 2,
    iline_step,
    dtype=np.int32,
)
new_xline_dim = np.arange(
    int(min_xline_combined // xline_step) * xline_step,
    int(max_xline_combined) + xline_step * 2,
    xline_step,
    dtype=np.int32,
)

# create a new empty dataset and a blank cdp_x for dims broadcasting
survey_comb = create_seismic_dataset(
    iline=new_iline_dim, xline=new_xline_dim, twt=survey_1.twt
)
survey_comb["cdp_x"] = (
    ("iline", "xline"),
    np.empty((new_iline_dim.size, new_xline_dim.size)),
)

# calculate the x and y using the survey 1 affine which is our base grid and reshape to the dataset grid
new_cdp_xy = affine_survey1.transform(
    np.dstack(
        [
            survey_comb.iline.broadcast_like(survey_comb.cdp_x).values.ravel(),
            survey_comb.xline.broadcast_like(survey_comb.cdp_x).values.ravel(),
        ]
    )[0]
)
new_cdp_xy_grid = new_cdp_xy.reshape((new_iline_dim.size, new_xline_dim.size, 2))

# plot to check
plt.figure(figsize=(10, 10))

survey_comb_plot = plt.plot(
    new_cdp_xy_grid[:, :, 0], new_cdp_xy_grid[:, :, 1], color="red", alpha=0.5
)
survey_1_plot = plt.plot(
    survey_1.cdp_x.values[::10, ::10], survey_1.cdp_y.values[::10, ::10], color="grey"
)
survey_2_plot = plt.plot(
    survey_2.cdp_x.values[::10, ::10], survey_2.cdp_y.values[::10, ::10], color="blue"
)

plt.legend(
    [survey_1_plot[0], survey_2_plot[0], survey_comb_plot[0]],
    ["survey_1", "survey_2", "survey_merged"],
)

# put the new x and y in the empty dataset
survey_comb["cdp_x"] = (("iline", "xline"), new_cdp_xy_grid[..., 0])
survey_comb["cdp_y"] = (("iline", "xline"), new_cdp_xy_grid[..., 1])
survey_comb = survey_comb.set_coords(("cdp_x", "cdp_y"))

# %% [markdown]
# ## Resampling `survey_2` to `survey_1` geometry
#
# Resampling one dataset to another requires us to tell Xarray/dask where we want the new traces to be. First we can convert the x and y coordinates of the combined survey to the il xl of survey 2 using the affine transform of survey 2. This transform works il/xl to x and y and therefore we need it inverted.

# %%
survey_2_new_ilxl_loc = affine_survey2.inverted().transform(new_cdp_xy)

# %% [markdown]
# We then need to create a sampling DataArray. If we passed the iline and cross line locations from the previous cell unfortunately Xarray would broadcast the inline and xline values against each other. The simplest way is to create a new stacked flat dimension which we can unstack into a cube later.
#
# We also need to give it unique names so xarray doesn't confuse the new dimensions with the old dimensions.

# %%
flat = (
    survey_comb.rename(  # use the combined survey
        {"iline": "new_iline", "xline": "new_xline"}
    )  # renaming inline and xline so they don't confuse xarray
    .stack(
        {"flat": ("new_iline", "new_xline")}
    )  # and flatten the iline and xline axes using stack
    .flat  # return just the flat coord object which is multi-index (so we can unstack later)
)
flat

# %%
# create a dictionary of new coordinates to sample to. We use flat here because it will easily allow us to
# unstack the data after interpolation. It's necessary to use this flat dataset, otherwise xarray will try to
# interpolate on the cartesian product of iline and xline, which isn't really what we want.

resampling_data_arrays = dict(
    iline=xr.DataArray(survey_2_new_ilxl_loc[:, 0], dims="flat", coords={"flat": flat}),
    xline=xr.DataArray(survey_2_new_ilxl_loc[:, 1], dims="flat", coords={"flat": flat}),
)

resampling_data_arrays

# %% [markdown]
# The resampling can take a while with larger cubes, so it is good to use dask and to output the cube to disk at this step.
# When using dask for processing it is better to output to disk regularly, as this will improve how your code runs and the overall memory usage.
#
# *If you are executing this exmample locally, the task progress can be viewed by opening the dask [client](http://localhost:8787/status).*

# %%
survey_2_resamp = survey_2.interp(**resampling_data_arrays)
survey_2_resamp_newgeom = (
    survey_2_resamp.drop_vars(("iline", "xline"))
    .unstack("flat")
    .rename({"new_iline": "iline", "new_xline": "xline"})
    .data
)
survey_2_resamp_newgeom.to_netcdf("data/survey_2_1.nc", compute=True, engine="h5netcdf")

# %% [markdown]
# ## Combining Cubes
#
# The last step is combining the cube is to load the two datasets to be combined and concatenate them together along a new axis. This will simplify reduction processes later.

# %%
survey_2_resamp_newgeom = xr.open_dataarray(
    "data/survey_2_1.nc", chunks=dict(iline=10, xline=10, twt=100), engine="h5netcdf"
)

# %%
survey_2_resamp_newgeom.expand_dims({"survey": [2]})

# %%
# concatenate survey with new dimension "survey".
survey_comb["data"] = xr.concat([survey_1.data, survey_2_resamp_newgeom], "survey")

# %%
survey_comb

# %% [markdown]
# We can check the merged surveys by looking at some plots. If we select just the first survey the values will be 1. If we select just the second survey the values will be 2. And if we take the mean along the survey dimension, then where the surveys overlap the values will be 1.5.
#
# For seismic data, a better form of conditioning and reduction might be required for merging traces together to ensure a smoother seam.

# %%
sel = survey_comb.sel(xline=1190)

fig, axs = plt.subplots(ncols=3, figsize=(30, 10))

sel.isel(survey=0).data.T.plot(ax=axs[0], yincrease=False, vmin=1, vmax=2)
sel.isel(survey=1).data.T.plot(ax=axs[1], yincrease=False, vmin=1, vmax=2)
sel.data.mean("survey").T.plot(ax=axs[2], yincrease=False, vmin=1, vmax=2)

# %%
survey_comb.twt

# %% [markdown]
# And if we inspect a time slice.

# %%
sel = survey_comb.sel(twt=100)

fig, axs = plt.subplots(ncols=3, figsize=(30, 10))

sel.isel(survey=0).data.T.plot(ax=axs[0], yincrease=False, vmin=1, vmax=2)
sel.isel(survey=1).data.T.plot(ax=axs[1], yincrease=False, vmin=1, vmax=2)
sel.data.mean("survey").T.plot(ax=axs[2], yincrease=False, vmin=1, vmax=2)

# %%
