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
# # Cube interpolation and depth conversion
#
# This notebook demonstrates depth conversion with Xarray (data loaded via SEGY-SAK). For larger volumes, the dask example should be followed, by creating a client with workers first.

# %% editable=true slideshow={"slide_type": ""}
import pathlib
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from segysak.segy import segy_header_scan

# %%
## to run this example with dask uncomment and run the following

# from dask.distributed import Client
# client = Client()
# client.cluster.scale(2, memory="0.5gb")
# client

# %% editable=true slideshow={"slide_type": ""} tags=["hide-code"]
# Disable progress bars for small examples
from segysak.progress import Progress

Progress.set_defaults(disable=True)

# %% [markdown]
# ## Load the data
#
# First the seismic data and velocity data needs to be loaded. Note the Velocity data is in depth and has significantly fewer inlines and crosslines. That is because the velocity data is sub-sampled relative to the seismic data. This is common where velocities have been picked or calculated on a sparse grid through the seismic volume.

# %% editable=true slideshow={"slide_type": ""}
# the data files
volve3d_path = pathlib.Path("data/volve10r12-full-twt-sub3d.sgy")
volve3d_vels_path = pathlib.Path("data/volve10-migvel-depth-sub3d.sgy")

# %% editable=true slideshow={"slide_type": ""}
# load the data
vel_ds = xr.open_dataset(
    volve3d_vels_path,
    dim_byte_fields={"iline": 189, "xline": 193},
    extra_byte_fields={"cdpx": 181, "cdpy": 185},
    chunks={"iline": 1, "xline": -1, "samples": -1},
)
vel_ds = vel_ds.rename({"data": "migintvel", "samples": "depth"})

volve_ds = xr.open_dataset(
    volve3d_path,
    dim_byte_fields={"iline": 5, "xline": 21},
    extra_byte_fields={"cdp_x": 73, "cdp_y": 77},
    chunks={"iline": 1, "xline": -1, "samples": -1},
)
volve_ds = volve_ds.rename(
    {
        "samples": "twt",
    }
)

print("Velocity data dims:", vel_ds.dims)
print("Seismic data dims:", volve_ds.dims)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Create a TWT volume from the velocity cube
#
# The velocity cube is in depth. So a mapping to TWT needs to be created to match
# the input seismic data. The mapping is essentially a TWT cube sampled in depth.
# When depth conversion is performed we interpolate or sample the seismic data to
# these TWT time values which implicitly gives the new seismic volume in depth.
# Lets think about why it is done in this way.
#
# Just calculating the depth for each TWT sample in the original seismic cube using
# the provided velocities, would result in irregular sampling in the depth domain.
# Furthermore, the depth axis samples would be different for each trace, which have
# unique velocity functions. Usually, our seismic data is required to be on a
# regular grid, so unique and irregular depth sampling per trace is not appropriate.
#
# The velocities are migration interval velocities and represent the velocity in
# the interval of the sample. Lets look at a single trace with smooth plotting
# (left) and step based plotting (right).

# %%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
vel_trace = vel_ds.sel(iline=10095, xline=2256).migintvel
vel_trace.plot(y="depth", yincrease=False, label="smooth", ax=axs[0])
axs[0].step(vel_trace.values, vel_trace.depth, label="steps")
axs[0].legend()

sub = slice(100, 150)  # subslice to look at in zoom (right plot)
vel_trace.isel(depth=sub).plot(y="depth", yincrease=False, label="smooth", ax=axs[1])
axs[1].step(vel_trace.values[sub], vel_trace.depth[sub], label="steps")
axs[1].legend()


# %% [markdown]
# Each velocity sample has a unique two-way-time (TWT) which is calculated by integrating
# the velocity functions/traces. Now there is an explicit mapping between depth
# - the velocity axis - and TWT the output of the integration.
# The easiest way to do this, is to divide the velocity interval (depth axis step)
# by the associated velocity and then sum the results. The Xarray function to
# perform the sum iteratively along an axis is the cumulative sum or `cumsum`.
# The integration is also scaled from seconds to milliseconds to match the
# data vertical axis and doubled (for two way travel time TWT).
#
# Using the Xarray `cumsum` here is important. As our data is lazily loaded,
# Xarray/Dask have been building a computation graph, this describes the steps
# the data processing will take, but doesn't actually execute any calculations
# until output is required. This minimises the amount of computation needed.
#
# To display the following trace functions, only a small amount of data is loaded
# with subsequent calculations carried out. Any data not needed for visualisation
# is not loaded or converted to depth. That is why its quite fast.
# Ideally, we should use Xarray methods, as they handle the application of
# functions and the computational graph. If we were to apply NumPys `cumsum`
# instead, Dask cannot resolve the computation graph and the delaying of operations
# is broken. As a result, a full computation occurs.
#
# The lesson here, is stay within the Xarray/Dask API if possible.
#

# %%
twt_ds = (2000 * 20 / vel_ds.migintvel).cumsum("depth")

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
twt_trace = twt_ds.sel(iline=10095, xline=2256).compute()
twt_trace.plot(y="depth", yincrease=False, label="smooth", ax=axs[0])
axs[0].step(twt_trace.values, twt_trace.depth, label="steps")
axs[0].set_xlabel("twt (ms)")
axs[0].legend()

sub = slice(100, 150)  # subslice to look at in zoom (right plot)
twt_trace.isel(depth=sub).plot(y="depth", yincrease=False, label="smooth", ax=axs[1])
axs[1].step(twt_trace.values[sub], twt_trace.depth[sub], label="steps")
axs[1].legend()
axs[1].set_xlabel("twt (ms)")

# %% [markdown]
# Give the new twt to depth mapping the variable `twtc` and assign it back to the
# velocity data volume.
# Remember, `twtc` links the depth dimension to a TWT which is required to convert
# the seismic from TWT to depth.
#
# This variable `twtc` again exists in the computation graph, and no calculations
# have yet been carried out. In fact, the data isn't even
# loaded into memory.

# %%
# assign the twt_ds back to vel_ds

vel_ds["twtc"] = twt_ds
vel_ds

# %% [markdown]
# ## Velocity cube interpolation
#
# The velocity volume must now be upsampled (interpolated) from the sparse velocity
# cube to match the seismic data. This will create a trace in the velocity volume
# for each trace in the seismic volume. The method of interpolation can be simple
# like `linear` but we will use `cubic` or spline based interpolation.
# Xarray interpolation uses the `scipy.interpolate` library.

# %% editable=true slideshow={"slide_type": ""}
# interpolate velocities to volve volume
vel_dsi = vel_ds.interp_like(volve_ds.data, method="cubic")
print("Velocity interpolated data dims:", vel_dsi.dims)

# %% [markdown]
# The original (left) and upsampled data (right), velocity (top) and TWT (bottom).
# The vertical axis is depth.

# %% editable=true slideshow={"slide_type": ""}
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
vel_ds.sel(iline=10095).migintvel.T.plot(yincrease=False, ax=axs[0, 0])
vel_dsi.sel(iline=10095).migintvel.T.plot(yincrease=False, ax=axs[0, 1])
vel_ds.sel(iline=10095).twtc.T.plot(yincrease=False, ax=axs[1, 0], cmap="plasma")
vel_dsi.sel(iline=10095).twtc.T.plot(yincrease=False, ax=axs[1, 1], cmap="plasma")

# %% [markdown]
# ## Depth conversion
#
# The first step of depth conversion is to specify the sampling along the depth axis that we wish the seismic output to have. In this case we create a 3m spaced range up to 4500m. The `twtc` cube is interpolated onto this grid to create the TWT samples at which we need to resample the seismic cube. We then add this new spatially upsampled data which has been vertically sub-sampled to 3m to the original seismic Dataset.

# %%
twtc_subsampled_3m = vel_dsi["twtc"].interp(
    coords={"depth": np.arange(0, 4500, 3)}, method="cubic"
)

# add the twt information to the seismic volume dataset
volve_ds["twtc"] = twtc_subsampled_3m

# %% [markdown]
# Depth conversion occurs on a trace-by-trace basis, matching a velocity function (or the `twtc` mapping in this case) to each seismic trace.
# The best way to then apply a depth conversion function is using the `map_blocks` functionality of Xarray. This is a new method that Xarray provides which
# is similar to `group_by` in Pandas. In a nutshell however, it applies a function to a block of your Dataset, defined by its chunking.
# Therefore, we must rechunk the data to ensure that when we apply our trace resampling `trace_resample` that dask applys the process on a trace-by-trace basis.
#
# Alternatively, you chould write a `trace_resample` function which handles arbitrary blocks of seismic data. That would likely be more performant but that is a bit beyond this example.

# %%
# rechunk so be can apply over blocks that are trace (inline/xline) consistent
volve_ds = volve_ds.chunk({"iline": 1, "xline": 1, "twt": -1, "depth": -1})


# %% [markdown]
# The `trace_resample` function will be mapped over each trace using the `xr.map_blocks` function. Each block is represented by a chunk (why we rechunked to traces).
#
# We can test the function by supplying a single traces `.isel(iline=[1], xline=[1])`. The square brackets on the dimension selectors are important, they tell Xarray not to squeeze dimensions
# which is how the data will be provided to our function.


# %%
def trace_resample(trace, var_name, seismic_var="data", out_var="twtc", method="cubic"):
    """Convert a trace from one sample dimension to another.

    Args:
        trace: the trace dataset provided by `xr.map_blocks`
        seismic_var: the variable name of the trace seismic data
        out_var: The variable which maps the dimension to be resampled to the new dimension (e.g. twt -> depth)
        method: The interpolatin method for `interp1d` from scipy.interpolate.
    """
    # do dimension checks on input trace
    in_dims = tuple(trace[seismic_var].squeeze().dims)
    out_dims = tuple(trace[out_var].squeeze().dims)
    # assert len(in_dims) == 1 # The chunks should only be supplied in one dimension
    # assert len(out_dims) == 1 # The chunks should only be supplied in one dimension
    in_dim, out_dim = in_dims[0], out_dims[0]

    trace_z = trace[seismic_var].interp(
        method="cubic", coords={in_dim: trace[out_var].squeeze().values}
    )
    trace_z = trace_z.rename({in_dim: out_dim})
    trace_z[out_dim] = trace[out_dim].values
    trace_z.name = var_name
    return trace_z  # .expand_dims(tuple(out_var_lost_dims)).drop_vars(in_dim[0])


# Test our twt2depth function
test_trace = volve_ds.isel(iline=[1], xline=[1])
tZ = trace_resample(test_trace, "dataZ")

# %% [markdown]
# Lets plot the sample traces to see how the output depth domain trace has been stretched and squeezed. Velocity increases with depth, and therefore the seismic data is stretched significantly the deeper (right end) it is.

# %%
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

test_trace.data.compute().plot(ax=axs[0])
tZ.compute().plot(ax=axs[1])
plt.tight_layout()

# %% [markdown]
# We can lazily apply depth conversion to the whole volume now using the `xr.map_blocks` function.
#
# > `map_blocks` is still in beta, and its arguments may change.

# %% editable=true slideshow={"slide_type": ""}
volve_dsZ = volve_ds.map_blocks(
    trace_resample, args=("dataZ",), template=volve_ds["twtc"]
)
volve_dsZ.name = "data"

# %% [markdown]
# Lets check a single inline to see if our seismic data and velocity data match up after the depth conversion. Looks good, the higher vleocity section in the middle of the depth range lines up with key reflectors in the data!

# %%
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), sharex=True)
twt_plot = volve_ds.sel(iline=10095).data.T.plot(
    yincrease=False, ax=axs[0], cmap="Greys"
)
twt_plot.axes.set_title(f"TWT {twt_plot.axes.get_title()}")
z_plot = volve_dsZ.sel(iline=10095).T.plot(yincrease=False, ax=axs[1], cmap="Greys")
zv_plot = vel_dsi.sel(iline=10095).migintvel.T.plot(
    yincrease=False, ax=axs[1], alpha=0.3
)
z_plot.axes.set_title(f"Depth w/IntVel {z_plot.axes.get_title()}")
# fig.tight_layout()

# %% [markdown]
# ## Saving the depth converted volume
#
# Now, we haven't actually performed depth conversion of the whole cube. To do this we need to use the `to_segy` method, although you could write to any of the other Xarray supported data storage types. `segyio` the library that
# SEGY-SAK uses for writing seismic data works better on contiguous chunks of seismic data. Therefore we are going to rechunk our data to one crossline `xline` at a time. Before we write it to disk.
#
# This is the longest running cell in the notebook. Why? Well, actually, all the functions we lazily applied earlier are being run. So this one cell is responsible for executing all the interpolation, scaling, chunking and resampling that we
# specified before. The process is smart enough, that only the necessary data is read into memory for each step.

# %%
volve_ds["dataZ"] = volve_dsZ.chunk({"xline": -1, "depth": -1})

volve_ds.seisio.to_segy(
    "data/outZ.segy",
    vert_dimension="depth",
    data_var="dataZ",
    trace_header_map={"cdp_x": 181, "cdp_y": 185},
    iline=189,
    xline=193,
)

# %% [markdown]
# Lets check the results.

# %%
ds = xr.open_dataset(
    "data/outZ.segy",
    dim_byte_fields={"iline": 189, "xline": 193},
    extra_byte_fields={"cdpx": 181, "cdpy": 185},
)
ds.sel(iline=10095).data.plot(y="samples", yincrease=False)
