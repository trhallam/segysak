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
# # Reading and Writing ZGY
#
# SEGY-SAK support readying and writing ZGY files via the [open-zgy](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/seismic/open-zgy). Currently `open-zgy` is not available via pip and is baked into the `SEGY-SAK` release, so it may not be the latest version.
#
# ZGY support is experimental and functionality may change.

# %% [markdown]
# ## Converting SEG-Y to ZGY
#
# ZGY conversion is currently only support via in-memory operations. A dataset must be fully loaded before it can be exported to ZGY.

# %%
from segysak.segy import segy_loader

ds = segy_loader("data/volve10r12-full-twt-sub3d.sgy")

# %%
# import ZGY comes with warnings
from segysak.openzgy import zgy_writer

zgy_writer(ds, "data/volve10r12-full-twt-sub3d.zgy")

# %% [markdown]
# ## Reading ZGY
#
# ZGY files can be read back into a dataset by using the `zgy_loader` function. Because all the information containing geometry is saved in ZGY metadata, no scanning or additional arguments other than the file name are required.

# %%
from segysak.openzgy import zgy_loader

ds_zgy = zgy_loader("data/volve10r12-full-twt-sub3d.zgy")

# %% [markdown]
# ## Results

# %% [markdown]
# Check that the data looks the same.

# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 12))

ds.data.sel(twt=400).plot(ax=axs[0, 0])
axs[0, 0].set_title("Loaded from SEGY: " + axs[0, 0].title.get_text())
ds_zgy.data.sel(twt=400).plot(ax=axs[0, 1])
axs[0, 1].set_title("Loaded from ZGY: " + axs[0, 1].title.get_text())

ds.data.sel(xline=2250).T.plot(ax=axs[1, 0])
axs[1, 0].set_title("Loaded from SEGY: " + axs[1, 0].title.get_text())
ds_zgy.data.sel(xline=2250).T.plot(ax=axs[1, 1])
axs[1, 1].set_title("Loaded from ZGY: " + axs[1, 1].title.get_text())


# %% [markdown]
# Check that the global coordinates are the same.

# %% tags=[]
fig, axs = plt.subplots(ncols=2, figsize=(15, 6))

p1 = axs[0].pcolormesh(
    ds.sel(twt=400).cdp_x,
    ds.sel(twt=400).cdp_y,
    ds.sel(twt=400).data,
    cmap="seismic",
    vmin=-5,
    vmax=5,
    shading="auto",
)
plt.colorbar(p1, ax=axs[0], label="Seismic Amplitude")
axs[0].set_title("Loaded from SEGY")

p2 = axs[1].pcolormesh(
    ds_zgy.sel(twt=400).cdp_x.values,
    ds_zgy.sel(twt=400).cdp_y.values,
    ds_zgy.sel(twt=400).data.values,
    cmap="seismic",
    vmin=-5,
    vmax=5,
    shading="auto",
)
plt.colorbar(p1, ax=axs[1], label="Seismic Amplitude")
axs[1].set_title("Loaded from ZGY")

# %%
