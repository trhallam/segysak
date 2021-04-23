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


# %% [markdown] jupyter={"source_hidden": true}
# # SEG-Y to Vector DataFrames and Back
#
# The connection of segysak to `xarray` greatly simplifies the process of vectorising segy 3D data and returning it to SEGY. To do this, one can use the close relationship between `pandas` and `xarray`.
#
# ## Loading Data
#
# We start by loading data normally using the `segy_loader` utility. For this example we will use the Volve example sub-cube.

# %%
import pathlib
from IPython.display import display
from segysak.segy import segy_loader, well_known_byte_locs, segy_writer

volve_3d_path = pathlib.Path("data/volve10r12-full-twt-sub3d.sgy")
print("3D", volve_3d_path.exists())

volve_3d = segy_loader(volve_3d_path, **well_known_byte_locs("petrel_3d"))

# %% [markdown]
# ## Vectorisation
#
# Once the data is loaded it can be converted to a `pandas.DataFrame` directly from the loaded `Dataset`. The Dataframe is multi-index and contains columns for each variable in the originally loaded dataset. This includes the seismic amplitude as `data` and the `cdp_x` and `cdp_y` locations. If you require smaller volumes from the input data, you can use xarray selection methods prior to conversion to a DataFrame.

# %%
volve_3d_df = volve_3d.to_dataframe()
display(volve_3d_df)

# %% [markdown]
# We can remove the multi-index by resetting the index of the DataFrame. Vectorized workflows such as machine learning can then be easily applied to the DataFrame.

# %%
volve_3d_df_reindex = volve_3d_df.reset_index()
display(volve_3d_df_reindex)

# %% [markdown]
# ## Return to Xarray
#
# It is possible to return the DataFrame to the Dataset for output to SEGY. To do this the multi-index must be reset. Afterward, `pandas` provides the `to_xarray` method.

# %%
volve_3d_df_multi = volve_3d_df_reindex.set_index(["iline", "xline", "twt"])
display(volve_3d_df_multi)
volve_3d_ds = volve_3d_df_multi.to_xarray()
display(volve_3d_ds)

# %% [markdown]
# The resulting dataset requires some changes to make it compatible again for export to SEGY.
# Firstly, the attributes need to be set. The simplest way is to copy these from the original SEG-Y input. Otherwise they can be set manually. `segysak` specifically needs the `sample_rate` and the `coord_scalar` attributes.

# %%
volve_3d_ds.attrs = volve_3d.attrs
display(volve_3d_ds.attrs)

# %% [markdown]
# The `cdp_x` and `cdp_y` positions must be reduced to 2D along the vertical axis "twt" and set as coordinates.

# %%
volve_3d_ds["cdp_x"] = volve_3d_ds["cdp_x"].mean(dim=["twt"])
volve_3d_ds["cdp_y"] = volve_3d_ds["cdp_y"].mean(dim=["twt"])
volve_3d_ds = volve_3d_ds.set_coords(["cdp_x", "cdp_y"])
volve_3d_ds

# %% [markdown]
# Afterwards, use the `segy_writer` utility as normal to return to SEGY.

# %%
segy_writer(volve_3d_ds, "test.segy")
