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
# # SEG-Y to Vector DataFrames and Back
#
# The connection of segysak to `xarray` greatly simplifies the process of vectorising segy 3D data and returning it to SEGY. To do this, one can use the close relationship between `pandas` and `xarray`.
#
# ## Loading Data
#
# We start by loading data normally using the `segy_loader` utility. For this example we will use the Volve example sub-cube.

# %% editable=true slideshow={"slide_type": ""} tags=["hide-code"]
# Disable progress bars for small examples
from segysak.progress import Progress
Progress.set_defaults(disable=True)

# %% editable=true slideshow={"slide_type": ""}
import pathlib
import xarray as xr
from IPython.display import display

volve_3d_path = pathlib.Path("data/volve10r12-full-twt-sub3d.sgy")
print("3D", volve_3d_path.exists())

volve_3d = xr.open_dataset(volve_3d_path, dim_byte_fields={'iline': 5, 'xline': 21}, extra_byte_fields={'cdp_x': 73, 'cdp_y': 77})

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Vectorisation
#
# Once the data is loaded it can be converted to a `pandas.DataFrame` directly from the loaded `Dataset`. The Dataframe is multi-index and contains columns for each variable in the originally loaded dataset. This includes the seismic amplitude as `data` and the `cdp_x` and `cdp_y` locations. If you require smaller volumes from the input data, you can use xarray selection methods prior to conversion to a DataFrame.

# %% editable=true slideshow={"slide_type": ""}
volve_3d_df = volve_3d.to_dataframe()
display(volve_3d_df)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# We can remove the multi-index by resetting the index of the DataFrame. Vectorized workflows such as machine learning can then be easily applied to the DataFrame.

# %% editable=true slideshow={"slide_type": ""}
volve_3d_df_reindex = volve_3d_df.reset_index()
display(volve_3d_df_reindex)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Return to Xarray
#
# It is possible to return the DataFrame to the Dataset for output to SEGY. To do this the multi-index must be reset. Afterward, `pandas` provides the `to_xarray` method.

# %% editable=true slideshow={"slide_type": ""}
volve_3d_df_multi = volve_3d_df_reindex.set_index(["iline", "xline", "samples"])
display(volve_3d_df_multi)
volve_3d_ds = volve_3d_df_multi.to_xarray()
display(volve_3d_ds)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The resulting dataset requires some changes to make it compatible again for export to SEGY.
# Firstly, the attributes need to be set. The simplest way is to copy these from the original SEG-Y input. Otherwise they can be set manually. `segysak` specifically needs the `sample_rate` and the `coord_scalar` attributes.

# %% editable=true slideshow={"slide_type": ""}
volve_3d_ds.attrs = volve_3d.attrs
display(volve_3d_ds.attrs)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# The `cdp_x` and `cdp_y` positions must be reduced to 2D along the vertical axis "twt" and set as coordinates.

# %% editable=true slideshow={"slide_type": ""}
volve_3d_ds["cdp_x"]

# %% editable=true slideshow={"slide_type": ""}
volve_3d_ds["cdp_x"] = volve_3d_ds["cdp_x"].mean(dim=["samples"])
volve_3d_ds["cdp_y"] = volve_3d_ds["cdp_y"].mean(dim=["samples"])
volve_3d_ds = volve_3d_ds.set_coords(["cdp_x", "cdp_y"])
volve_3d_ds

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Afterwards, use the `to_segy` method as normal to return to SEGY.

# %% editable=true slideshow={"slide_type": ""}
volve_3d_ds.seisio.to_segy("data/test.segy", iline=189, xline=193, trace_header_map={'cdp_x':181, 'cdp_y':185})

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Very large datasets
#
# If you have a very large dataset (SEG-Y file), it may be possible to use `ds.to_dask_dataframe()` which can perform operations, including the writing of data in a lazy manner.
