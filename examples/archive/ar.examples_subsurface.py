# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Using `segysak` with `subsurface`
#
# **The interface from `segysak` to [`subsurface`](https://github.com/softwareunderground/subsurface) is experimental and currently under active development. This example will be updated to reflect current functionality and compatability with `subsurface`.**
#
# Subsurface is a library which provides data structures to facilitate communication of data efficiently between the various packages of the geoscience and subsurface Python stack. Subsurface uses Xarray data structures internally which have a defined format and interface enabling data to be sent and received between libraries.
#
# The subsurface interchange is being built throught the `seisio` accessor in `segysak`.

# %%
from segysak.segy import segy_loader
from IPython.display import display
import subsurface as ss
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# Load in some test data.

# %%
segycube = segy_loader("data/volve10r12-full-twt-sub3d.sgy")

# %% [markdown]
# A 3D SEGY-SAK Xarray Dataset can be converted to a Subsurface `StructuredGrid` using the `to_subsurface()` method.

# %%
ss_cube = segycube.seisio.to_subsurface()
# ss_cube is Structured Data in subsurface
display(ss_cube)

# %% [markdown]
# Cast to a `ss.StructuredGrid` and plot using PyVista

# %%
ss_grid = ss.StructuredGrid(ss_cube)
pvgrid = ss.visualization.to_pyvista_grid(ss_grid, "data")
_ = ss.visualization.pv_plot([pvgrid])
