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
#     display_name: Python (segysak)
#     language: python
#     name: segysak
# ---

# %% [markdown]
# # Using `segysak` with `subsurface`.

# %%
from segysak.segy import segy_loader
import subsurface as ss
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# Load in some test data.

# %%
segycube = segy_loader("data/volve10r12-full-twt-sub3d.sgy")

# %%
ss_cube = segycube.seisio.to_subsurface()
# ss_cube is Structured Data in subsurface
display(ss_cube)

# %% [markdown]
# Cast to a `ss.StructuredGrid` and plot using PyVista 

# %%
ss_grid = ss.StructuredGrid(ss_cube)
pvgrid = ss.visualization.to_pyvista_grid(ss_grid, "data")
ss.visualization.pv_plot([pvgrid])
