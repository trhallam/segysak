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
# # Extract an arbitrary line from a 3D volume
#
# Arbitrary lines are often defined as peicewise lines on time/z slices or basemap views that draw a path through features of interest or for example betweem well locations.
#
# By extracting an arbitrary line we hope to end up with a uniformly sampled vertical section of data that traverses the path where the sampling interval is of the order of the bin interval of the dataset
#
#

# %% nbsphinx="hidden"
import warnings
warnings.filterwarnings('ignore')

# %%
from os import path
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Load Small 3D Volume from Volve

# %%
volve_3d_path = path.join("data", "volve10r12-full-twt-sub3d.sgy")
print(f"{volve_3d_path} exists: {path.exists(volve_3d_path)}")

# %%
from segysak.segy import segy_loader, get_segy_texthead, segy_header_scan, segy_header_scrape, well_known_byte_locs

volve_3d = segy_loader(volve_3d_path, **well_known_byte_locs("petrel_3d"))

# %% [markdown]
# ## Arbitrary Lines
#
# We define a line as lists of cdp_x & cdp_y points. These can be inside or outside of the survey, but oviously should intersect it in order to be useful.

# %%
arb_line_A = (np.array([434300, 434600, 435500, 436300]), np.array([6.4786e6, 6.4780e6, 6.4779e6, 6.4781e6]))
arb_line_B = (np.array([434000, 434600, 435500, 436500]), np.array([6.4786e6, 6.4776e6, 6.4786e6, 6.4775e6]))

# %% [markdown]
# Let's see how these lines are placed relative to the survey bounds. We can see *A* is full enclosed whilst *B* has some segments outside.

# %%
ax = volve_3d.seis.plot_bounds()
ax.plot(arb_line_A[0], arb_line_A[1], '.-')
ax.plot(arb_line_B[0], arb_line_B[1], '.-')

# %% [markdown]
# Let's extract line A. 
#
# We specify a `bin_spacing_hint` which is our desired bin spacing along the line. The function use this hint to calculate the closest binspacing that maintains uniform sampling.
#
# We have also specified the `method='linear'`, this is the default but you can specify and method that `DataArray.interp` accepts

# %%
from time import time

tic = time()
line_A = volve_3d.seis.interp_line(arb_line_A[0], arb_line_A[1], bin_spacing_hint=10)
toc = time()
print(f"That took {toc-tic} seconds")

# %%
plt.figure(figsize=(8,8))
plt.imshow(line_A.to_array().squeeze().T, aspect="auto", cmap="RdBu", vmin=-10, vmax=10)

# %%
tic = time()
line_B = volve_3d.seis.interp_line(arb_line_B[0], arb_line_B[1], bin_spacing_hint=10)
toc = time()
print(f"That took {toc-tic} seconds")

# %%
plt.figure(figsize=(10,8))
plt.imshow(line_B.to_array().squeeze().T, aspect="auto", cmap="RdBu", vmin=-10, vmax=10)

# %% [markdown]
# ## Petrel Shapefile
#
# We have an arbitrary line geometry defined over this small survey region stored in a shape file exported from Petrel. 
#
# Let's load that and extract an arbirary line using segysak. We also have the seismic data extracted along that line by Petrel, so we can see how that compares.

# %%
import shapefile
from pprint import pprint

# %% [markdown]
# Load the shapefile and get the list of points

# %%
sf = shapefile.Reader(path.join("data","arbitrary_line.shp"))
line_petrel = sf.shapes() 
print(f"shapes are type {sf.shapeType} == POLYLINEZ")
print(f"There are {len(sf.shapes())} shapes in here")
print(f"The line has {len(sf.shape(0).points)} points")

points_from_shapefile = [list(t) for t in list(zip(*sf.shape(0).points))]
pprint(points_from_shapefile)

# %% [markdown]
# Load the segy containing the line that Petrel extracted along this geometry

# %%
line_extracted_by_petrel_path = path.join("data", "volve10r12-full-twt-arb.sgy")
print(f"{line_extracted_by_petrel_path} exists: {path.exists(line_extracted_by_petrel_path)}")
line_extracted_by_petrel = segy_loader(line_extracted_by_petrel_path, **well_known_byte_locs("petrel_3d"))

# %% [markdown]
# Extract the line using segysak

# %%
tic = time()
line_extracted_by_segysak = volve_3d.seis.interp_line(*points_from_shapefile,
                                        bin_spacing_hint=10,
                                        line_method='linear',
                                        xysel_method='linear')
toc = time()
print(f"That took {toc-tic} seconds")

# %% [markdown]
# Plot the extracted lines side by side

# %%
fig, axs = plt.subplots(1,2, figsize=(16,8))

axs[0].imshow(line_extracted_by_segysak.to_array().squeeze().T, aspect="auto", cmap="RdBu", vmin=-10, vmax=10)
axs[0].set_title("segysak")
axs[1].imshow(line_extracted_by_petrel.to_array().squeeze().T, aspect="auto", cmap="RdBu", vmin=-10, vmax=10)
axs[1].set_title("petrel")

# %% [markdown]
# Plot the geometry, trace header locatons along with the volve 3d bound box, to make sure things line up.

# %%
plt.figure(figsize=(10,10))
ax = volve_3d.seis.plot_bounds(ax=plt.gca())

# plot path
ax.plot(*points_from_shapefile)
ax.scatter(*points_from_shapefile)

# plot trace positons from extracted lines based on header
ax.scatter(line_extracted_by_segysak.cdp_x, line_extracted_by_segysak.cdp_y, marker='x', color='k')
ax.scatter(line_extracted_by_petrel.cdp_x, line_extracted_by_petrel.cdp_y, marker='+', color='r')

ax.set_xlim(434500,435500)
ax.set_ylim(6477800, 6478600)
plt.legend(labels=['bounds','geom','corner','segysak','petrel'])
