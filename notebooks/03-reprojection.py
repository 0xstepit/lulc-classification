# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: lulc-classification-with-unet
#     language: python
#     name: lulc-classification-with-unet
# ---

# %% [markdown]
# # Reprojection
#
# This notebook investigate how to reproject and combine the categorical rasters from the WorldCover dataset to match the composite image we created for the area of interest.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import pathlib
import sys

sys.path.append(os.path.abspath(".."))

import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
import numpy as np
import xarray as xr
from sklearn.metrics import confusion_matrix

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.io import WORLDCOVER_RAW_DIR, MULTISEASONAL_SCENE_DIR
from src.data.sentinel2 import get_data_profile

# %% [markdown]
# Get the file paths of two `.tif` images associated with the WorldCover dataset:

# %%
files = list(WORLDCOVER_RAW_DIR.glob("*.tif"))[:2]

# %%
list(files)

# %% [markdown]
# Now we get the profile information from the composite image we will use as the feture raster:

# %%
profile = get_data_profile(str(MULTISEASONAL_SCENE_DIR/"composite.tif"))
profile

# %%
file_N36W009 = rxr.open_rasterio(files[0])
file_N36W009

# %%
type(file_N36W009.variable._data), file_N36W009.chunks

# %% [markdown]
# This means that the array is not loaded yet, but will be loaded at the first access. Since the size of the array is pretty big, we can prefer to downsample it before visualizing the data. The resampling method we will use is nearest neighborhood because the data is categorical:

# %%
factor = 1 / 5
new_width = int(file_N36W009.rio.width * factor)
new_height = int(file_N36W009.rio.height * factor)

file_N36W009_downsampled = file_N36W009.rio.reproject(
    file_N36W009.rio.crs,
    shape=(new_height, new_width),
    resampling=Resampling.nearest,
)

# %%
file_N36W009_downsampled

# %% [markdown]
# We can verify we still have only the values used by the WorldCover dataset:

# %%
np.unique(file_N36W009_downsampled)

# %% [markdown]
# We can also load the dask-backed xarray:

# %%
file_N36W006 = rxr.open_rasterio(files[1], chunks=True, lock=False)

# %%
type(file_N36W006.variable._data), file_N36W006.chunks

# %% [markdown]
# ## Reproject on composite
#
# We can now investigate how to reproject the WorldCover categorical tiles on the specific area of interest we care about defined by the multi seasonal composite raster:

# %%
composite = rxr.open_rasterio(MULTISEASONAL_SCENE_DIR / "composite.tif", chunks=True, lock=False)

# %%
composite.rio.crs

# %%
composite.rio.bounds(), print(composite.rio.crs)

# %%
file_N36W009.rio.bounds()

# %%
file_N36W009_reprojected = file_N36W009.rio.reproject_match(composite)

# %%
file_N36W009_reprojected.rio.bounds()

# %% [markdown]
# Notice that the WorldCover tile was not covering the entire tile, but despite that, the reprojected array is now covering the full tile.

# %%
file_N36W006_reprojected = file_N36W006.rio.reproject_match(composite)

# %%
file_N36W006_reprojected.rio.bounds()

# %% [markdown]
# We can finally combine the two reprojected raster into a single raster:

# %%
wc = xr.where(file_N36W009_reprojected != 0, file_N36W009_reprojected, file_N36W006_reprojected)

# %%
wc

# %% [markdown]
# We didn't use the `merge_array` function from `rioxarray` because the two raster were defined over the same bounding box.

# %% [markdown]
# ## Clip box
#
# We can also adopt the first clip and then reproject approach. This way, we will be able to merge the WorldCover tiles, making the process easier if there are more than 2 rasters, since using the `where` method would not be that easy in this case.

# %%
file_N36W006_clipped = file_N36W006.rio.reproject(composite.rio.crs).rio.clip_box(*composite.rio.bounds())

# %%
file_N36W006_clipped.rio.bounds()

# %%
file_N36W009_clipped = file_N36W009.rio.reproject(composite.rio.crs).rio.clip_box(*composite.rio.bounds())

# %% [markdown]
# The merge order is important like for the `where` method.

# %%
wc_merged = merge_arrays([file_N36W009_clipped, file_N36W006_clipped])

# %%
wc_merged = wc_merged.rio.reproject_match(composite)

# %%
wc_merged

# %%
np.sum(wc_merged != wc).values, np.sum(wc_merged == wc).values, wc_merged.size == wc.size

# %%
np.sum(wc_merged != wc).values / wc_merged.size * 100

# %% [markdown]
# The issue with this strategy is that we resampled twice, once for the `reproject` and once for the `reproject_match`. A better strategy is:

# %%
wc_merged_2 = merge_arrays([file_N36W009, file_N36W006])
wc_merged_2 = wc_merged_2.rio.reproject_match(composite)

# %%
np.sum(wc_merged_2 != wc).values / wc_merged_2.size * 100

# %% [markdown]
# ### Classes confusion
#
# Let's see the confusion matrix between the two approaches to evaluate their consistency.

# %%
WORLDCOVER_CLASSES = {
    0: ("No Data", (255, 255, 255)),
    1: ("Tree cover", (0,100,0)),
    2: ("Shrubland", (255, 187, 34)),
    3: ("Grassland", (255, 255, 76)),
    4: ("Cropland", (240, 150, 255)),
    5: ("Build-up", (250, 0, 0)),
    6: ("Bare/sparse vegetation", (180, 180, 180)),
    7: ("Snow and ice", (240, 240, 240)),
    8: ("Permanent water bodies", (0, 100, 200)),
    9: ("Herbaceous wetland", (0, 150, 160)),
    10: ("Mangroves", (0, 207, 117)),
    11: ("Moss and lichen", (250, 230, 160)),
}

# %%
wc_1 = wc_reclassed.values.ravel()
wc_2 = wc_merged_reclassed.values.ravel()

# %%
present_classes = sorted(np.unique(np.concatenate([wc_1, wc_2])).tolist())
labels = [WORLDCOVER_CLASSES[i][0] for i in present_classes]

# %%
cm = confusion_matrix(wc_1, wc_2, labels=present_classes)

# %%
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues", norm="log")

ax.set_xticks(range(len(present_classes)), labels, rotation=45, ha="right")
ax.set_yticks(range(len(present_classes)), labels)
ax.set_xlabel("Approach 2 (clip -> merge -> reprtoject_match)")
ax.set_ylabel("Approach 1 (reproject_match -> where)")
ax.set_title("Approach 1 vs 2 agreement")

for i in range(len(present_classes)):
    for j in range(len(present_classes)):
        if cm[i, j]:
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8, color="black")
fig.colorbar(im, ax=ax, shrink=0.8)

# %% [markdown]
# So, given the percentage we computed before, the sum on the diagonal accounts for $95\%$ of the pixels, while on the off-diagonal, we have the $5\%$. Since the disagreement should come from the nearest neighbor approach, we should have high values in the off-diagonal for spatially adjacent pairs.
#
# The values could be more informative by reporting the percentage of each class but I'm lazy...
#
# We can also display the disagreement map:

# %%
disagree = (wc != wc_merged).squeeze()

# %%
disagree.sum()

# %%
fig, ax, = plt.subplots(figsize=(8, 8))
ax.imshow(disagree, cmap="Reds", interpolation="nearest")
ax.set_title(f"A vs B disagreement - {float(disagree.mean()) * 100:.2f}% of pixels")
ax.axis("off")


# %% [markdown]
# ## Visualization
#
# We will now visualize the WorldCover dataset covering our AOI by using the same colors used in their website:

# %%
def normalize_rgb(rgb):
    return tuple([x / 255 for x in rgb])


# %%
n = len(WORLDCOVER_CLASSES.items())
colors = [normalize_rgb(WORLDCOVER_CLASSES.get(i, ("", (0, 0, 0)))[1]) for i in range(n)]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=range(n + 1), ncolors=n)
patches = [
    mpatches.Patch(color=normalize_rgb(color), label=f"{_label}") for (_label, color) in WORLDCOVER_CLASSES.values()
]

# %%
CLASS_MAPPING = { 0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7, 80: 8, 90: 9, 95: 10, 100: 11 }

# %% [markdown]
# We first need to change the classes used by WorldCover into sequential class numbers starting from 0 to 11:

# %%
wc_reclassed = xr.zeros_like(wc)
for k, v in CLASS_MAPPING.items():
    wc_reclassed = xr.where(wc == k, v, wc_reclassed)

wc_merged_reclassed = xr.zeros_like(wc_merged)
for k, v in CLASS_MAPPING.items():
    wc_merged_reclassed = xr.where(wc_merged == k, v, wc_merged_reclassed)

# %%
np.unique(wc_reclassed), len(np.unique(wc_reclassed))

# %%
np.unique(wc_merged_reclassed), len(np.unique(wc_merged_reclassed))

# %% [markdown]
# Our area of interest does not contain all the classes.

# %%
fig, ax = plt.subplots(ncols=2, figsize=(14, 6),  gridspec_kw={"width_ratios": [3, 2]})

wc_reclassed.plot(ax=ax[0], cmap=cmap, norm=norm)
ax[0].set_title("WorldCover raster over AOI", fontweight="bold")

ax[1].legend(
    handles=patches,
    loc="center",
    frameon=True,
    fontsize=10,
    handlelength=2,
    handleheight=1.5,
    borderpad=1,
)
ax[1].axis("off")

# %%
_wc = rxr.open_rasterio("../data/labels/worldcover_classes.tif")

# %%
fig, ax = plt.subplots(ncols=2, figsize=(14, 6),  gridspec_kw={"width_ratios": [3, 2]})

_wc.plot(ax=ax[0], cmap=cmap, norm=norm)
ax[0].set_title("WorldCover raster over AOI", fontweight="bold")

ax[1].legend(
    handles=patches,
    loc="center",
    frameon=True,
    fontsize=10,
    handlelength=2,
    handleheight=1.5,
    borderpad=1,
)
ax[1].axis("off")

# %%
np.unique(_wc)

# %%
