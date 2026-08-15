# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: lulc-classification
#     language: python
#     name: lulc-classification
# ---

# %% [markdown]
# # Tiled read
#
# In this notebook, we will perform a visual analysis of the windowed read provided by rasterio. With a windowed read, the intention is to read only a sub-part of the raster, called a tile, and to iterate through all of them in sequence to reduce the space required in RAM compared to loading the entire dataset at once.
#
# This feature is particularly useful when we have to operate on multiple scenes of the same tile or with many bands at once. By reading only a small chunk of data from the entire raster, we can easily perform computations across multiple bands without bloating RAM.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from dotenv import load_dotenv

from lulc.io import RAW_DATA_DIR
from lulc.viz.images import robust_plot

# %%
load_dotenv()
matplotlib.rc_file(matplotlib.matplotlib_fname())

# %%
root, target = next(
    (root, file) for root, _, files in RAW_DATA_DIR.walk() for file in files
)

# %%
all_band_file = root / target
all_band_file

# %% [markdown]
# When opening a file with rasterio, we can get the [tiles size for the raster](https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html) used in the internal block storage layout with `src.block_shapes`.

# %%
with rasterio.open(all_band_file) as src:
    profile = src.profile
    block_shapes = src.block_shapes
    block_windows = src.block_windows
block_shapes

# %% [markdown]
# Notice that each tuple in the list is associated to 1 bands, so blocks have the same shape across a band.
#
# Since we also know that each scene has 5181 rows and 5169 columns, we can compute the number of tiles in each direction, and whether there will be any skewed tile or not.

# %%
print(f"For columns direction: {5169 // 1024, 5169 % 1024}")
print(f"For rows direction: {5181 // 1024, 5181 % 1024}")

# %% [markdown]
# So, considering the columns direction, we will have 5 full tiles and 1 tile with only 49 pixels.
#
# The information about the window block size is controlled with the rasterio Profile properties `blockxsize` and `blockysize`:

# %%
profile["blockxsize"], profile["blockysize"]

# %% [markdown]
# We can investigate how to use windowed read by considering the Green band of our raster:

# %%
with rasterio.open(all_band_file) as src:
    print(type(src))
    green = src.read(1)
    bounds = src.bounds

# %%
robust_plot(green, bounds)
plt.title("Green band Andalusia (Spain)")
plt.xlabel("longitude [m]")
plt.ylabel("latitude [m]")

# %% [markdown]
# We can now recreate the same image block by block:

# %%
(p2, p98) = np.nanpercentile(green, [2, 98])

# %%
13 // 6

# %%
n_tiles = 10
fig, ax = plt.subplots(
    ncols=6,
    nrows=6,
    figsize=(18, 18),
)

row, col = 0, 0
windows_info = []
with rasterio.open(all_band_file) as src:
    block_size = src.block_shapes[0]
    windows_info.append(f"size is: {block_size}")

    for ji, window in src.block_windows(1):
        windows_info.append(f"{ji}: {window}")
        data = src.read(1, window=window)
        ax[col // 6, col % 6].imshow(data, vmin=p2, vmax=p98, cmap="gray")
        ax[col // 6, col % 6].axis("off")
        col += 1

plt.suptitle("Block read Andalusia (Spain)", y=1.05)
plt.tight_layout(pad=1)

# %% [markdown]
# As we can see, the last tile on each row is not full since it does not include enough pixels. The same is true also for the entire last row.

# %%
print("\n".join(windows_info))
