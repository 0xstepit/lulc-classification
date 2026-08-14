# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: geospatial-misc
#     language: python
#     name: geospatial-misc
# ---

# %% [markdown]
# # Tiled read
#
# In this notebook, we will perform a visual analysis of the windowed read provided by rasterio. With windowed read, it is intended the capability to read only a sub-part of the raster, called a tile, and to iterate through all of them to reduce the space required in RAM to load the entire dataset at once. This feature is particularly useful when we have to operate on multiple scene of the same tile or with many bands at once. By reading only a small chunk of data from the entire raster, we can easily perform computations across multiple bands without bloating RAM.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys

sys.path.append(os.path.abspath(".."))

# %%
from lulc.io import RAW_DATA_DIR

# %%
root, target = next(
    (root, file) for root, _, files in RAW_DATA_DIR.walk() for file in files
)

# %%
all_band_file = root / target
all_band_file

# %%
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# %% [markdown]
# When opening a file with rasterio, we can get the [tiles size for the raster](https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html) with `src.block_shapes`.

# %%
with rasterio.open(all_band_file) as src:
    profile = src.profile
    block_shapes = src.block_shapes
block_shapes

# %% [markdown]
# Since we also know that each scene has 5181 rows and 5169 columns, we can compute the number of tiles in each direction, and whether there will be any skewed tile.

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

# %%
p2 = np.nanpercentile(green, 2)
p98 = np.nanpercentile(green, 98)

# %%
plt.imshow(green, vmin=p2, vmax=p98, cmap="Greens_r")

# %%
n_tiles = 10
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(18, 6))

row, col = 0, 0
with rasterio.open(all_band_file) as src:
    block_size = src.block_shapes[0]
    print(f"size is: {block_size}")

    for ji, window in src.block_windows(1):
        print(f"{ji}: {window}")
        data = src.read(1, window=window)

        if col == 6:
            row += 1

        ax[row, col % 6].imshow(data, vmin=p2, vmax=p98, cmap="Greens_r")
        ax[row, col % 6].axis("off")

        col += 1

        if col == 12:
            break

plt.tight_layout(pad=1)

# %% [markdown]
# As we can see, the last tile on each row is not full since it does not include enough pixels. The same is true also for the entire last row.
