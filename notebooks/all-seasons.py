# ---
# jupyter:
#   jupytext:
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
# # All Seasons
#
# In this notebook, we will perform some visualizations and sanity checks on the seasonal composite images.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from dotenv import load_dotenv

from lulc.config import load_config
from lulc.io import GLOBAL_CONFIG, IMGS_DIR, SEASONAL_SCENES
from lulc.viz.images import store_figure

# %%
load_dotenv()
matplotlib.rc_file(matplotlib.matplotlib_fname())
cfg = load_config(GLOBAL_CONFIG)

# %%
IMGS_DIR.mkdir(parents=True, exist_ok=True)

SEASONAL_SCENES

# %%
with rasterio.open(SEASONAL_SCENES[0], "r") as src:
    profile = src.profile
profile

# %%
print(f"Band count: {profile['count']}")
print(f"Width x heigth: {profile['width']} x {profile['height']}")
print(f"CRS: {profile['crs']}")
print(f"Is tiled: {profile['tiled']}")

# %%
band_per_season = 13
rgb_idx = np.array([1, 2, 3])
indices_idx = np.array([11, 12, 13])
num_seasons = 4

# %%
# Accumulate the rasters of RGB for each season.
rgb_per_season = []
bands = [1, 2, 3]
dim_transposition = (1, 2, 0)

for season_file in SEASONAL_SCENES:
    with rasterio.open(season_file, "r") as src:
        rgb = np.transpose(src.read(bands), dim_transposition)
        rgb = rgb[:, :, ::-1]
        rgb = rgb / 10_000.0
        rgb = np.clip(rgb, 0, 1)
        rgb_per_season.append(rgb)

# %%
all_data = np.stack(rgb_per_season)  # shape (4, H, W, 3)

p2 = np.nanpercentile(all_data, 2)
p98 = np.nanpercentile(all_data, 98)

rgb_per_season_stretched = [
    np.clip((rgb - p2) / (p98 - p2), 0, 1) for rgb in rgb_per_season
]

# %%
all_data.shape

# %%
print(p2, p98)

# %%
title = f"Seasonal median composites {cfg.aoi.selected.name.title()}"
subtitles = ["DJF", "MAM", "JJA", "SON"]

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
ax = ax.flatten()
for i, _rgb in enumerate(rgb_per_season_stretched):
    ax[i].imshow(_rgb, vmin=p2, vmax=p98)
    ax[i].set_title(subtitles[i], color="black", fontsize=13, pad=8)
    ax[i].axis("off")
plt.suptitle(
    title,
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout(pad=1.5, h_pad=1, w_pad=0.1)

store_figure(title)

# %%
# Accumulate the rasters of computed indices for each season.
indices_per_season = []
indices_bands = [11, 12, 13]

for season_file in SEASONAL_SCENES:
    with rasterio.open(season_file, "r") as src:
        indices = src.read(indices_bands)
        indices = np.ma.clip(indices, -1, 1)
        indices_per_season.append(indices)

# %%
title = f"NDVI seasonal median composite {cfg.aoi.selected.name.title()}"
subtitles = ["DJF", "MAM", "JJA", "SON"]

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), facecolor="white")
ax = ax.flatten()

for i, _indices in enumerate(indices_per_season):
    ndvi = _indices[0, :, :]
    im = ax[i].imshow(ndvi, vmin=-1, vmax=1, cmap="RdYlGn")
    ax[i].set_title(subtitles[i], color="black", fontsize=13, pad=8)
    ax[i].axis("off")
    cbar = fig.colorbar(im, ax=ax[i], shrink=0.7, orientation="vertical")
    cbar.set_label("NDVI", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

plt.suptitle(title, fontsize=15, fontweight="bold", color="black")

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1)

store_figure(title)


# %%
def compute_percentiles(arrays, index):
    arrs = []
    for _, arr in enumerate(arrays):
        arrs.append(arr[index, :, :])
    arrs = np.stack(arrs)
    return np.nanpercentile(arrs, [2, 98])


# %%
title = f"NDVI seasonal median composite {cfg.aoi.selected.name.title()}"
subtitles = ["DJF", "MAM", "JJA", "SON"]
p2, p98 = compute_percentiles(indices_per_season, 0)

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), facecolor="white")
ax = ax.flatten()

for i, _indices in enumerate(indices_per_season):
    ndvi = _indices[0, :, :]
    im = ax[i].imshow(ndvi, vmin=p2, vmax=p98, cmap="RdYlGn")
    ax[i].set_title(subtitles[i], color="black", fontsize=13, pad=8)
    ax[i].axis("off")
    cbar = fig.colorbar(im, ax=ax[i], shrink=0.7, orientation="vertical")
    cbar.set_label("NDVI", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

plt.suptitle(title, fontsize=15, fontweight="bold", color="black")

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1)

# store_figure(title)

# %%
title = f"NDWI seasonal median composite {cfg.aoi.selected.name.title()}"
subtitles = ["DJF", "MAM", "JJA", "SON"]

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), facecolor="white")
ax = ax.flatten()

for i, _indices in enumerate(indices_per_season):
    ndvi = _indices[2, :, :]
    im = ax[i].imshow(ndvi, vmin=-1, vmax=1, cmap="RdBu")
    ax[i].set_title(subtitles[i], color="black", fontsize=13, pad=8)
    ax[i].axis("off")
    cbar = fig.colorbar(im, ax=ax[i], shrink=0.7, orientation="vertical")
    cbar.set_label("NDWI", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

plt.suptitle(title, fontsize=15, fontweight="bold", color="black")

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1)

store_figure(title)

# %%
title = f"NDWI seasonal median composite {cfg.aoi.selected.name.title()}"
subtitles = ["DJF", "MAM", "JJA", "SON"]
p2, p98 = compute_percentiles(indices_per_season, 2)

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), facecolor="white")
ax = ax.flatten()

for i, _indices in enumerate(indices_per_season):
    ndvi = _indices[2, :, :]
    im = ax[i].imshow(ndvi, vmin=p2, vmax=p98, cmap="RdBu")
    ax[i].set_title(subtitles[i], color="black", fontsize=13, pad=8)
    ax[i].axis("off")
    cbar = fig.colorbar(im, ax=ax[i], shrink=0.7, orientation="vertical")
    cbar.set_label("NDWI", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

plt.suptitle(title, fontsize=15, fontweight="bold", color="black")

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1)

# store_figure(title)

# %%
title = f"NDBI seasonal median composite {cfg.aoi.selected.name.title()}"
subtitles = ["DJF", "MAM", "JJA", "SON"]

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), facecolor="white")
ax = ax.flatten()

for i, _indices in enumerate(indices_per_season):
    ndvi = _indices[1, :, :]
    im = ax[i].imshow(ndvi, vmin=-1, vmax=1, cmap="RdGy_r")
    ax[i].set_title(subtitles[i], color="black", fontsize=13, pad=8)
    ax[i].axis("off")
    cbar = fig.colorbar(im, ax=ax[i], shrink=0.7, orientation="vertical")
    cbar.set_label("NDBI", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

plt.suptitle(title, fontsize=15, fontweight="bold", color="black")

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1)

store_figure(title)

# %%
title = f"NDBI seasonal median composite {cfg.aoi.selected.name.title()}"
subtitles = ["DJF", "MAM", "JJA", "SON"]
p2, p98 = compute_percentiles(indices_per_season, 1)

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), facecolor="white")
ax = ax.flatten()

for i, _indices in enumerate(indices_per_season):
    ndvi = _indices[2, :, :]
    im = ax[i].imshow(ndvi, vmin=p2, vmax=p98, cmap="RdGy_r")
    ax[i].set_title(subtitles[i], color="black", fontsize=13, pad=8)
    ax[i].axis("off")
    cbar = fig.colorbar(im, ax=ax[i], shrink=0.7, orientation="vertical")
    cbar.set_label("NDBI", color="black")
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

plt.suptitle(title, fontsize=15, fontweight="bold", color="black")

plt.tight_layout(pad=1.5, h_pad=1, w_pad=1)

# store_figure(title)
