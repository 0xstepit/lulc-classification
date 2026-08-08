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
# # All bands viz
#
# This notebook is used to get a sense of the downloaded and aggregated bands for each scene over the AOI

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys

sys.path.append(os.path.abspath(".."))

from src.io import RAW_DATA_DIR

# %%
file_name = (
    RAW_DATA_DIR
    / "DJF/S2B_MSIL2A_20220220T110959_N0510_R137_T30STG_20240516T171047_ALLBANDS.tif"
)

# %%
file_name

# %%
# !gdalinfo -stats {file_name}

# %%
import matplotlib.pyplot as plt
import numpy as np
import rasterio

# %%
with rasterio.open(file_name, "r") as src:
    data = src.read()
    profile = src.profile
    bounds = src.bounds

# %%
data.shape

# %%
profile

# %%
profile["crs"], profile["transform"]


# %%
def robust_plot(data, bounds: list | None = None) -> None:
    """
    Plot the provided data with the Viridis cmap and cmap max and min given
    by the 2nd and 98th percentiles of the data.
    """
    extent = []
    if bounds is not None:
        left, bottom, right, top = bounds
        extent = [left, right, bottom, top]
    plt.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        cmap="viridis",
    )


# %%
robust_plot(data[9], bounds)

# %%
from src.config import load_config

cfg = load_config()

# %%
band_names = [band.split("_")[0] for band in cfg.msi.get_bands()]
band_names

# %%
import math


# %%
def plot_bands(data, bands, bounds=None):
    n, h, w = data.shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top] if bounds else None

    for i, ax in enumerate(axes):
        if i >= n:
            ax.set_visible(False)
            continue
        band = data[i].astype(float)
        lo, hi = np.percentile(band[band > 0], [2, 98])
        ax.imshow(
            np.clip((band - lo) / (hi - lo), 0, 1),
            extent=extent,
            origin="upper",
            cmap="gray",
        )
        ax.set_title(f"Band {bands[i]}", fontsize=9)

    plt.tight_layout()


# %%
plot_bands(data, band_names, bounds)
