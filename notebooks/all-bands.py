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
# # All bands
#
# This notebook is used to get a sense of the downloaded and aggregated bands for each scene over the AOI

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from dotenv import load_dotenv

from lulc.config import load_config
from lulc.io import GLOBAL_CONFIG, RAW_DATA_DIR
from lulc.viz.images import robust_plot

# %%
load_dotenv()
matplotlib.rc_file(matplotlib.matplotlib_fname())
cfg = load_config(GLOBAL_CONFIG)

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
robust_plot(data[9], bounds)
plt.title("NIR band Andalusia (Spain)")
plt.xlabel("longitude [m]")
plt.ylabel("latitude [m]")

# %%
cfg.msi.band_names


# %%
def plot_bands(data, bands, bounds=None):
    n = data.shape[0]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    _, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
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
