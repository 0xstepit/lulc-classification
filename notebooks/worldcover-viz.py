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
# # WorldCover viz
#
# This notebook is used to provide a visualizaztion of the raster created with the WorldCover data for the AOI:

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
import os

sys.path.append(os.path.abspath(".."))

# %%
from src.io import WORLDCOVER_LABELS, GLOBAL_CONFIG
import rasterio
from src.config import load_config
from src.viz.images import store_figure
from src.viz.colors import normalize_rgb
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# %%
config = load_config(GLOBAL_CONFIG)

# %%
cn = config.worldcover.class_names
ctc = config.worldcover.class_to_color

# %%
ctc

# %%
cn

# %%
with rasterio.open(WORLDCOVER_LABELS, "r") as src:
    labels = src.read()

# %%
np.unique(labels)

# %%
NUM_CLASSES = len(config.worldcover.class_names)

colors = [normalize_rgb(ctc.get(idx, (0, 0, 0))) for idx in range(NUM_CLASSES)]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=range(NUM_CLASSES+1), ncolors=NUM_CLASSES)

patches = [
    mpatches.Patch(
        facecolor=normalize_rgb(ctc.get(idx, (0, 0, 0))),
        label=f"{cn.get(idx, (""))}",
        edgecolor="black",
        linewidth=1
    ) for idx in range(NUM_CLASSES)
]

# %%
title = f"WorldCover composite {config.aoi.selected.name.title()}"

fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')

ax.imshow(labels[0, :, :], cmap=cmap, norm=norm)

# Suptitle centers on the entire figure
#plt.suptitle(
#    title,
#    fontsize=18,
#    fontweight="bold",
#    color="black"
#)
ax.set_title(
    title,
    fontsize=18,
    fontweight="bold",
    color="black",
    pad=20
)
fig.legend(
# Without the layout=contrained allows to set it below the figure
#    handles=patches, loc="upper center", ncols=NUM_CLASSES / 3, bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9
    handles=patches, ncols=1, loc="outside right center", frameon=False, fontsize=9
)
ax.axis("off")

store_figure(title)

#plt.tight_layout(pad=1.5, h_pad=1, w_pad=1)

# %%
