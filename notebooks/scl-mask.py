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
# # SCL mask
#
# This notebook is used to support the development of the SCL mask that we will apply to all bands images to filter out pixels associated to noise.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from dotenv import load_dotenv

from lulc.config import load_config
from lulc.io import GLOBAL_CONFIG, RAW_DATA_DIR

# %%
load_dotenv()
matplotlib.rc_file(matplotlib.matplotlib_fname())
cfg = load_config(GLOBAL_CONFIG)

# %%
counter = 0
for _root, _, files in RAW_DATA_DIR.walk():
    for file in files:
        print(f"{counter}: {file}")
        counter += 1

# %% [markdown]
# We can just select the first file since the workflow will be the same for all images:

# %%
root, target = next(
    (root, file) for root, _, files in RAW_DATA_DIR.walk() for file in files
)

# %%
root, target

# %%
with rasterio.open(root / target, "r") as src:
    data = src.read()

# %%
data.shape

# %%
scl_band_index = cfg.msi.scl_band_index
mask_classes = cfg.msi.scl_mask_classes

# %%
scl = data[scl_band_index]

# %% [markdown]
# Now, we want to print the map by using the same colors defined in the [Sentinel Hub](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/):

# %%
SCL_CLASSES = {
    0: ("No Data", "#000000"),
    1: ("Saturated or defective", "#ff0000"),
    2: ("Topographic shadows", "#2f2f2f"),
    3: ("Cloud shadows", "#643200"),
    4: ("Vegetation", "#00a000"),
    5: ("Not-vegetated", "#ffe65a"),
    6: ("Water", "#0000ff"),
    7: ("Unclassified", "#808080"),
    8: ("Cloud medium probability", "#c0c0c0"),
    9: ("Cloud high probability", "#ffffff"),
    10: ("Thin cirrus", "#64c8ff"),
    11: ("Snow or ice", "#ff96ff"),
}

# Build a listed colormap aligned to class values
n = max(SCL_CLASSES) + 1
colors = [SCL_CLASSES.get(i, ("", "#000000"))[1] for i in range(n)]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries=range(n + 1), ncolors=n)

# %%
fig, axes = plt.subplots(ncols=2, figsize=(14, 8), gridspec_kw={"width_ratios": [3, 1]})

axes[0].imshow(scl, cmap=cmap, norm=norm, interpolation="none")
axes[0].set_title("SCL Scene Classification")
axes[0].axis("off")

patches = [
    mpatches.Patch(facecolor=color, label=f"{v}-{label}", edgecolor="black")
    for v, (label, color) in SCL_CLASSES.items()
]

axes[1].legend(
    handles=patches,
    loc="center",
    frameon=False,
    fontsize=10,
    handlelength=2,
    handleheight=1.5,
    borderpad=1,
)
axes[1].axis("off")

plt.tight_layout()

# %%
mask_classes

# %%
mask = np.isin(scl, mask_classes)

# %%
binary_cmap = mcolors.ListedColormap(["white", "red"])

# %%
fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
fig.suptitle("SCL mask and masked points", y=0.85)

im = ax[0].imshow(scl, cmap=cmap, norm=norm)
ax[0].axis("off")

ax[1].imshow(mask, cmap=binary_cmap, vmin=0, vmax=1, interpolation="nearest")
ax[1].axis("off")

# %% [markdown]
# Create an RGB

# %%
blue = data[0]
green = data[1]
red = data[2]

# Generate the RGB image with channels in the first axis.
rgb = np.stack([red, green, blue], axis=0)

# %%
np.isnan(rgb).sum()

# %%
# Number of True in the mask
mask.sum()

# %%
print(data.shape)  # (C, H, W)
print(mask.shape)

# %% [markdown]
# Since np.nan is valid only for `float` types, we need to cast the array:

# %%
rgb_masked = rgb.copy().astype(float)
rgb_masked[:, mask] = np.nan

# %%
img = np.transpose(rgb_masked, (1, 2, 0))
img = np.maximum(img * 0.0001 - 0.1, 0.0)

# %%
(p2, p98) = np.nanpercentile(img, [2, 98])
img = np.clip((img - p2) / (p98 - p2), 0, 1)

# %%
fig, ax = plt.subplots(ncols=3, figsize=(14, 8))

ax[0].imshow(mask, cmap=binary_cmap, vmin=0, vmax=1, interpolation="nearest")
ax[0].axis("off")
ax[0].set_title("Masked points")
ax[0].set_xticks([])
ax[0].set_yticks([])
bbox = ax[0].get_position()
rect = mpatches.Rectangle(
    (bbox.x0 - 0.01, bbox.y0 - 0.01),
    bbox.width + 0.01,
    bbox.height + 0.01,
    linewidth=1,
    edgecolor="black",
    facecolor="none",
    transform=fig.transFigure,
)
fig.patches.append(rect)

ax[1].imshow(img, vmin=p2, vmax=p98)
ax[1].set_title("Image with interpolation")
ax[1].axis("off")

ax[2].imshow(img, interpolation="none")
ax[2].set_title("Image without interpolation")
ax[2].axis("off")

plt.show()

# %% [markdown]
# How much time did I lost to discover that the issue was just matplotlib visualization that interpolated the values?

# %%
fig, axes = plt.subplots(ncols=4, figsize=(20, 6))

titles = [
    "Band 1 (Blue)",
    "Band 2 (Green)",
    "Band 3 (Red)",
    "RGB Composite",
]

for ax, title in zip(axes, titles, strict=True):
    ax.set_title(title)
    ax.axis("off")

axes[0].imshow(img[:, :, 0], cmap="Blues_r", interpolation="none")
axes[1].imshow(img[:, :, 1], cmap="Greens_r", interpolation="none")
axes[2].imshow(img[:, :, 2], cmap="Reds_r", interpolation="none")
axes[3].imshow(img, interpolation="none")

plt.tight_layout(pad=1.5)
plt.show()
