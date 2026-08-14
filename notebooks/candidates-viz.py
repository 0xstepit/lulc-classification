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
# ## Areas of interest
#
# Simple visualization of the candidate area of interest in a World map.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import sys

sys.path.append(os.path.abspath(".."))

# %%
import geodatasets
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

from lulc.config import load_config
from lulc.config.viz import load_viz_config
from lulc.io import GLOBAL_CONFIG, VIZ_CONFIG
from lulc.viz.images import set_matplotlib_global_config, store_figure

# %%
cfg = load_config(GLOBAL_CONFIG)

# %%
viz_cfg = load_viz_config(VIZ_CONFIG)
set_matplotlib_global_config(viz_cfg)

# %%
aois = cfg.aoi.candidates
aois

# %%
world = gpd.read_file(geodatasets.get_path("naturalearth land"))

# %%
aoi_points = {
    "points": {},
    "lon": [],
    "lat": [],
}
for name, coords in aois.items():
    p = Point(coords)
    aoi_points["points"][name] = p
    aoi_points["lon"].append(p.x)
    aoi_points["lat"].append(p.y)

# %%
aoi_points

# %%
min(aoi_points["lat"]), max(aoi_points["lat"])

# %%
title = "Areas of interest (Europe)"

fig, ax = plt.subplots(
    ncols=2,
    figsize=(12, 6),
    gridspec_kw={"width_ratios": [2] + [1]},
    constrained_layout=True,
)

# First image
world.plot(ax=ax[0], color="lightgray", edgecolor="black")
for name, p in aoi_points["points"].items():
    ax[0].plot(p.x, p.y, "o", markersize=10, label=name.title())
ax[0].legend()
ax[0].set_xlabel("longitude [deg]")
ax[0].set_ylabel("latitude [deg]")

# Second image
world.plot(ax=ax[1], color="lightgray", edgecolor="black")
for name, p in aoi_points["points"].items():
    ax[1].plot(p.x, p.y, "o", markersize=10, label=name.title())
margin = 5
ax[1].set_ylim(min(aoi_points["lat"]) - margin, max(aoi_points["lat"]) + margin)
ax[1].set_xlim(min(aoi_points["lon"]) - margin, max(aoi_points["lon"]) + margin)
ax[1].set_xlabel("longitude [deg]")
ax[1].set_ylabel("latitude [deg]")

# Add some space as fraction of the entire width.
fig.get_layout_engine().set(wspace=0.1)

_ = fig.suptitle(title, fontweight="bold", y=0.9)

store_figure(title)
