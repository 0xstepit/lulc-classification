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
# # WorldCover remapping
#
# In this notebook we explore how to use a lookup table to convert the non-consecutive classes in our WorldCover raster into consecutive ones.

# %%
# Boilerplate code to automatically reload imports and to have
# access to the project sourcecode.
# %load_ext autoreload
# %autoreload 2

import os
import sys

sys.path.append(os.path.abspath(".."))

# %%
import numpy as np
import rasterio

from lulc.io import WORLDCOVER_LABELS

# %%
with rasterio.open(WORLDCOVER_LABELS) as labels_src:
    labels = labels_src.read()

# %%
labels

# %%
nodata_value = 0
present_labels = [int(v) for v in np.unique(labels) if int(v) != nodata_value]
present_labels

# %% [markdown]
# Create the remapping from these non consecutive values to a consecutive sequence:

# %%
mapping = {label: value for value, label in enumerate(present_labels)}
mapping

# %% [markdown]
# We can now create a lookup table:

# %%
lut = np.full(256, 255, dtype=np.uint8)
for value, label in mapping.items():
    lut[value] = label
lut

# %% [markdown]
# Now we can quickly convert the entire label raster from the original labels to our more classic sequential ones:

# %%
lut[labels]
