"""Project-wide constants and the band-naming helpers derived from them.

This module is a leaf: it holds no runtime dependency on any other `lulc`
subpackage. `MSIConfig` is imported for typing only, so that `lulc.io` — which
imports this module for `SEASONS_ORDER` — does not end up sitting above the
configuration package in the dependency graph.
"""

from typing import TYPE_CHECKING

# We need it to reference to MSIConfig in the annotation without actually
# importing it.
if TYPE_CHECKING:
    from lulc.config import MSIConfig

# Defines the correct order of the season IDs.
SEASONS_ORDER = ["DJF", "MAM", "JJA", "SON"]

# Season acronym to associated month numbers.
SEASON_MONTHS = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
}

INDEX_NAMES = ["NDVI", "NDBI", "NDWI"]

# Canonical name of the Scene Classification Layer. It is the one band that is
# consumed during masking rather than carried into the composites.
SCL_BAND_NAME = "scl"


def seasonal_band_names(cfg: MSIConfig) -> list[str]:
    """Return the channel names of a seasonal composite, in write order.

    The spectral bands come first, in the canonical order declared by
    `[msi] band_order`, minus the SCL layer which masking consumes. The spectral
    indices are appended afterwards.
    """
    spectral = [name for name in cfg.band_order if name != SCL_BAND_NAME]
    return spectral + INDEX_NAMES


def composite_band_names(cfg: MSIConfig, num_seasons: int) -> list[str]:
    """Return the position of the season bands in the all season raster."""
    return seasonal_band_names(cfg) * num_seasons
