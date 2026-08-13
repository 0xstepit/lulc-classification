# Defines the correct order of the season IDs.
from lulc.config import MSIConfig

SEASONS_ORDER = ["DJF", "MAM", "JJA", "SON"]

# Season acronym to associated month numbers.
SEASON_MONTHS = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
}

INDEX_NAMES = ["NDVI", "NDBI", "NDWI"]


def seasonal_band_names(cfg: MSIConfig) -> list[str]:
    # Well, yes the SCL band is hardcoded..
    ordered_bands = [
        cfg.band_names[band]
        for band in cfg.get_bands_list()
        if "scl" not in band.lower()
    ]
    return ordered_bands + INDEX_NAMES


def composite_band_names(cfg: MSIConfig, num_seasons: int) -> list[str]:
    return seasonal_band_names(cfg) * num_seasons
