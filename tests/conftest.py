"""Shared test fixtures.

Payload fixtures return a freshly built literal on every request, so each test
owns its nested dictionaries outright.
"""

from collections.abc import Callable

import pytest

from lulc.config.config import MSIConfig


@pytest.fixture
def aoi_data() -> dict:
    """Payload accepted by `AoiConfig.from_dict`. Mutate it freely."""
    return {
        "max_cloud_coverage": 10.0,
        "min_scenes": 30,
        "min_scenes_per_season": 5,
        "single_tile": True,
        "candidates": {"andalusia": [-6.0805, 37.2323]},
        "size": 50_000,
        "year": 2022,
        "selected": {"name": "andalusia", "tile": "MGRS-30STG"},
    }


@pytest.fixture
def bands_to_channels() -> dict[str, dict[str, int]]:
    """Index-to-channel mapping accepted by `IndicesConfig`."""
    return {
        "ndvi": {"nir": 6, "red": 2},
        "ndwi": {"green": 1, "nir": 6},
    }


@pytest.fixture
def msi_kwargs() -> dict:
    """Base `MSIConfig` payload."""
    return {
        "target_resolution": 10,
        "scl_mask_classes": [0, 1, 3, 7, 8, 9, 10, 11],
        "band_order": [
            "blue",
            "green",
            "red",
            "red_edge1",
            "red_edge2",
            "red_edge3",
            "nir",
            "narrow_nir",
            "swir1",
            "swir2",
            "scl",
        ],
        "bands": {
            10: {
                "B02_10m": "blue",
                "B03_10m": "green",
                "B04_10m": "red",
                "B08_10m": "nir",
            },
            20: {
                "B05_20m": "red_edge1",
                "B06_20m": "red_edge2",
                "B07_20m": "red_edge3",
                "B8A_20m": "narrow_nir",
                "B11_20m": "swir1",
                "B12_20m": "swir2",
                "SCL_20m": "scl",
            },
        },
    }


# Pytest factory fixture pattern.
@pytest.fixture
def make_msi(msi_kwargs: dict) -> Callable[..., MSIConfig]:
    """Build an `MSIConfig`, replacing individual fields by keyword.

    Request `msi_kwargs` alongside this fixture to read the baseline values; both
    resolve to the same object within a single test.
    """

    def _make(**overrides) -> MSIConfig:
        return MSIConfig(**(msi_kwargs | overrides))

    return _make
