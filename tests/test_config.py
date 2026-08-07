import dataclasses
from pathlib import Path

import pytest

from src.config.config import (
    AoiConfig,
    CompositesConfig,
    Config,
    IndicesConfig,
    MSIConfig,
    SelectedCandidate,
    StacConfig,
    WorldCoverConfig,
    load_config,
)
from src.config.dataset import PatchesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_TOML = """
[aoi]
max_cloud_coverage    = 10.0
min_scenes            = 30
min_scenes_per_season = 5
single_tile           = true
size                  = 50_000
year                  = 2022

[aoi.selected]
bounding_box = [-6.36, 37.00, -5.79, 37.45]
name = "andalusia"
tile = "MGRS-30STG"

[aoi.candidates]
andalusia  = [-6.0805, 37.2323]
brandeburg = [13.4402, 52.7654]

[stac]
collection = "sentinel-2-l2a"
url        = "https://stac.dataspace.copernicus.eu/v1"

[msi]
scl_mask_classes  = [0, 1, 3, 7, 8, 9, 10, 11]
target_resolution = 10

[msi.bands]
10 = ["B02_10m", "B03_10m", "B04_10m", "B08_10m"]
20 = ["B05_20m", "B06_20m", "B07_20m", "B8A_20m", "B11_20m", "B12_20m", "SCL_20m"]

[composites]
max_scenes_per_season = 10
tiles_size            = 1024
skip_partial_blocks   = true

[composites.seasons]
DJF = ["2021-12-01", "2022-02-28"]
JJA = ["2022-06-01", "2022-08-31"]

[indices.bands_to_channels]
ndvi = { "nir" = 6, "red" = 2 }
ndwi = { "green" = 1, "nir" = 6 }

[worldcover]
grid_url     = "https://example.invalid/grid.geojson"
nodata_value = 0
url          = "https://example.invalid"
version      = "v200"
year         = 2021

[worldcover.class_to_color]
0  = [255, 255, 255]
10 = [0, 100, 0]

[worldcover.class_mapping]
0  = 0
10 = 1

[worldcover.class_names]
0  = "No Data"
10 = "Tree Cover"

[patches]
block_size                = 1024
buffer                    = 1000
max_nan_fraction          = 0.5
normalization_percentiles = [1, 99]
seed                      = 3
seed_candidates           = 200
size                      = 256
stats_retention_fraction  = 0.25

[patches.split]
test  = 0.15
train = 0.7
val   = 0.15
"""


def _write_toml(tmp_path: Path, content: str = VALID_TOML) -> Path:
    file_path = tmp_path / "analysis.toml"
    file_path.write_text(content)
    return file_path


class TestLoadConfig:
    def test_returns_every_section_as_its_own_dataclass(self, tmp_path):
        cfg = load_config(_write_toml(tmp_path))

        assert isinstance(cfg, Config)
        assert isinstance(cfg.aoi, AoiConfig)
        assert isinstance(cfg.stac, StacConfig)
        assert isinstance(cfg.msi, MSIConfig)
        assert isinstance(cfg.composites, CompositesConfig)
        assert isinstance(cfg.indices, IndicesConfig)
        assert isinstance(cfg.worldcover, WorldCoverConfig)
        assert isinstance(cfg.patches, PatchesConfig)

    def test_builds_the_selected_candidate_as_a_dataclass(self, tmp_path):
        # Regression: the section used to be handed over as a raw dict, so
        # cfg.aoi.selected.bounding_box raised AttributeError in the scripts.
        cfg = load_config(_write_toml(tmp_path))

        assert isinstance(cfg.aoi.selected, SelectedCandidate)
        assert cfg.aoi.selected.name == "andalusia"
        assert cfg.aoi.selected.tile == "MGRS-30STG"
        assert cfg.aoi.selected.bounding_box == [-6.36, 37.00, -5.79, 37.45]

    def test_parses_the_section_values(self, tmp_path):
        cfg = load_config(_write_toml(tmp_path))

        assert cfg.aoi.year == 2022
        assert cfg.aoi.size == 50_000
        assert cfg.stac.collection == "sentinel-2-l2a"
        assert cfg.msi.target_resolution == 10
        assert cfg.composites.tiles_size == 1024
        assert cfg.indices.get_channel("ndvi", "nir") == 6
        assert cfg.worldcover.version == "v200"
        assert cfg.patches.split == {"test": 0.15, "train": 0.7, "val": 0.15}

    def test_raises_when_the_file_does_not_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.toml")

    def test_raises_when_a_section_is_missing(self, tmp_path):
        without_stac = VALID_TOML.replace('collection = "sentinel-2-l2a"', "")

        with pytest.raises((KeyError, TypeError)):
            load_config(_write_toml(tmp_path, without_stac))

    def test_the_project_configuration_file_loads(self):
        cfg = load_config(REPO_ROOT / "config" / "analysis.toml")

        assert isinstance(cfg, Config)
        assert isinstance(cfg.aoi.selected, SelectedCandidate)


class TestSelectedCandidate:
    def test_defaults_to_no_bounding_box_and_no_tile(self):
        candidate = SelectedCandidate(name="andalusia")

        assert candidate.bounding_box is None
        assert candidate.tile is None

    def test_is_frozen(self):
        candidate = SelectedCandidate(name="andalusia")

        with pytest.raises(dataclasses.FrozenInstanceError):
            candidate.name = "lombardy"  # type: ignore[misc]


class TestAoiConfig:
    AOI_DATA = {
        "max_cloud_coverage": 10.0,
        "min_scenes": 30,
        "min_scenes_per_season": 5,
        "single_tile": True,
        "candidates": {"andalusia": [-6.0805, 37.2323]},
        "size": 50_000,
        "year": 2022,
        "selected": {"name": "andalusia", "tile": "MGRS-30STG"},
    }

    def test_from_dict_nests_the_selected_candidate(self):
        cfg = AoiConfig.from_dict(dict(self.AOI_DATA))

        assert isinstance(cfg.selected, SelectedCandidate)
        assert cfg.selected.name == "andalusia"
        assert cfg.year == 2022

    def test_uses_defaults_for_size_and_single_tile(self):
        cfg = AoiConfig(
            selected=SelectedCandidate(name="andalusia"),
            year=2022,
            max_cloud_coverage=10.0,
            min_scenes=30,
            min_scenes_per_season=5,
            candidates={"andalusia": [-6.0805, 37.2323]},
        )

        assert cfg.size == 50
        assert cfg.single_tile is True

    def test_raises_when_there_are_no_candidates(self):
        data = dict(self.AOI_DATA, candidates={})

        with pytest.raises(ValueError, match="at least one area of interest"):
            AoiConfig.from_dict(data)

    @pytest.mark.parametrize("coordinates", [[], [1.0], [1.0, 2.0, 3.0]])
    def test_raises_when_a_candidate_is_not_a_lon_lat_pair(self, coordinates):
        data = dict(self.AOI_DATA, candidates={"andalusia": coordinates})

        with pytest.raises(ValueError, match="defined by lat/lon"):
            AoiConfig.from_dict(data)


class TestStacConfig:
    def test_uses_defaults_for_the_retry_policy(self):
        cfg = StacConfig(url="https://example.invalid", collection="sentinel-2-l2a")

        assert cfg.page_size == 20
        assert cfg.timeout == 90.0
        assert cfg.max_retries == 6
        assert cfg.backoff_factor == 1.5
        assert cfg.retry_statuses == [429, 500, 502, 503, 504]

    def test_default_retry_statuses_are_not_shared_between_instances(self):
        first = StacConfig(url="https://example.invalid", collection="a")
        second = StacConfig(url="https://example.invalid", collection="b")

        first.retry_statuses.append(418)

        assert second.retry_statuses == [429, 500, 502, 503, 504]

    def test_raises_when_the_url_is_empty(self):
        with pytest.raises(ValueError, match="URL cannot be empty"):
            StacConfig(url="", collection="sentinel-2-l2a")

    def test_raises_when_the_collection_is_empty(self):
        with pytest.raises(ValueError, match="collection cannot be empty"):
            StacConfig(url="https://example.invalid", collection="")

    def test_is_frozen(self):
        cfg = StacConfig(url="https://example.invalid", collection="sentinel-2-l2a")

        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.timeout = 1.0  # type: ignore[misc]


class TestIndicesConfig:
    BANDS_TO_CHANNELS = {"ndvi": {"nir": 6, "red": 2}, "ndwi": {"green": 1, "nir": 6}}

    def test_get_channel_returns_the_mapped_position(self):
        cfg = IndicesConfig(bands_to_channels=dict(self.BANDS_TO_CHANNELS))

        assert cfg.get_channel("ndvi", "nir") == 6
        assert cfg.get_channel("ndvi", "red") == 2
        assert cfg.get_channel("ndwi", "green") == 1

    def test_raises_for_an_unknown_index(self):
        cfg = IndicesConfig(bands_to_channels=dict(self.BANDS_TO_CHANNELS))

        with pytest.raises(KeyError):
            cfg.get_channel("ndbi", "swir")

    def test_raises_for_a_band_missing_from_the_index(self):
        cfg = IndicesConfig(bands_to_channels=dict(self.BANDS_TO_CHANNELS))

        with pytest.raises(KeyError):
            cfg.get_channel("ndvi", "swir")


class TestMSIConfig:
    BANDS = {
        "10": ["B02_10m", "B03_10m", "B04_10m", "B08_10m"],
        "20": [
            "B05_20m",
            "B06_20m",
            "B07_20m",
            "B8A_20m",
            "B11_20m",
            "B12_20m",
            "SCL_20m",
        ],
    }

    def _config(self, **overrides) -> MSIConfig:
        kwargs = {
            "target_resolution": 10,
            "scl_mask_classes": [0, 1, 3, 7, 8, 9, 10, 11],
            "bands": {key: list(value) for key, value in self.BANDS.items()},
        }
        kwargs.update(overrides)
        return MSIConfig(**kwargs)

    def test_converts_the_toml_string_resolutions_to_int(self):
        cfg = self._config()

        assert set(cfg.bands) == {10, 20}

    def test_counts_every_band_across_the_resolutions(self):
        cfg = self._config()

        assert cfg.num_bands == 11

    def test_raises_when_the_target_resolution_has_no_bands(self):
        with pytest.raises(ValueError, match="target resolution 60"):
            self._config(target_resolution=60)

    def test_get_bands_list_is_flat_and_sorted(self):
        cfg = self._config()

        assert cfg.get_bands_list() == [
            "B02_10m",
            "B03_10m",
            "B04_10m",
            "B05_20m",
            "B06_20m",
            "B07_20m",
            "B08_10m",
            "B11_20m",
            "B12_20m",
            "B8A_20m",
            "SCL_20m",
        ]

    def test_scl_index_points_at_the_scl_band(self):
        # The index is hardcoded, so it must stay consistent with the band order
        # the download script writes.
        cfg = self._config()

        assert cfg.get_bands_list()[cfg.get_scl_band_index()] == "SCL_20m"


class TestCompositesConfig:
    def test_keeps_the_season_date_ranges(self):
        cfg = CompositesConfig(
            max_scenes_per_season=10,
            seasons={"DJF": ["2021-12-01", "2022-02-28"]},
            tiles_size=1024,
            skip_partial_blocks=True,
        )

        assert cfg.seasons["DJF"] == ["2021-12-01", "2022-02-28"]

    @pytest.mark.parametrize(
        "dates", [[], ["2022-06-01"], ["2022-06-01", "2022-07-01", "2022-08-31"]]
    )
    def test_raises_when_a_season_is_not_a_date_pair(self, dates):
        with pytest.raises(ValueError, match="season JJA"):
            CompositesConfig(
                max_scenes_per_season=10,
                seasons={"JJA": dates},
                tiles_size=1024,
                skip_partial_blocks=True,
            )

    def test_is_frozen(self):
        cfg = CompositesConfig(
            max_scenes_per_season=10,
            seasons={},
            tiles_size=1024,
            skip_partial_blocks=True,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.tiles_size = 512  # type: ignore[misc]


class TestWorldCoverConfig:
    def _config(self, **overrides) -> WorldCoverConfig:
        kwargs = {
            "url": "https://example.invalid",
            "grid_url": "https://example.invalid/grid.geojson",
            "version": "v200",
            "year": 2021,
            "nodata_value": 0,
            "class_to_color": {"0": [255, 255, 255], "10": [0, 100, 0]},
            "class_mapping": {"0": 0, "10": 1},
            "class_names": {"0": "No Data", "10": "Tree Cover"},
        }
        kwargs.update(overrides)
        return WorldCoverConfig(**kwargs)

    def test_converts_the_toml_string_keys_to_int(self):
        cfg = self._config()

        assert cfg.class_names == {0: "No Data", 10: "Tree Cover"}
        assert cfg.class_mapping == {0: 0, 10: 1}
        assert cfg.class_to_color == {0: [255, 255, 255], 10: [0, 100, 0]}

    def test_leaves_already_integer_keys_untouched(self):
        cfg = self._config(class_names={0: "No Data"}, class_mapping={0: 0})

        assert cfg.class_names == {0: "No Data"}
        assert cfg.class_mapping == {0: 0}

    def test_is_frozen(self):
        cfg = self._config()

        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.version = "v100"  # type: ignore[misc]
