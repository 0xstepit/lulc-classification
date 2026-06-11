import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.geometry import BoundingBox
from src.io import CONFIG_DIR, SENTINEL_CONFIG


@dataclass
class AoIConfig:
    name: str
    year: int
    max_cloud_coverage: float
    min_scenes: int
    min_scenes_per_season: int
    candidates: dict[str, list[float]]
    bounding_box: BoundingBox | None = None
    tile: str | None = None
    size: float = 50

    def __post_init__(self) -> None:
        if len(self.candidates) == 0:
            raise ValueError("There should be at least one area of interest specified")

        for k, v in self.candidates.items():
            if len(v) != 2:
                raise ValueError(
                    f"An area of interested is defined by lat/lon, received {len(v)} values for {k}"
                )


@dataclass(frozen=True)
class StacConfig:
    """
    Configuration class for the STAC client.
    """

    url: str
    collection: str

    def __post_init__(self) -> None:
        if self.url == "":
            raise ValueError("STAC endpoint URL cannot be empty")

        if self.collection == "":
            raise ValueError("STAC collection cannot be empty")


@dataclass
class MSIConfig:
    """
    Configuraion class for the MultiSpectral images of the Sentinel2 mission.
    """

    bands: dict
    target_resolution: int
    scl_mask_classes: list[int]

    def __post_init__(self):
        self.num_bands = len([band for res in self.bands for band in self.bands[res]])
        self.bands = {int(k): v for k, v in self.bands.items()}

    def get_bands(self):
        return sorted([band for res in self.bands for band in self.bands[res]])

    def get_scl_band_index(self):
        """Returns the SCL Sentinel2 band position in the scenes. This value is hardcoded
        because it assumes that each scene has bands ordered like in the configuration
        file.

        Returns
        -------
        int
            The SCL map index in a scene.

        """
        return 10


@dataclass(frozen=True)
class Composites:
    """Seasonal composite raster configuration.

    Attributes
    ----------
        max_scenes_per_season: Max number of scenes to use per season.
        seasons: Start and date of each season.
        statistic: Statistic to use within each seasons.
    """

    max_scenes_per_season: int
    seasons: dict[str, list[str]]
    statistic: str

    def __post_init__(self):
        for name, dates in self.seasons.items():
            if len(dates) != 2:
                raise ValueError(f"season {name} should have exactly two dates")


@dataclass(frozen=True)
class Sentinel2Config:
    aoi: AoIConfig
    stac: StacConfig
    msi: MSIConfig
    composites: Composites


def load_sentinel2_config(path: Path = CONFIG_DIR) -> Sentinel2Config:
    file_path = CONFIG_DIR / SENTINEL_CONFIG

    if not file_path.exists():
        raise FileNotFoundError(
            f"File named {SENTINEL_CONFIG} does not exists in {path}"
        )

    with open(file_path, "rb") as f:
        _cfg = tomllib.load(f)

        aoi = AoIConfig(**_cfg.pop("aoi"))
        stac = StacConfig(**_cfg.pop("stac"))
        msi = MSIConfig(**_cfg.pop("msi"))
        composites = Composites(**_cfg.pop("composites"))

    return Sentinel2Config(aoi, stac, msi, composites)
