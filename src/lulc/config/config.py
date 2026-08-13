import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from lulc.config.dataset import PatchesConfig
from lulc.domain import BoundingBox


@dataclass(frozen=True)
class SelectedCandidate:
    """The selected candidate for the analysis.

    Attributes
    ----------
    name : name of the AOI.
    bounding_box : bounding box of the AOI.
    tile : Sentinel-2 CDSE tile containing the AOI.
    """

    name: str
    bounding_box: BoundingBox | None = None
    tile: str | None = None


@dataclass
class AoiConfig:
    """Area Of Interest configuration class.

    Attributes
    ----------
    selected: the configuration of the selected candidate.
    year : year of evaluation of the AOI in the preliminary analysis.
    max_cloud_coverage : cloud coverage percentage after which scenes are filtered out.
    min_scenes : minimum of scenes to consider an AOI valid.
    min_scenes_per_season : minimum of scenes for each season to consider an AOI valid.
    candidates : center point in (lon, lat) for candidates AOI.
    size : lenght of each side of the AOI.
    single_tile : specifies if the analysis should only consider the tile with the highest
                  number of scenes.
    """

    selected: SelectedCandidate
    year: int
    max_cloud_coverage: float
    min_scenes: int
    min_scenes_per_season: int
    candidates: dict[str, list[float]]
    size: float = 50
    single_tile: bool = True

    def __post_init__(self) -> None:
        if len(self.candidates) == 0:
            raise ValueError("There should be at least one area of interest specified")

        for k, v in self.candidates.items():
            if len(v) != 2:
                raise ValueError(
                    f"An area of interested is defined by lat/lon, received {len(v)} values for {k}"
                )

    @classmethod
    def from_dict(cls, aoi_data: dict) -> Self:

        return cls(
            max_cloud_coverage=aoi_data["max_cloud_coverage"],
            min_scenes=aoi_data["min_scenes"],
            min_scenes_per_season=aoi_data["min_scenes_per_season"],
            single_tile=aoi_data["single_tile"],
            candidates=aoi_data["candidates"],
            size=aoi_data["size"],
            year=aoi_data["year"],
            selected=SelectedCandidate(**aoi_data["selected"]),
        )


@dataclass(frozen=True)
class StacConfig:
    """Configuration class for the STAC client.

    Attributes
    ----------
    url : url of the server.
    collection : STAC collection to use.
    """

    url: str
    collection: str
    page_size: int = 20
    timeout: float = 90.0
    max_retries: int = 6
    backoff_factor: float = 1.5
    retry_statuses: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )  # needed because frozen class

    def __post_init__(self) -> None:
        if self.url == "":
            raise ValueError("STAC endpoint URL cannot be empty")

        if self.collection == "":
            raise ValueError("STAC collection cannot be empty")


@dataclass(frozen=True)
class IndicesConfig:
    """Configuration for the indices used in the analysis.

    Attributes
    ----------
    bands_to_channels : dictionary mapping the band name to the position in the scene channel
                        dimension for each indices considered.
    """

    bands_to_channels: dict[str, dict[str, int]]

    def get_channel(self, index: str, band: str) -> int:
        return self.bands_to_channels[index][band]


@dataclass
class MSIConfig:
    """Configuraion class for the MultiSpectral images of the Sentinel2 mission.

    Attributes
    ----------
    bands :
    target_resolution :
    scl_mask_classes :
    num_bands :
    bands :
    band_names:
    """

    target_resolution: int
    scl_mask_classes: list[int]
    bands: dict
    band_names: dict

    def __post_init__(self):
        self.num_bands = len([band for res in self.bands for band in self.bands[res]])
        self.bands = {int(k): v for k, v in self.bands.items()}

        if self.target_resolution not in list(self.bands.keys()):
            raise ValueError(
                f"target resolution {self.target_resolution} is not in the analysed bands"
            )

        if len(self.band_names.keys()) != self.num_bands:
            raise ValueError(
                "band names is different than number of bands, "
                f"expected {self.num_bands} but got {len(self.band_names.keys())}"
            )

    def get_bands_list(self) -> list[str]:
        """Returns the sorted list of Sentinel-2 bands used in the analysis.

        Returns
        -------
        list[str]
            The sorted list of band names.
        """
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
class CompositesConfig:
    """Seasonal composite raster configuration.

    Attributes
    ----------
    max_scenes_per_season: Max number of scenes to use per season.
    seasons: Start and date of each season.
    statistic: Statistic to use within each seasons.
    tiles_size: dimension for [H, W] of the window to tile each scene.
    skip_partial_scenes: skip or not partial blocks in the final composite. We can
        have partial blocks if the block size used in the image stored on disk is not
        a factor of raster width and height.
    """

    max_scenes_per_season: int
    seasons: dict[str, list[str]]
    tiles_size: int
    skip_partial_blocks: bool

    def __post_init__(self):
        for name, dates in self.seasons.items():
            if len(dates) != 2:
                raise ValueError(f"season {name} should have exactly two dates")


@dataclass(frozen=True)
class WorldCoverConfig:
    url: str
    grid_url: str
    version: str
    year: str
    nodata_value: int
    class_to_color: dict[int, list[int]]
    class_mapping: dict[int, int]
    class_names: dict[int, str]

    def __post_init__(self):
        # We need to bypass the frozen class because TOML keys are always string:
        object.__setattr__(
            self,
            "class_names",
            {int(key): value for key, value in self.class_names.items()},
        )
        object.__setattr__(
            self,
            "class_mapping",
            {int(key): value for key, value in self.class_mapping.items()},
        )
        object.__setattr__(
            self,
            "class_to_color",
            {int(key): value for key, value in self.class_to_color.items()},
        )


@dataclass(frozen=True)
class Config:
    aoi: AoiConfig
    stac: StacConfig
    msi: MSIConfig
    composites: CompositesConfig
    indices: IndicesConfig
    worldcover: WorldCoverConfig
    patches: PatchesConfig


def load_config(file_path: Path) -> Config:
    """Load project configuration from disk.

    Parameters
    ----------
    file_path : Path
        The path pointing to the configuration file in `.toml` format.

    Returns
    -------
    Config
        The project configuration file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"file {file_path} does not exists")

    with open(file_path, "rb") as f:
        _cfg = tomllib.load(f)

        aoi = AoiConfig.from_dict(_cfg.pop("aoi"))
        stac = StacConfig(**_cfg.pop("stac"))
        msi = MSIConfig(**_cfg.pop("msi"))
        composites = CompositesConfig(**_cfg.pop("composites"))
        indices = IndicesConfig(**_cfg.pop("indices"))
        worldcover = WorldCoverConfig(**_cfg.pop("worldcover"))
        patches = PatchesConfig(**_cfg.pop("patches"))

    return Config(aoi, stac, msi, composites, indices, worldcover, patches)
