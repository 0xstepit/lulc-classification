"""Configuration structures and loaders."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from lulc.config.dataset import PatchesConfig
from lulc.constants import SCL_BAND_NAME
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
        """Perform validation on the instance values."""
        if len(self.candidates) == 0:
            raise ValueError("There should be at least one area of interest specified")

        for k, v in self.candidates.items():
            if len(v) != 2:
                raise ValueError(
                    f"An area of interested is defined by lat/lon, received {len(v)} values for {k}"
                )

    @classmethod
    def from_dict(cls, aoi_data: dict) -> Self:
        """Create a class instance from the provided dictionary."""
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
        """Perform validation on the instance values."""
        if self.url == "":
            raise ValueError("STAC endpoint URL cannot be empty")

        if self.collection == "":
            raise ValueError("STAC collection cannot be empty")


@dataclass(frozen=True)
class IndicesConfig:
    """Configuration for the indices used in the analysis.

    Attributes
    ----------
    bands_to_channels : map from the band name to the position in
    the scene channel dimension for each indices considered.
    """

    bands_to_channels: dict[str, dict[str, int]]

    def get_channel(self, index: str, band: str) -> int:
        """Return the channel associated with the provided band and index."""
        return self.bands_to_channels[index][band]

    @classmethod
    def from_band_names(
        cls, indices: dict[str, dict[str, str]], msi: MSIConfig
    ) -> Self:
        """Resolve canonical band names into channel positions.

        Parameters
        ----------
        indices : dict[str, dict[str, str]]
            Index name -> {role: canonical band name}. The role is the argument
            the index formula expects, e.g. NDBI consumes `swir` and `nir`, and
            its `swir` role is filled by the `swir1` band.
        msi : MSIConfig
            Supplies the canonical channel order.

        Returns
        -------
        IndicesConfig
            The configuration with every band name resolved to its channel.
        """
        return cls(
            bands_to_channels={
                index: {role: msi.channel_index(band) for role, band in roles.items()}
                for index, roles in indices.items()
            }
        )


@dataclass(frozen=True)
class MSIConfig:
    """Configuration for the MultiSpectral Instrument of the Sentinel-2 mission.

    Attributes
    ----------
    target_resolution : resolution, in metres, every band is resampled to.
    scl_mask_classes : Scene Classification Layer classes to mask out.
    band_order : canonical band names, in the order channels are written.
    bands : native resolution -> {provider asset name: canonical band name}.
    """

    target_resolution: int
    scl_mask_classes: list[int]
    band_order: list[str]
    bands: dict[int, dict[str, str]]

    def __post_init__(self) -> None:
        """Perform validation on the instance values."""
        if len(set(self.band_order)) != len(self.band_order):
            raise ValueError(f"band_order contains duplicates: {self.band_order}")

        if self.target_resolution not in self.bands:
            raise ValueError(
                f"target resolution {self.target_resolution} is not in the analysed bands"
            )

        mapped = {name for assets in self.bands.values() for name in assets.values()}
        declared = set(self.band_order)
        if mapped != declared:
            raise ValueError(
                "band_order and the provider band mapping must describe the same "
                f"bands; only in band_order: {sorted(declared - mapped)}, "
                f"only in bands: {sorted(mapped - declared)}"
            )

        if SCL_BAND_NAME not in declared:
            raise ValueError(f"band_order must contain the {SCL_BAND_NAME!r} band")

    @property
    def num_bands(self) -> int:
        """Total number of Sentinel-2 bands across every resolution."""
        return len(self.band_order)

    @property
    def band_names(self) -> dict[str, str]:
        """Provider asset name to canonical band name, flattened over resolutions."""
        return {
            asset: name
            for assets in self.bands.values()
            for asset, name in assets.items()
        }

    @property
    def scl_band_index(self) -> int:
        """Channel index of the Scene Classification Layer in a downloaded scene."""
        return self.channel_index(SCL_BAND_NAME)

    def channel_index(self, band: str) -> int:
        """Return the channel index of a canonical band name."""
        try:
            return self.band_order.index(band)
        except ValueError:
            raise ValueError(
                f"unknown band {band!r}; configured bands are {self.band_order}"
            ) from None

    def get_bands_list(self) -> list[str]:
        """Return the provider asset names, ordered by the canonical channel order."""
        asset_by_name = {name: asset for asset, name in self.band_names.items()}
        return [asset_by_name[name] for name in self.band_order]

    def reference_asset(self, resolution: int) -> str:
        """Return any provider asset at the given native resolution.

        Used to probe a rasterio profile before a scene is assembled, so which
        asset it is does not matter as long as the resolution matches.
        """
        if resolution not in self.bands:
            raise ValueError(
                f"no bands configured at resolution {resolution}; "
                f"available: {sorted(self.bands)}"
            )
        return next(iter(self.bands[resolution]))


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
        """Perform validation on the instance values."""
        for name, dates in self.seasons.items():
            if len(dates) != 2:
                raise ValueError(f"season {name} should have exactly two dates")


@dataclass(frozen=True)
class WorldCoverConfig:
    """Configuration class for the WorldCover data."""

    url: str
    grid_url: str
    version: str
    year: str
    nodata_value: int
    class_to_color: dict[int, list[int]]
    class_mapping: dict[int, int]
    class_names: dict[int, str]

    def __post_init__(self):
        """Perform validation on the instance values."""
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
    """Project configuration class."""

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

    with Path.open(file_path, "rb") as f:
        _cfg = tomllib.load(f)

        aoi = AoiConfig.from_dict(_cfg.pop("aoi"))
        stac = StacConfig(**_cfg.pop("stac"))

        # Clean way to handle the issue that TOML keys are always string.
        msi_data = _cfg.pop("msi")
        msi_data["bands"] = {int(k): v for k, v in msi_data["bands"].items()}
        msi = MSIConfig(**msi_data)

        indices = IndicesConfig.from_band_names(_cfg.pop("indices")["bands"], msi)

        composites = CompositesConfig(**_cfg.pop("composites"))
        worldcover = WorldCoverConfig(**_cfg.pop("worldcover"))
        patches = PatchesConfig(**_cfg.pop("patches"))

    return Config(aoi, stac, msi, composites, indices, worldcover, patches)
