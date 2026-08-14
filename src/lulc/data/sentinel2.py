"""Functions and classes to work with the Sentinel-2 mission data."""

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pystac
import rasterio
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.profiles import Profile
from rasterio.windows import Window
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from urllib3.util.retry import Retry

from lulc.config import Config
from lulc.constants import SEASON_MONTHS
from lulc.domain import BoundingBox, SceneCounts
from lulc.provenance import stamp

logger = logging.getLogger(__name__)

# Militaty Grid Reference System prefix used in the Setninel-2 data.
MGRS_PREFIX = "MGRS-"


class StacClient:
    """A thin STAC client wrapper to simplify data access and management."""

    def __init__(self, cfg: Config) -> None:
        # By configuring the Retry on the STAC client we can re-execute
        # the single HTTP request that failed resuming the pagination.
        retry = Retry(
            total=cfg.stac.max_retries,
            backoff_factor=cfg.stac.backoff_factor,
            status_forcelist=cfg.stac.retry_statuses,
            allowed_methods=frozenset(
                {"GET", "POST"}
            ),  # required because STAC uses POST and urllib3 does not retry them by default
            backoff_jitter=1.0,
            respect_retry_after_header=True,
        )
        stac_io = StacApiIO(timeout=cfg.stac.timeout, max_retries=retry)
        self._client = Client.open(cfg.stac.url, stac_io=stac_io)
        # TODO: separate the config into smaller chunks and inject only what is relevant
        # for the client.
        self._cfg = cfg

    def search_items(
        self,
        bbox: BoundingBox,
        datetime: str | None = None,
        grid_code: str | None = None,
        single_tile: bool = False,
    ) -> list[pystac.Item]:
        """Search STAC items filtered by the inputs.

        Parameters
        ----------
        bbox : BoundingBox
            Bounding box that must be contained in the scene.
        datetime : str | None
            Date or range of dates used to filter the scenes. If not provided, it falls back
            to the configuration year for the AOI.
        grid_code : str | None
            Grid code of the tile associated with the scene containin the bounding box.
        single_tile: bool
            If True, only the scenes associated with the most frequent tile are returned.

        Returns
        -------
        list[pystac.Item]
            A list of STAC Items.
        """
        query: dict[str, Any] = {
            "eo:cloud_cover": {"lt": self._cfg.aoi.max_cloud_coverage}
        }
        if grid_code is not None:
            query["grid:code"] = {"eq": grid_code}

        if datetime is None:
            year = self._cfg.aoi.year
            datetime = f"{year}-01-01/{year}-12-31"

        search = self._client.search(
            collections=[self._cfg.stac.collection],
            bbox=list(bbox),
            datetime=datetime,
            query=query,
            limit=self._cfg.stac.page_size,
        )

        items = list(search.items())

        if single_tile and grid_code is None:
            items = filter_most_frequent_tile(items)

        return items


@dataclass
class ResamplingStrategy:
    """Datawrapper to control rasterio resampling strategy.

    Attributes
    ----------
    factor : float
        The resampling factor is > 1 for upsampling and < 1 for downsampling.
    method : Resampling
        The resampling method. It should be bilinear for continuous variables and nearest
        neighboor for categorical ones.
    """

    factor: float = 1
    method: Resampling = Resampling.nearest

    def get_factor(self) -> float:
        """Return the strategy resampling factor."""
        return self.factor

    def get_method(self) -> Resampling:
        """Return the strategy sampling method."""
        return self.method


def count_scenes_by_seasons(items: list[pystac.Item]) -> SceneCounts:
    """Count the number of scene per each season.

    Parameters
    ----------
    items : list[pystac.Item]
        A list of STAC items.

    Returns
    -------
    SceneCounts
        The information abound the counting for each provided scene.
    """
    scene_counts = SceneCounts()

    for item in items:
        if item.datetime is None:
            continue
        month = item.datetime.month
        for season, months in SEASON_MONTHS.items():
            if month in months:
                scene_counts.increment_counter(season)

    return scene_counts


def _log_retry(retry_state) -> None:
    logger.warning(
        f"retrying get_scene for {retry_state.args[0]} "
        f"(attempt {retry_state.attempt_number}): {retry_state.outcome.exception()}"
    )


@retry(
    retry=retry_if_exception_type((RasterioIOError, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
)
def get_data_profile(item_path: str | Path) -> Profile:
    """Return the rasterio Profile for the raster at the provided path.

    Parameters
    ----------
    item_path : str | Path
        Path referencing the STAC Item location locally or on the web.

    Returns
    -------
    Profile
        The Rasterio profile of the item.
    """
    with rasterio.open(item_path) as src:
        return src.profile


@retry(
    retry=retry_if_exception_type((RasterioIOError, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
)
def get_scene(
    item_path: str,
    window: Window,
    resampling_strategy: ResamplingStrategy,
) -> np.ndarray:
    """Get the data associated with a scene with support for windowing and resampling."""
    resampling_factor = resampling_strategy.get_factor()
    resampling_method = resampling_strategy.get_method()

    with rasterio.open(item_path) as src:
        return src.read(
            1,
            window=window,
            out_shape=(
                int(window.height * resampling_factor),
                int(window.width * resampling_factor),
            ),
            resampling=resampling_method,
        )


def create_all_bands_scene(
    out_file: Path,
    profile: Profile,
    window: Window,
    target_res: float,
    assets: dict,
    bands: list[str],
    cfg: Config,
    channels: list[str],
    **extra: str,
) -> None:
    """Create a raster from the provided STAC assets and store it on disk.

    Bands whose resolution differs from the target one are resampled.

    Parameters
    ----------
    out_file : Path
        Filepath where the raster is stored.
    profile : Profile
        Rasterio profile to use for the raster.
    window : Window
        Rasterio window to use for the area of interest.
    target_res : float
        Final resolution for the raster.
    assets : dict
        STAC assets to download.
    bands : list[str]
        Provider asset names to include, in write order.
    cfg : Config
        Project configuration, used to build the provenance tags.
    channels : list[str]
        Canonical channel names, aligned with `bands`. Written as the band
        descriptions and recorded in the provenance tags.
    **extra : str
        Additional provenance tags, e.g. the originating STAC item id.
    """
    # Open a file in write mode with rasterio and the provided profile.
    logger.info(f"opening file {out_file.name}")
    with rasterio.open(out_file, "w", **profile) as dst:
        for idx, band in enumerate(bands):
            resolution = get_resolution_from_band_name(band)

            # Resampling strategy is bilinear for continuous variables and nearest
            # for categorical, like the SCL mask.
            resampling_strategy = ResamplingStrategy(
                resolution / target_res,
                Resampling.bilinear if "SCL" not in band else Resampling.nearest,
            )

            data = get_scene(assets[band].href, window, resampling_strategy)

            validate_scene_band(profile, data)

            # Notice that bands are stored starting from 1 in GDAL.
            dst.write(data, idx + 1)

        # Stamped once, after every band is written, rather than inside the loop.
        stamp(dst, cfg, channels, **extra)


def validate_scene_band(profile: Profile, data: np.ndarray) -> None:
    """Validate a downloaded scene against the desired composition profile.

    Parameters
    ----------
    profile: rasterio.profiles.Profile
        The profile used in the aggregated raster.
    data: numpy.ndarray
        The data associated with the raster.

    Raises
    ------
    ValueError
        Returns an error if the dimensions of the data don't match the profile ones.
    """
    if data.ndim != 2:
        raise ValueError(
            f"expected 2-dimensional data, obtained a {data.ndim}-dimensional one"
        )

    h, w = data.shape

    if w != profile["width"]:
        raise ValueError(f"wrong width: expected {profile['width']}, received {w}")

    if h != profile["height"]:
        raise ValueError(f"wrong height: expected {profile['height']}, received {h}")


def get_resolution_from_band_name(band: str) -> float:
    """Return the resolution from the band name.

    This is a shortcut instead of looking at the transform matric of the associated data because
    we know we are using Sentinel2 dataset. This way, we don't have to read the dataset twice
    to know the resolution and perform pre-processing.
    """
    res_with_unit = band.split("_")[1]
    res = res_with_unit.replace("m", "")

    try:
        return float(res)
    except ValueError as e:
        raise ValueError(
            f"{band} does not contain resolution info in the form of <BAND_NAME>_<RES>m"
        ) from e


def get_tile_id(item: pystac.Item) -> str | None:
    """Get the identifier of the tile associated with the provided STAC item.

    The function is valid only for Sentinel-2 data. The function assumes that the
    ID is contained in the STAC properties at the key `grid:code`.

    Parameters
    ----------
    item : pystac.Item
        The items of which we want to know the tile ID.

    Returns
    -------
    str
        The tile ID without the MGRS prefix.
    """
    value = item.properties.get("grid:code")
    if isinstance(value, str) and value:
        return value.removeprefix(MGRS_PREFIX).strip().upper()

    # # An example of Sentinel2 tile name: S2A_MSIL2A_20220903T110631_N0510_R137_T29SQA_20240729T160319
    # return item.id.split("_")[5]
    return None


def filter_most_frequent_tile(items: list[pystac.Item]) -> list[pystac.Item]:
    """Return the Items of the most frequent tile."""
    tile_counts = Counter(get_tile_id(item) for item in items)
    selected_tile = tile_counts.most_common(1)[0][0]
    return [item for item in items if get_tile_id(item) == selected_tile]


def evaluate_candidate_validity(cfg: Config, scene_counts: SceneCounts) -> bool:
    """Check if an AOI is a valid area based on the provided result.

    The validity is based on the number of scenes available.

    Parameters
    ----------
    cfg : Sentinel2Config
        Sentinel2 analysis configuration.
    candidate_analysis : CandidateResult
        Result of the candidate analysis used to evaluate its validity.

    Returns
    -------
    bool
        True if the AOI is a valid candidate, False otherwise.
    """
    # Evaluate minimum overall scenes.
    if scene_counts.total < cfg.aoi.min_scenes:
        return False
    # Evaluate minimum scenes per season.
    for _, count in scene_counts.by_season.items():
        if count < cfg.aoi.min_scenes_per_season:
            return False

    return True


# TODO: here we should use the value in the Item asset.
def rescale_reflectances(raster: np.ndarray):
    """Rescale reflectances values to original range."""
    rescaled = raster.astype(np.float32)
    rescaled = np.where(rescaled == 0, np.nan, rescaled)
    # Clip negative reflectances.
    return np.maximum(rescaled * 0.0001 - 0.1, 0.0)
