import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pystac
import rasterio
from pystac_client import Client
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

from src.config import BoundingBox, Sentinel2Config
from src.constants import SEASON_MONTHS

logger = logging.getLogger("sentinel2")


class SentinelClient:
    """The sentinel clinet for the CDSE database."""

    def __init__(self, cfg: Sentinel2Config) -> None:
        self._client = Client.open(cfg.stac.url)
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
        return self.factor

    def get_method(self) -> Resampling:
        return self.method


@dataclass
class SceneCounts:
    """Contains information associated with the number of scenes found for each season.

    Attributes
    ----------
    total : int
        The overall number of scenes.
    by_season : dict[str, int]
        The number of available scenes per season.
    """

    total: int = 0
    by_season: dict = field(
        default_factory=lambda: {"DJF": 0, "MAM": 0, "JJA": 0, "SON": 0}
    )

    def increment_counter(self, season: str):
        self.by_season[season] += 1
        self.total += 1


def count_scenes_by_seasons(items: list[pystac.Item]) -> SceneCounts:
    """Counts the number of scene per each season.

    Parameters
    ----------
    items : list[pystac.Item]
        A list of STAC items.

    Returns
    -------
    SceneCounts
        The information abound the counting.
    """
    # It seems that mathced does not work here ???
    # total = search.matched()
    scene_counts = SceneCounts()

    for item in items:
        if item.datetime is None:
            continue
        month = item.datetime.month
        for season, months in SEASON_MONTHS.items():
            # TODO: should not happen, but log not existing month.
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
def get_data_profile(item_path: str) -> Profile:
    """Returns the rasterio Profile for the raster at the provided path.

    Parameters
    ----------
    item_path : str
        Path referencing the STAC Item.

    Returns
    -------
    Profile
        The Rasterio profile of the item.
    """
    with rasterio.open(item_path) as src:
        profile = src.profile

    return profile


@retry(
    retry=retry_if_exception_type((RasterioIOError, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
)
def get_scene(
    item_path: str,
    window: Window,
    resampling_strategy: ResamplingStrategy = ResamplingStrategy(),
) -> np.ndarray:
    """Get the data associated with a scene with support for windowing and resampling."""
    resampling_factor = resampling_strategy.get_factor()
    resampling_method = resampling_strategy.get_method()

    with rasterio.open(item_path) as src:
        scaled_window = Window(
            window.col_off / resampling_factor,  # type: ignore[call-arg]
            window.row_off / resampling_factor,
            window.width / resampling_factor,
            window.height / resampling_factor,
        )

        # An output shape is always created and used, so it should be ok to always set it here,
        # even with the original size.
        # https://github.com/rasterio/rasterio/blob/4e5bce88ea3c84b41a394244fe1cad6a5b8eb854/rasterio/_io.pyx#L544-L547
        data = src.read(
            1,
            window=scaled_window,
            out_shape=(int(window.height), int(window.width)),
            resampling=resampling_method,
        )
    return data


def download_all_bands_scene(
    out_file: Path,
    profile: Profile,
    window: Window,
    target_res: float,
    assets: dict,
    bands: list[str],
) -> None:
    with rasterio.open(out_file, "w", **profile) as dst:
        for idx, band in enumerate(bands):
            logger.info(f"starting download for band {band} and index {idx}")

            # Resolution information
            resolution = get_resolution_from_band_name(band)

            resampling_strategy = ResamplingStrategy(
                resolution / target_res,
                Resampling.bilinear if "SCL" not in band else Resampling.nearest,
            )

            data = get_scene(assets[band].href, window, resampling_strategy)

            validate_scene_band(profile, data)

            # Notice that bands are stored starting from 1 in GDAL.
            dst.write(data, idx + 1)


def validate_scene_band(profile: Profile, data: np.ndarray) -> None:
    """
    Validate a downloaded scene against the desired composition profile.

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
    """Returns the resolution from the band name. This is a shortcut instead of looking at the
    transform matric of the associated data because we know we are using Sentinel2 dataset.
    This way, we don't have to read the dataset twice to know the resolution and perform pre-processing.
    """
    res_with_unit = band.split("_")[1]
    res = res_with_unit.replace("m", "")

    try:
        return float(res)
    except ValueError:
        raise ValueError(
            f"{band} does not contain resolution info in the form of <BAND_NAME>_<RES>m"
        )


def get_tile_id(item: pystac.Item) -> str:
    """Get the identifier of the tile associated with the provided STAC item. The function
    is valid only for Sentinel2 data containedj in the CDSE database.

    Parameters
    ----------
    item : pystac.Item
        The items of which we want to know the tile ID.

    Returns
    -------
    str
        The tile ID.
    """
    # An example of Sentinel2 tile name: S2A_MSIL2A_20220903T110631_N0510_R137_T29SQA_20240729T160319
    return item.id.split("_")[5]


def filter_most_frequent_tile(items: list[pystac.Item]) -> list[pystac.Item]:
    tile_counts = Counter(get_tile_id(item) for item in items)
    selected_tile = tile_counts.most_common(1)[0][0]
    return [item for item in items if get_tile_id(item) == selected_tile]
