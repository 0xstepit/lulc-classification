import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pystac
import rasterio
from pystac_client import Client
from rasterio import warp
from rasterio.enums import Resampling
from rasterio.profiles import Profile
from rasterio.windows import from_bounds

from src.config import BBox, Sentinel2Config
from src.logger import setup_logging

setup_logging()
logger = logging.getLogger("sentinel2")


@dataclass
class ResamplingStrategy:
    """Datawrapper to control rasterio resampling strategy"""

    # > 1 for upsampling, < 1 for downsampling.
    factor: float = 1
    method: Resampling = Resampling.nearest

    def get_factor(self) -> float:
        return self.factor

    def get_method(self) -> Resampling:
        return self.method


class SentinelClient:
    def __init__(self, cfg: Sentinel2Config) -> None:
        self._client = Client.open(cfg.stac.url)
        self._cfg = cfg

    def search_scenes(
        self,
        bbox: BBox,
        datetime: str | None = None,
    ) -> list[pystac.Item]:
        """
        Datatime can be the whole year if nothing is specified or the provided period.
        """
        if datetime is None:
            year = self._cfg.aoi.year
            datetime = f"{year}-01-01/{year}-12-31"

        search = self._client.search(
            collections=[self._cfg.stac.collection],
            bbox=list(bbox),
            datetime=datetime,
            query={"eo:cloud_cover": {"lt": self._cfg.aoi.max_cloud_coverage}},
        )

        return list(search.items())


def get_data_profile(item_path: str, window_bbox: list | None = None) -> Profile:
    """Return the rasterio Profile for the raster at the provided path."""
    with rasterio.open(item_path) as src:
        if window_bbox is not None:
            # Transform from degree to meters.
            left, bottom, right, top = warp.transform_bounds(
                "EPSG:4326", src.crs, *window_bbox
            )

            window = from_bounds(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                transform=src.transform,
            )
        profile = src.profile

        # Height and width of the output shape are influenced by cropping
        height = int(window.height) if window_bbox else src.height
        width = int(window.width) if window_bbox else src.width

    profile["width"] = width
    profile["height"] = height

    return profile


def download_scene(
    out_file: Path,
    profile: Profile,
    bbox: BBox,
    target_res: float,
    assets: dict,
    bands: list[str],
) -> None:
    """
    Download a Setninel2 scene with composed bands.

    Parameters
    ----------
    bbox: BBox
        Window to apply to the original raster to get a subset of the data.

    Returns
    -------
    out: type
        description
    """
    with rasterio.open(out_file, "w", **profile) as dst:
        for idx, band in enumerate(bands):
            logger.info(f"starting download for band {band} and index {idx}")

            # Resolution information
            resolution = get_resolution_from_band_name(band)

            resampling_strategy = ResamplingStrategy(
                resolution / target_res,
                Resampling.bilinear if "SCL" not in band else Resampling.nearest,
            )

            data = get_scene(assets[band].href, bbox, resampling_strategy)

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
        raise ValueError(f"wrong width: expcted {profile['width']}, received {w}")

    if h != profile["height"]:
        raise ValueError(f"wrong height: expcted {profile['height']}, received {h}")


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


def get_scene(
    item_path: str,
    window_bbox: BBox | None = None,  # EPSG: 4326
    resampling_strategy: ResamplingStrategy = ResamplingStrategy(),
) -> np.ndarray:
    """Get the data associated with a scene with support for windowing and resampling."""
    window = None

    resampling_factor = resampling_strategy.get_factor()
    resampling_method = resampling_strategy.get_method()

    with rasterio.open(item_path) as src:
        if window_bbox is not None:
            # Transform from degree to meters.
            left, bottom, right, top = warp.transform_bounds(
                "EPSG:4326", src.crs, *window_bbox
            )

            window = from_bounds(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                transform=src.transform,
            )

        # Height and width of the outpur shape are influenced both by cropping and by
        # resampling.
        height = int((window.height if window else src.height) * resampling_factor)
        width = int((window.width if window else src.width) * resampling_factor)

        # An output shape is always created and used, so it should be ok to always set it here,
        # even with the original size.
        # https://github.com/rasterio/rasterio/blob/4e5bce88ea3c84b41a394244fe1cad6a5b8eb854/rasterio/_io.pyx#L544-L547
        data = src.read(
            1,
            window=window,
            out_shape=(height, width),
            resampling=resampling_method,
        )
    return data
