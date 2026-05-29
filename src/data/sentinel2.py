from pathlib import Path

import pystac
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling

from src.config import BBOX, Sentinel2Config


class SentinelClient:
    def __init__(self, cfg: Sentinel2Config) -> None:
        self._client = Client.open(cfg.stac.url)
        self._cfg = cfg

    def search_scenes(self, bbox: BBOX) -> list[pystac.Item]:
        year = self._cfg.aoi.year

        search = self._client.search(
            collections=[self._cfg.stac.collection],
            bbox=list(bbox),
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": self._cfg.aoi.max_cloud_coverage}},
        )

        return list(search.items())


def get_data_profile(item_path: str) -> dict:
    with rasterio.open(item_path) as src:
        profile = src.profile
    return profile


def get_scene(
    item_path: str,
    resampling_factor: float = 1,
    resampling_method: Resampling = Resampling.nearest,
) -> None:
    with rasterio.open(item_path) as src:
        # An output shape is always created and used so it should be ok to always set it here,
        # even with the original size.
        # https://github.com/rasterio/rasterio/blob/4e5bce88ea3c84b41a394244fe1cad6a5b8eb854/rasterio/_io.pyx#L544-L547
        data = src.read(
            1,
            out_shape=(
                src.count,
                int(src.height * resampling_factor),
                int(src.width * resampling_factor),
            ),
            resampling=resampling_method,
        )
    return data
