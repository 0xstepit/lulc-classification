import logging
import os
import urllib.request as ureq
from pathlib import Path
from urllib.parse import urlparse
import numpy as np

import geopandas as gpd
from rasterio.profiles import Profile
from rasterio.warp import reproject
from shapely.geometry import box

from src.config import WorldCoverConfig
from src.geometry import BoundingBox

logger = logging.getLogger("worldcover")


def get_worldcover_tiles(grid: gpd.GeoDataFrame, bbox: BoundingBox) -> list[str]:
    aoi_geom = box(*bbox)
    geom = gpd.GeoSeries(aoi_geom, crs="EPSG:4326").to_crs(grid.crs).iloc[0]
    aoi_grids = grid[grid.intersects(geom)]
    return aoi_grids["ll_tile"].values


def download_tile(cfg: WorldCoverConfig, tile: str, out_dir: Path) -> None:
    url = get_worldcover_url(cfg.year, cfg.version, tile)

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    out_file = out_dir / filename

    if out_file.exists():
        logger.info(f"file {out_file} already exists, skipping")
        return

    logger.info(f"downloading tile {tile} at {url}")

    ureq.urlretrieve(url, out_file)


def get_worldcover_url(year: str, version: str, tile_id: str) -> str:
    return (
        f"https://esa-worldcover.s3.amazonaws.com/v200/2021/map/"
        f"ESA_WorldCover_10m_{year}_{version}_{tile_id}_Map.tif"
    )

def build_aligned_labels(tile_files: list[Path], ref_profile: Profile) -> np.ndarray:
