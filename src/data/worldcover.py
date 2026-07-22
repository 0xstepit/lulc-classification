import logging
import os
import urllib.request as ureq
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import xarray as xr
from rioxarray.merge import merge_arrays
from shapely.geometry import box

from src.config import WorldCoverConfig
from src.geometry import BoundingBox

logger = logging.getLogger("worldcover")


def get_worldcover_tile_ids(grid: gpd.GeoDataFrame, bbox: BoundingBox) -> list[str]:
    """Returns the WorldCover tile IDs from the provided grid that intersect with the
    bounding box.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        A table containing all the WorldCover grid and geometries.
    bbox : BoundingBox
        The bounding box used to select grid tiles.

    Returns
    -------
    list[str]
        The list of tile IDs used in the WorldCover grid system.

    """
    aoi_geom = box(*bbox)
    if grid.crs is None:
        raise ValueError("world cover grid must have a CRS")
    geom = gpd.GeoSeries(aoi_geom, crs="EPSG:4326").to_crs(grid.crs).iloc[0]
    aoi_grids = grid[grid.intersects(geom)]

    return list(aoi_grids["ll_tile"].values)


def create_worldcover_tile(
    cfg: WorldCoverConfig, tile_ids: list[str], target: xr.DataArray, out_file: Path
) -> None:
    if out_file.exists():
        logger.info(f"file {out_file} already exists, skipping")
        return

    worldcover_rasters = []
    for tile_id in tile_ids:
        url = get_worldcover_url(cfg.year, cfg.version, tile_id)
        worldcover_rasters.append(rxr.open_rasterio(url))

    worldcover_raster = merge_arrays(worldcover_rasters)
    worldcover_raster = worldcover_raster.rio.reproject_match(target)
    worldcover_raster = _reclass_raster(worldcover_raster, cfg.class_mapping)
    worldcover_raster.rio.to_raster(out_file)


def _reclass_raster(
    raster: xr.DataArray, class_mapping: dict[int, int]
) -> xr.DataArray:
    lut = np.zeros(256, dtype=np.uint8)
    for new_class, wc_class in class_mapping.items():
        lut[wc_class] = new_class
    reclassed = lut[raster.values.astype(np.uint8)]

    return raster.copy(data=reclassed)


# NOTE: this function is deprecated in favor of on-the-fly wc raster creation.
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
    """Return the URL identifying the location of the tile associated with the provided ID
    and for the specific year and WorldCover dataset version.

    Parameters
    ----------
    year : str
        The year of the WorldCover data.
    version : str
        The WorldCover version.
    tile_id : str
        The grid tile IDs in the WorldCover naming system.

    Returns
    -------
    str
        The URL of the resource associated with the provided inputs.

    """
    return (
        f"https://esa-worldcover.s3.amazonaws.com/{version}/{year}/map/"
        f"ESA_WorldCover_10m_{year}_{version}_{tile_id}_Map.tif"
    )
