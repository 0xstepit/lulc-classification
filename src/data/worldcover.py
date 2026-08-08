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


def get_worldcover_tile_ids(grid: gpd.GeoDataFrame, bbox: BoundingBox) -> list:
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
    cfg: WorldCoverConfig, tile_ids: list[str], target: xr.DataArray
) -> xr.DataArray:

    worldcover_rasters = []
    for tile_id in tile_ids:
        url = get_worldcover_url(cfg.year, cfg.version, tile_id)
        worldcover_rasters.append(rxr.open_rasterio(url))

    worldcover_raster = merge_arrays(worldcover_rasters)
    worldcover_raster = worldcover_raster.rio.reproject_match(target)
    return _reclass_raster(worldcover_raster, cfg.class_mapping)


def compute_class_stats(file_path: Path, class_names: dict[int, str]) -> dict:
    nodata = 0

    composite = rxr.open_rasterio(file_path)
    if not isinstance(composite, xr.DataArray):  # to make basedpyright happy...
        raise TypeError(f"expected [{file_path}] to open as a DataArray")

    ids, counts = np.unique(composite, return_counts=True)
    count_by_id = dict(zip(ids, counts))

    total_pixels = int(composite.size)
    total_nodata = count_by_id.get(nodata, 0)
    total_valid = total_pixels - total_nodata
    per_class = []
    for cid, count in count_by_id.items():
        # Skip the nodata label
        if cid == nodata:
            continue
        per_class.append(
            {
                "id": int(cid),
                "name": class_names.get(int(cid), "unknown"),
                "pixels": int(count),
                "pct_total": round(100 * count / total_pixels, 3),
                "pct_valid": round(100 * count / total_valid, 3),
                "is_rare": bool(100 * count / total_valid < 0.5),
            }
        )
    present = {int(c) for c in count_by_id if c != nodata}
    valid_classes = set(class_names) - {nodata}
    present_counts = [c["pixels"] for c in per_class]

    summary = {
        "total_pixels": total_pixels,
        "total_valid": total_valid,
        "total_nodata": total_nodata,
        "nodata_pct": round(100 * total_nodata / total_pixels, 3),
        "n_classes_present": len(per_class),
        "missing_classes": sorted(valid_classes - present),
        "imbalance_ratio": round(max(present_counts) / min(present_counts), 2),
    }

    return {"per_class": per_class, "summary": summary}


def _reclass_raster(
    raster: xr.DataArray, class_mapping: dict[int, int]
) -> xr.DataArray:
    lut = np.zeros(256, dtype=np.uint8)
    for wc_class, new_class in class_mapping.items():
        lut[int(wc_class)] = int(new_class)
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
