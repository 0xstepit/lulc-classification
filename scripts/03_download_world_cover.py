import logging
from pathlib import Path

import geopandas as gpd
import rioxarray as rxr
import xarray as xr

from src.config import load_config
from src.data.worldcover import (
    create_worldcover_tile,
    get_worldcover_tile_ids,
)
from src.io import (
    GLOBAL_CONFIG,
    LABELS_DIR,
    MULTISEASONAL_SCENE,
    WORLDCOVER_LABELS,
    WORLDCOVER_RAW_DIR,
)
from src.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__ if __name__ != "__main__" else Path(__file__).stem)


def main():
    cfg = load_config(GLOBAL_CONFIG)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    WORLDCOVER_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # If the file is already present, short circuit termination.
    if WORLDCOVER_LABELS.exists():
        logger.info(f"[{WORLDCOVER_LABELS}] file already exists, skipping.")
        return

    grid = gpd.read_file(cfg.worldcover.grid_url)
    if cfg.aoi.bounding_box is None:
        raise ValueError("the AOI bounding box must be specified for this script")
    tiles = get_worldcover_tile_ids(grid, cfg.aoi.bounding_box)

    logger.info(f"AOI intersects with WorldCover tiles: [{tiles}]")

    composite = rxr.open_rasterio(MULTISEASONAL_SCENE)
    if not isinstance(composite, xr.DataArray):  # to make basedpyright happy...
        raise TypeError(f"expected [{MULTISEASONAL_SCENE}] to open as a DataArray")

    create_worldcover_tile(cfg.worldcover, tiles, composite, WORLDCOVER_LABELS)


if __name__ == "__main__":
    main()
