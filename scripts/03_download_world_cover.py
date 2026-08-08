"""
Download the WorldCover data associated with the AOI. Since the WolrdCover uses a different tiling
system, the tiles intersectiong with our created seasonal raster are reprojected and merged to
create labels raster.
"""

import logging
from pathlib import Path

import geopandas as gpd
import rioxarray as rxr
import xarray as xr

from src.config import load_config, load_reporter_config
from src.data.worldcover import (
    compute_class_stats,
    create_worldcover_tile,
    get_worldcover_tile_ids,
)
from src.io import (
    GLOBAL_CONFIG,
    LABELS_DIR,
    REPORTER_CONFIG,
    REPORTS_DIR,
    SEASONAL_SCENES,
    WORLDCOVER_LABELS,
)
from src.logger import setup_logging
from src.reporter.reporter import Reporter

setup_logging()
logger = logging.getLogger(__name__ if __name__ != "__main__" else Path(__file__).stem)


def main():
    cfg = load_config(GLOBAL_CONFIG)

    # Load and instantiate analysis reporter.
    reporter = Reporter(REPORTS_DIR, load_reporter_config(REPORTER_CONFIG))

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # If the file is already present, short circuit termination.
    if WORLDCOVER_LABELS.exists():
        logger.info(f"[{WORLDCOVER_LABELS}] file already exists, skipping.")
        return

    grid = gpd.read_file(cfg.worldcover.grid_url)
    if cfg.aoi.selected.bounding_box is None:
        raise ValueError("the AOI bounding box must be specified for this script")
    tiles = get_worldcover_tile_ids(grid, cfg.aoi.selected.bounding_box)

    reporter.add("tiles_selection", tiles, bounding_box=cfg.aoi.selected.bounding_box)

    logger.info(f"AOI intersects with WorldCover tiles: [{tiles}]")

    composite = rxr.open_rasterio(SEASONAL_SCENES[0])
    if not isinstance(composite, xr.DataArray):  # to make basedpyright happy...
        raise TypeError(f"expected [{SEASONAL_SCENES[0]}] to open as a DataArray")

    labels = create_worldcover_tile(cfg.worldcover, tiles, composite)
    labels.rio.to_raster(WORLDCOVER_LABELS)

    report = compute_class_stats(WORLDCOVER_LABELS, cfg.worldcover.class_names)

    reporter.add("class_stats", report["per_class"])
    reporter.add("raster_stats", report["summary"])
    reporter.save("worldcover_raster.json")


if __name__ == "__main__":
    main()
