import logging

import geopandas as gpd

from src.config import load_config
from src.data.sentinel2 import get_data_profile
from src.data.worldcover import download_tile, get_worldcover_tiles
from src.io import (
    GLOBAL_CONFIG,
    LABELS_DIR,
    MULTISEASONAL_SCENE_DIR,
    WORLDCOVER_LABELS,
    WORLDCOVER_RAW_DIR,
)
from src.logger import setup_logging

setup_logging()
logger = logging.getLogger("download_world_cover")


def main():
    cfg = load_config(GLOBAL_CONFIG)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    WORLDCOVER_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # If the file is already present, short circuit termination.
    if WORLDCOVER_LABELS.exists():
        logger.info(f"{WORLDCOVER_LABELS} file already exists, skipping.")
        return

    # TODO: file name should be a constant, not hardcoded...
    ref_profile = get_data_profile(str(MULTISEASONAL_SCENE_DIR / "composite.tif"))

    grid = gpd.read_file(cfg.worldcover.grid_url)
    tiles = get_worldcover_tiles(grid, cfg.aoi.bounding_box)

    logger.info(f"AOI intersects with WorldCover tiles: {tiles}")

    for tile in tiles:
        download_tile(cfg.worldcover, tile, WORLDCOVER_RAW_DIR)

    downloaded_files = WORLDCOVER_RAW_DIR.glob("*.tif")
    for downloaded_file in downloaded_files:



if __name__ == "__main__":
    main()
# def _compute_global_band_medians(files: list[Path], tiles_size: int) -> np.ndarray:
#     block_medians: list[np.ndarray] = []
#
#     with contextlib.ExitStack() as stack:
#         sources = [stack.enter_context(rasterio.open(f)) for f in files]
#
#         for _, window in sources[0].block_windows(1):
#             if (window.width, window.height) != (tiles_size, tiles_size):
#                 continue
#
#             for source in sources:
#                 block = source.read(window=window)
#                 block_medians.append(np.nanmedian(block, axis=(1, 2)))
#
#     return np.median(np.stack(block_medians, axis=0), axis=0)
#
#
# def _fill_seasonal_nans(
#     season_block: list[np.ndarray], global_band_medians: np.ndarray
# ) -> list[np.ndarray]:
#     season_stack = np.stack(season_block, axis=0)  # [n_seasons, C, H, W]
#
#     annual_median = np.nanmedian(season_stack, axis=0)  # [C, H, W]
#
#     still_nan = np.isnan(annual_median)
#     if still_nan.sum() > 0:
#         fallback = np.broadcast_to(
#             global_band_medians[:, None, None], annual_median.shape
#         )
#         annual_median[still_nan] = fallback[still_nan]
#     filled = np.where(np.isnan(season_stack), annual_median[None], season_stack)
#     return [filled[i] for i in range(filled.shape[0])]
