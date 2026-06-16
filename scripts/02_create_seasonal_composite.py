""" """

import contextlib
import logging
from pathlib import Path

import numpy as np
import rasterio
from dotenv import load_dotenv
from rasterio.enums import Resampling
from rasterio.profiles import Profile
from rasterio.windows import Window

from src.config import Config, load_reporter_config, load_sentinel2_config
from src.data.sentinel2 import (
    get_data_profile,
)
from src.io import ANALYSIS_DIR, CONFIG_DIR, RAW_DIR, REPORTER_CONFIG
from src.logger import setup_logging
from src.preprocessing.composites import create_seasonal_composite
from src.preprocessing.indices import compute_indices
from src.preprocessing.masking import (
    create_scl_mask,
    get_masked_bands,
)
from src.reporter.reporter import Reporter

# We access the Copernicus DB so we need the env variable for the S3-like access.
load_dotenv()

setup_logging()
logger = logging.getLogger("create_seasonal_composite")

SEASONAL_SCENE_FOLDER = "seasonal"
SEASONAL_SCENE_SUFFIX = "_SEASONAL.tif"

# Defines the correct of the channels in the final composite image.
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def _process_window(
    src: rasterio.DatasetReader, window: Window, cfg: Config
) -> np.ndarray:
    data = src.read(window=window)

    mask = create_scl_mask(data, cfg.msi.get_scl_band_index(), cfg.msi.scl_mask_classes)

    masked_bands = get_masked_bands(data, cfg.msi.get_scl_band_index(), mask)

    # NOTE: we know that SCL was at the end of the data so indices positions are still valid
    # but it should be improved.
    indices = compute_indices(cfg.indices, masked_bands)

    return np.concatenate([masked_bands, indices], axis=0)


def _create_seasonal_profile(profile: Profile, tiles_size: int) -> Profile:
    """Update the provide reference profile to account for the 10 bands + 3 indices channels and
    the possible final W and H reduction due to not complete tile clearing.

    Parameters
    ----------
    profile : Profile
        Reference profile to update.
    tiles_size : int
        Size of the tile block used to read the image raster.

    Returns
    -------
    Profile
        The updated profile to use for each season median scene.

    """
    new_width = profile["width"] - profile["width"] % tiles_size
    new_height = profile["height"] - profile["height"] % tiles_size

    if new_width != profile["width"] or new_height != profile["height"]:
        logger.warning(
            "AOI extent is not a multiple of tiles_size=%d, dropping the trailing "
            "%dx%d px (right/bottom edge) from the seasonal composite",
            tiles_size,
            profile["width"] - new_width,
            profile["height"] - new_height,
        )

    # The transform's origin (upper-left corner of pixel (0, 0)) is unaffected:
    # block_windows() always starts at (0, 0), so only trailing right/bottom
    # blocks are ever skipped, never the leading ones.
    profile.update(count=13, dtype="float32", width=new_width, height=new_height)

    return profile


def _compute_global_band_medians(files: list[Path], tiles_size: int) -> np.ndarray:
    block_medians: list[np.ndarray] = []

    with contextlib.ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(f)) for f in files]

        for _, window in sources[0].block_windows(1):
            if (window.width, window.height) != (tiles_size, tiles_size):
                continue

            for source in sources:
                block = source.read(window=window)
                block_medians.append(np.nanmedian(block, axis=(1, 2)))

    return np.median(np.stack(block_medians, axis=0), axis=0)


def _fill_seasonal_nans(
    season_block: list[np.ndarray], global_band_medians: np.ndarray
) -> list[np.ndarray]:
    season_stack = np.stack(season_block, axis=0)  # [n_seasons, C, H, W]

    annual_median = np.nanmedian(season_stack, axis=0)  # [C, H, W]

    still_nan = np.isnan(annual_median)
    if still_nan.sum() > 0:
        fallback = np.broadcast_to(
            global_band_medians[:, None, None], annual_median.shape
        )
        annual_median[still_nan] = fallback[still_nan]
    filled = np.where(np.isnan(season_stack), annual_median[None], season_stack)
    return [filled[i] for i in range(filled.shape[0])]


def main():
    cfg = load_sentinel2_config()

    # Load and instantiate analysis reporter.
    cfg_reporter = load_reporter_config(REPORTER_CONFIG)
    reporter = Reporter(ANALYSIS_DIR, cfg_reporter)

    # Scenes folder configuraton.
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    SEASONAL_DIR = RAW_DIR / SEASONAL_SCENE_FOLDER
    SEASONAL_DIR.mkdir(parents=True, exist_ok=True)

    season_ids = cfg.composites.seasons.keys()

    seasonal_files = {}
    # Iterate over each season folder to preprocess season's scenes.
    for season_dir in sorted(RAW_DIR.iterdir()):
        # Ensure we only process data inside the season dirs.
        if not season_dir.is_dir() or season_dir.name not in list(season_ids):
            continue
        logger.info(f"processing files for season {season_dir.name}")

        out_file = SEASONAL_DIR / f"{season_dir.name + SEASONAL_SCENE_SUFFIX}"
        seasonal_files[season_dir.name] = out_file

        if out_file.exists():
            logger.info(f"output file {out_file} already exists, skipping")
            continue

        # Create an array containing all the considered season file path.
        files = [
            f
            for f in season_dir.glob("*.tif")
            if not f.name.endswith(SEASONAL_SCENE_SUFFIX)
        ]

        # TODO: We should update the transform too.
        profile = _create_seasonal_profile(
            get_data_profile(str(files[0])), cfg.composites.tiles_size
        )

        # Create a nested context containing all rasterio data reader needed. Since each season
        # can have a different number of scene, we use ExitStack to properly close them.
        with contextlib.ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(f)) for f in files]

            with rasterio.open(out_file, "w", **profile) as dst:
                # We safely assume that all sources have the same shape and are on the
                # same region.
                for ji, window in sources[0].block_windows(1):
                    logger.info(f"processing block {ji}")

                    # Skip not full blocks.
                    if (window.width, window.height) != (
                        cfg.composites.tiles_size,
                        cfg.composites.tiles_size,
                    ):
                        logger.warning(
                            f"skipping block {ji} with shape {window.height}x{window.width}"
                        )
                        continue

                    # Collect all the blocks from the season scene to compact.
                    # block_scenes = [
                    #     _process_window(src, window, cfg) for src in sources
                    # ]
                    file_percentage: list[dict[str, str | float]] = []
                    block_scenes = []
                    for source in sources:
                        block_scene = _process_window(source, window, cfg)
                        block_scenes.append(block_scene)

                        file = Path(source.name).name
                        perc_nan = round(
                            np.count_nonzero(np.isnan(block_scene))
                            / block_scene.size
                            * 100,
                            2,
                        )

                        file_percentage.append({"file": file, "nan_pct": perc_nan})

                    reporter.add(
                        "scene_block_nan_pct",
                        file_percentage,
                        season=season_dir.name,
                        block=ji,
                    )

                    logger.info(
                        "started the creation of the seasonal composite block image"
                    )
                    season_composite_block = create_seasonal_composite(block_scenes)
                    logger.info(
                        "completed the creation of the seasonal composite block image"
                    )

                    composite_nan_pct = round(
                        np.count_nonzero(np.isnan(season_composite_block))
                        / season_composite_block.size
                        * 100,
                        2,
                    )
                    logger.info(
                        f"percentage of NaN in composite is: {composite_nan_pct:.2f}%"
                    )

                    reporter.add(
                        "seasonal_composite_block_nan_pct",
                        {"nan_pct": composite_nan_pct},
                        season=season_dir.name,
                        block=ji,
                    )

                    logger.info("saving the block seasonal composite image")
                    dst.write(season_composite_block, window=window)

    reporter.save("create_seasonal_composite.json")

    # At this point we have the 4 seasonal composite and we can merge them into a single scene.
    # In doing so, we should proceed the same way and process each scene block by block.

    profile = get_data_profile(seasonal_files[SEASON_ORDER[0]])
    profile.update(count=profile["count"] * len(SEASON_ORDER))

    ordered_files = [seasonal_files[season] for season in SEASON_ORDER]

    out_file = SEASONAL_DIR / "composite.tif"

    global_band_medians = _compute_global_band_medians(
        ordered_files, cfg.composites.tiles_size
    )

    logger.info("starting seasonal composite genreation")
    with rasterio.open(out_file, "w", **profile) as dst:
        with contextlib.ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(f)) for f in ordered_files]

            for ji, window in sources[0].block_windows(1):
                block_scenes = [source.read(window=window) for source in sources]
                block_scenes = _fill_seasonal_nans(block_scenes, global_band_medians)

                stack_seasons = np.concatenate(block_scenes, axis=0)

                dst.write(stack_seasons, window=window)


if __name__ == "__main__":
    main()
