""" """

import contextlib
import logging
from pathlib import Path

import numpy as np
import rasterio
from dotenv import load_dotenv

from src.config import load_config, load_reporter_config
from src.constants import SEASON_ORDER
from src.data.rasterio import (
    create_masked_bands_and_indices_tile,
    create_seasonal_profile,
)
from src.data.sentinel2 import (
    get_data_profile,
)
from src.data.utils import compute_nan_pct
from src.io import (
    ANALYSIS_DIR,
    GLOBAL_CONFIG,
    MULTISEASONAL_SCENE_DIR,
    RAW_DATA_DIR,
    REPORTER_CONFIG,
    SEASONAL_SCENE_DIR,
    SEASONAL_SCENE_SUFFIX,
)
from src.logger import setup_logging
from src.preprocessing.composites import create_seasonal_composite
from src.reporter.reporter import Reporter

# We access the Copernicus DB so we need the env variable for the S3-like access.
load_dotenv()

setup_logging()
logger = logging.getLogger("create_seasonal_composite")

SKIP_PARTIAL_BLOCKS = True


def main():
    cfg = load_config(GLOBAL_CONFIG)

    # Load and instantiate analysis reporter.
    cfg_reporter = load_reporter_config(REPORTER_CONFIG)
    reporter = Reporter(ANALYSIS_DIR, cfg_reporter)

    # Here we store all the composite scenes for each season.
    SEASONAL_SCENE_DIR.mkdir(parents=True, exist_ok=True)
    # Here we store the all seasons composite image.
    MULTISEASONAL_SCENE_DIR.mkdir(parents=True, exist_ok=True)

    season_ids = cfg.composites.seasons.keys()

    # Keep track of the single season file so we can quickly access them for the composite.
    seasonal_files = {}

    # Iterate over each season folder to preprocess season's scenes.
    for season_dir in sorted(RAW_DATA_DIR.iterdir()):
        # Ensure we only process data inside the season dirs.
        if not season_dir.is_dir() or season_dir.name not in list(season_ids):
            continue

        logger.info(f"processing files for season {season_dir.name}")

        out_file = SEASONAL_SCENE_DIR / f"{season_dir.name}{SEASONAL_SCENE_SUFFIX}"

        seasonal_files[season_dir.name] = out_file

        if out_file.exists():
            logger.info(f"output file {out_file} already exists, skipping")
            continue

        # Create an array containing all the scene for the considered season.
        files = [f for f in season_dir.glob("*.tif")]

        # No need here to update the affine transform since in case of scene cutting,
        # only the pixels far from the origin are removed.
        profile = create_seasonal_profile(
            get_data_profile(str(files[0])),
            13,
            cfg.composites.tiles_size,
            SKIP_PARTIAL_BLOCKS,
        )

        # Create a nested context containing all rasterio data reader needed. Since each season
        # can have a different number of scene, we use ExitStack to properly close their context.
        with contextlib.ExitStack() as stack:
            # Accumulate nested contexts.
            sources = [stack.enter_context(rasterio.open(f)) for f in files]

            with rasterio.open(out_file, "w", **profile) as dst:
                # We safely assume that all sources have the same shape and are on the
                # same region.
                for block_idx, window in sources[0].block_windows(1):
                    logger.info(f"processing block {block_idx}")

                    # Skip not full blocks if configured.
                    if SKIP_PARTIAL_BLOCKS and (window.width, window.height) != (
                        cfg.composites.tiles_size,
                        cfg.composites.tiles_size,
                    ):
                        logger.warning(
                            f"skipping block {block_idx} with shape {window.height}x{window.width}"
                        )
                        continue

                    # Used to store nan percentage info for each scene block.
                    file_percentage: list[dict[str, str | np.float16]] = []
                    block_scenes = []
                    for source in sources:
                        block_scene = create_masked_bands_and_indices_tile(
                            cfg, source, window
                        )
                        block_scenes.append(block_scene)

                        block_nan_pct = compute_nan_pct(block_scene)

                        file_percentage.append(
                            {"file": Path(source.name).name, "nan_pct": block_nan_pct}
                        )

                    reporter.add(
                        "scene_block_nan_pct",
                        file_percentage,
                        season=season_dir.name,
                        block=block_idx,
                    )

                    logger.info("started the creation of the seasonal composite")
                    season_composite_block = create_seasonal_composite(block_scenes)
                    logger.info("completed the creation of the seasonal composite")

                    composite_nan_pct = compute_nan_pct(season_composite_block)
                    logger.info(
                        f"percentage of NaN in composite is: {composite_nan_pct:.2f}%"
                    )

                    reporter.add(
                        "seasonal_composite_block_nan_pct",
                        {"nan_pct": composite_nan_pct},
                        season=season_dir.name,
                        block=block_idx,
                    )

                    logger.info("saving the block seasonal composite")
                    dst.write(season_composite_block, window=window)

    reporter.save("create_seasonal_composite.json")

    # Create the all seasons composite

    profile = get_data_profile(seasonal_files[SEASON_ORDER[0]])
    profile.update(count=profile["count"] * len(SEASON_ORDER))

    ordered_files = [seasonal_files[season] for season in SEASON_ORDER]

    MULTISEASONAL_SCENE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = MULTISEASONAL_SCENE_DIR / "composite.tif"

    if out_file.exists():
        logger.info(f"output file {out_file} already exists, skipping")
        return

    logger.info("starting all seasonal composite generation")
    with rasterio.open(out_file, "w", **profile) as dst:
        with contextlib.ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(f)) for f in ordered_files]

            for block_idx, window in sources[0].block_windows(1):
                block_scenes = [source.read(window=window) for source in sources]

                stack_seasons = np.concatenate(block_scenes, axis=0)

                dst.write(stack_seasons, window=window)


if __name__ == "__main__":
    main()
