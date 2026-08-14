"""Create a seasonal composite rasters.

A seasonal composite raster is created for each season by combining each season
images into a median aggregate, and then by stacking together all the bands for
each seasons and spectral indexes into a single raster.
"""

# TODO: rescale properly reflectances

import contextlib
import logging
from pathlib import Path

import rasterio
from dotenv import load_dotenv

from lulc.config import load_config, load_reporter_config
from lulc.constants import seasonal_band_names
from lulc.data.raster import (
    create_masked_bands_and_indices_tile,
    create_seasonal_profile,
)
from lulc.data.sentinel2 import (
    get_data_profile,
)
from lulc.data.utils import compute_nan_pct
from lulc.io import (
    GLOBAL_CONFIG,
    RAW_DATA_DIR,
    REPORTER_CONFIG,
    REPORTS_DIR,
    SEASONAL_SCENE_DIR,
    SEASONAL_SCENE_SUFFIX,
)
from lulc.logger import setup_logging
from lulc.preprocessing.composites import create_seasonal_composite
from lulc.provenance import stamp
from lulc.reporter.reporter import Reporter

logger = logging.getLogger(__name__ if __name__ != "__main__" else Path(__file__).stem)


def main():  # noqa: D103
    # We access the Copernicus DB so we need the env variable for the S3-like access.
    load_dotenv()

    setup_logging()
    cfg = load_config(GLOBAL_CONFIG)

    reporter = Reporter(REPORTS_DIR, load_reporter_config(REPORTER_CONFIG))

    # Here we store all the composite scenes for each season.
    SEASONAL_SCENE_DIR.mkdir(parents=True, exist_ok=True)

    season_ids = cfg.composites.seasons.keys()

    # We remove the SCL band from the channels when we create the composite.
    num_channels = cfg.msi.num_bands - 1 + len(cfg.indices.bands_to_channels.keys())

    # Iterate over each season folder to preprocess season's scenes.
    for season_dir in sorted(RAW_DATA_DIR.iterdir()):
        # Ensure we only process data inside the season dirs.
        if not season_dir.is_dir() or season_dir.name not in list(season_ids):
            continue

        logger.info(f"processing scenes for season ({season_dir.name})")

        out_file = SEASONAL_SCENE_DIR / f"{season_dir.name}{SEASONAL_SCENE_SUFFIX}"

        if out_file.exists():
            logger.info(f"skip: output file ({out_file}) already exists")
            continue

        # Create an array containing all the scene paths for the considered season.
        files = list(season_dir.glob("*.tif"))

        # No need here to update the affine transform since in case of scene cutting,
        # only the pixels far from the origin are removed.
        profile = create_seasonal_profile(
            get_data_profile(str(files[0])),
            num_channels,
            cfg.composites.tiles_size,
            cfg.composites.skip_partial_blocks,
        )

        # Create a nested context containing all rasterio data reader needed. Since each season
        # can have a different number of scenes, we use ExitStack to properly close their context.
        with contextlib.ExitStack() as stack:
            # Accumulate nested contexts.
            sources = [stack.enter_context(rasterio.open(f)) for f in files]

            with rasterio.open(out_file, "w", **profile) as dst:
                # Add band names into the file.
                stamp(
                    dst,
                    cfg,
                    seasonal_band_names(cfg.msi),
                    stage="seasonal_composite",
                    season=season_dir.name,
                    n_scenes=str(len(files)),
                )

                # We safely assume that all sources have the same shape and are on the
                # same region.
                for block_idx, window in sources[0].block_windows(1):
                    logger.info(f"processing block {block_idx}")

                    # Skip not full blocks if configured.
                    if cfg.composites.skip_partial_blocks and (
                        (window.width, window.height)
                        != (
                            cfg.composites.tiles_size,
                            cfg.composites.tiles_size,
                        )
                    ):
                        logger.warning(
                            f"skipping block ({block_idx}) with shape "
                            f"({window.height}x{window.width}) because not full"
                        )
                        continue

                    # Used to store nan percentage info for each scene block.
                    file_percentage: list[dict[str, str | float]] = []
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


if __name__ == "__main__":
    main()
