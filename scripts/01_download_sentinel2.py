"""Download imagey from the Sentinel-2 mission database and create the associated all bands raster.

The result of the execution is a series of scenes composed by all the
bands of interest resampled at the configured resolution.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

from lulc.config import load_config
from lulc.data.sentinel2 import (
    StacClient,
    create_all_bands_scene,
    get_data_profile,
)
from lulc.data.utils import evenly_spaced_indexes
from lulc.geometry import create_window_and_transform_from_bbox
from lulc.io import ALL_BANDS_SCENE_SUFFIX, GLOBAL_CONFIG, RAW_DATA_DIR
from lulc.logger import setup_logging

logger = logging.getLogger(__name__ if __name__ != "__main__" else Path(__file__).stem)


def main():  # noqa: D103
    # We access the Copernicus DB so we need the env variable for the S3-like access.
    load_dotenv()

    setup_logging()

    # Create folder to store raw Setninel-2 scenes if it does not exist yet.
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load config and create the Sentinel client for CDSE.
    cfg = load_config(GLOBAL_CONFIG)

    tiles_size = cfg.composites.tiles_size
    target_resolution = cfg.msi.target_resolution

    client = StacClient(cfg)

    bbox = cfg.aoi.selected.bounding_box
    if bbox is None:
        raise ValueError("the bounding box for the AOI is not specified in the config")

    # We have to create the rasterio profile to use in the composed images. To do so, we
    # get the profile associated with an image of the desired resolution, correct it
    # to take into account the AOI, and then update it with the desired custom
    # profile properties.
    items = client.search_items(bbox, grid_code=cfg.aoi.selected.tile)
    target_item = cfg.msi.bands[target_resolution][0]
    ref_item = items[0].assets[target_item].href
    profile = get_data_profile(ref_item)

    # Since we're interested only on a subset of the entire tile, we need to create a window for
    # tiled read, and construct a new affine transformation associated with the new tile position.
    window, transform = create_window_and_transform_from_bbox(
        bbox, profile["crs"], profile["transform"]
    )

    # The final image will have one channel for each band of interested.
    profile.update(
        count=cfg.msi.num_bands,
        driver="GTiff",
        compress="lzw",
        tiled=True,
        transform=transform,
        # We have to manually update profile geometric information.
        width=int(window.width),
        height=int(window.height),
        interleave="band",
        # Size of each block when we tile-read a raster.
        blockxsize=tiles_size,
        blockysize=tiles_size,
    )

    bands = cfg.msi.get_bands_list()

    for name, dates in cfg.composites.seasons.items():
        logger.info(f"starting all bands composition for season ({name})")

        # Create the folder to collect the scene for the current season.
        SEASON_DIR = RAW_DATA_DIR / name
        SEASON_DIR.mkdir(parents=True, exist_ok=True)

        items = client.search_items(
            bbox, f"{dates[0]}/{dates[1]}", cfg.aoi.selected.tile
        )

        logger.info(f"retrieved ({len(items)}) items")

        # If the number of returned scenes is higher than the fixed maximum, we
        # sample evenly the to have a uniform distribution across the season.
        filtered_idx = evenly_spaced_indexes(
            len(items), cfg.composites.max_scenes_per_season
        )
        items = [items[idx] for idx in filtered_idx]

        logger.info(f"selected ({len(items)}) items after filter")

        for item in items:
            out_file = SEASON_DIR / f"{item.id}{ALL_BANDS_SCENE_SUFFIX}"

            # The all bands scene is initially created in a temporary file and saved only
            # if the process completes successfully. This way we can run again the script
            # to generate the image is something go wrong. We need this tmp file because
            # the script skips the generation if it is already on disk.
            if out_file.exists():
                logger.info(f"skip: output file ({out_file}) already exists")
            else:
                tmp_file = out_file.with_suffix(".tmp.tif")
                try:
                    create_all_bands_scene(
                        tmp_file,
                        profile,
                        window,
                        target_resolution,
                        item.assets,
                        bands,
                    )
                    tmp_file.rename(out_file)
                except Exception:
                    # Delete temporary file if still exists.
                    if tmp_file.exists():
                        tmp_file.unlink()
                    raise


if __name__ == "__main__":
    main()
