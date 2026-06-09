"""
This script is used to download all the required images from the Sentinel2 mission database.
"""

import logging

import rasterio
from dotenv import load_dotenv

from src.config import load_sentinel2_config
from src.data.sentinel2 import (
    SentinelClient,
    create_window_from_bbox,
    download_all_bands_scene,
    evenly_spaced_indexes,
    get_data_profile,
    get_scene,
)
from src.io import RAW_DIR
from src.logger import setup_logging

# We access the Copernicus DB so we need the env variable for the S3-like access.
load_dotenv()

setup_logging()
logger = logging.getLogger("download_sentinel2")

COMPOSED_SCENE_SUFFIX = "_ALLBANDS.tif"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_sentinel2_config()
    client = SentinelClient(cfg)

    # Bounding box around the AoI.
    bbox = cfg.aoi.bounding_box
    if bbox is None:
        raise ValueError("the bounding box of the area of interest is not specified")

    bands = cfg.msi.get_bands()

    # We have to create the profile to use in the ALL_BANDS images. To do so, we
    # get the profile associated with an image of the desired resolution, correct it
    # to take into account the AoI, and then update it with the desired custom
    # profile properties.
    items = client.search_scenes(bbox)

    assets = items[0].assets
    # We use the first band of the desired resolution.
    # TODO: assumes target res is in the dict, add check
    ref_assets = assets[cfg.msi.bands[cfg.msi.target_resolution][0]].href

    profile = get_data_profile(ref_assets)

    window, cropped_transform = create_window_from_bbox(
        bbox, profile["crs"], profile["transform"]
    )

    # The final image will have one channel for each band of interested for the considered raster.
    profile.update(
        count=cfg.msi.num_bands,
        driver="GTiff",
        compress="lzw",
        tiled=True,
        transform=cropped_transform,
        #  We have to manually update profile geometric information.
        width=int(window.width),
        height=int(window.height),
        interleave="band",
    )

    for name, dates in cfg.composites.seasons.items():
        logger.info(f"starting all bands composition for season {name}")

        # Create the folder to collect the scene for the current season.
        SEASON_DIR = RAW_DIR / name
        SEASON_DIR.mkdir(parents=True, exist_ok=True)

        stac_datetime = f"{dates[0]}/{dates[1]}"
        items = client.search_scenes(bbox, stac_datetime)

        logger.info(f"retrieved {len(items)} items")

        filtered_idx = evenly_spaced_indexes(
            len(items), cfg.composites.max_scenes_per_season
        )
        items = [items[idx] for idx in filtered_idx]
        logger.info(f"retrieved {len(items)} items after filter")

        for item in items:
            out_file = SEASON_DIR / f"{item.id}{COMPOSED_SCENE_SUFFIX}"

            if out_file.exists():
                logger.info(f"output file {out_file} already exists, skipping")
            else:
                download_all_bands_scene(
                    out_file,
                    profile,
                    window,
                    cfg.msi.target_resolution,
                    item.assets,
                    bands,
                )


if __name__ == "__main__":
    main()
