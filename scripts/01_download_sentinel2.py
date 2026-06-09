"""
This script is used to download all the required images from the Sentinel2 mission database.
"""

import logging

import rasterio
from dotenv import load_dotenv

from src.config import load_sentinel2_config
from src.data.sentinel2 import (
    SentinelClient,
    download_scene,
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

    bbox = cfg.aoi.bounding_box
    if not bbox:
        raise ValueError("the bounding box of the aoi is not specified")

    items = client.search_scenes(bbox)

    item = items[0]
    assets = item.assets

    # We need the profile for an asset mathching the target resolution
    # to specify the profile for the target composed scene. We use the first
    # image of the desired resolution.
    ref_assets = assets[cfg.msi.bands[cfg.msi.target_resolution][0]].href
    profile = get_data_profile(ref_assets, cfg.aoi.bounding_box)

    # The final image will have one channel for each band of interested for the
    # considered tile.
    profile.update(count=cfg.msi.num_bands, driver="GTiff", compress="lzw", tiled=True)

    out_file = RAW_DIR / f"{item.id}{COMPOSED_SCENE_SUFFIX}"
    if out_file.exists():
        logger.info(f"output file {out_file} already exists, skipping")
    else:
        download_scene(
            out_file,
            profile,
            cfg.aoi.bounding_box,
            cfg.msi.target_resolution,
            assets,
            cfg.msi.get_bands(),
        )


if __name__ == "__main__":
    main()
