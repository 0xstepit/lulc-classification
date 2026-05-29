"""
This script is used to download all the required images from the Sentinel2 mission database.
"""

import os

import rasterio
from dotenv import load_dotenv

from src.config import load_sentinel2_config
from src.data.sentinel2 import SentinelClient, get_data_profile, get_scene
from src.io import RAW_DIR

# We access the Copernicus DB so we need the env variable for the S3-like access.
load_dotenv()


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
    # to specify the profile for the target composed scene.
    ref_assets = assets[cfg.msi.bands[cfg.msi.target_resolution][0]].href
    profile = get_data_profile(ref_assets)
    profile.update(count=cfg.msi.num_bands, compress="lzw")

    with rasterio.open(
        RAW_DIR / f"{item.id}{COMPOSED_SCENE_SUFFIX}", "w", **profile
    ) as dst:
        data = get_scene(href)
        dst.write(data, 1)

    # get_scene(items[0].href)


if __name__ == "__main__":
    main()
