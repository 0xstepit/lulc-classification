""" """

import logging

import numpy as np
import rasterio
from dotenv import load_dotenv

from src.config import load_sentinel2_config
from src.data.sentinel2 import (
    download_all_bands_scene,
    get_data_profile,
)
from src.data.utils import evenly_spaced_indexes
from src.geometry import create_window_from_bbox
from src.io import RAW_DIR
from src.logger import setup_logging
from src.preprocessing.composites import create_seasonal_composite
from src.preprocessing.indices import compute_ndbi, compute_ndvi, compute_ndwi
from src.preprocessing.masking import apply_scl_mask, create_scl_mask

# We access the Copernicus DB so we need the env variable for the S3-like access.
load_dotenv()

setup_logging()
logger = logging.getLogger("create_seasonal_composite")

SEASONAL_SCENE_SUFFIX = "_SEASONAL.tif"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_sentinel2_config()
    scl_band_idx = cfg.msi.get_scl_band_index()
    mask_classes = cfg.msi.scl_mask_classes

    for season_dir in sorted(RAW_DIR.iterdir()):
        if not season_dir.is_dir():
            continue
        out_file = season_dir / f"{SEASONAL_SCENE_SUFFIX}"
        files = list(season_dir.glob("*.tif"))

        profile = get_data_profile(str(files[0]))

        scenes = []
        for file in files:
            logger.info(f"processing file {file}")
            with rasterio.open(file, "r") as src:
                data = src.read()

            mask = create_scl_mask(data, scl_band_idx, mask_classes)
            masked_data = apply_scl_mask(data, scl_band_idx, mask)

            # NOTE: we know that SCL was at the end of the data so indices are still valid
            # but it should be improved.

            # Compute NDVI index
            nir = masked_data[cfg.indices.get_channel("ndvi", "nir"), :, :]
            red = masked_data[cfg.indices.get_channel("ndvi", "red"), :, :]
            ndvi = compute_ndvi(nir, red)[
                np.newaxis
            ]  # we add a channel axis needed for the concatenation later.

            # Compute NDBI index
            swir = masked_data[cfg.indices.get_channel("ndbi", "swir"), :, :]
            nir = masked_data[cfg.indices.get_channel("ndbi", "nir"), :, :]
            ndbi = compute_ndbi(swir, nir)[np.newaxis]

            # Compute NDWI index
            green = masked_data[cfg.indices.get_channel("ndwi", "green"), :, :]
            nir = masked_data[cfg.indices.get_channel("ndwi", "nir"), :, :]
            ndwi = compute_ndwi(green, nir)[np.newaxis]

            scenes.append(np.concatenate([masked_data, ndvi, ndbi, ndwi], axis=0))
            logger.info(f"file processing completed")

        logger.info("started the creation of the seasonal composite image")
        season_composite = create_seasonal_composite(scenes)
        logger.info("completed the creation of the seasonal composite image")

        logger.info("saving the seasonal composite image")
        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(season_composite)

        break


if __name__ == "__main__":
    main()
