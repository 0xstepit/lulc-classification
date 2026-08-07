import logging

import numpy as np
import rasterio
from rasterio.profiles import Profile
from rasterio.windows import Window

from src.config import Config
from src.data.sentinel2 import rescale_reflectances
from src.preprocessing.indices import compute_indices
from src.preprocessing.masking import (
    create_scl_mask,
    get_masked_bands,
)

logger = logging.getLogger(__name__)


def create_seasonal_profile(
    profile: Profile, num_channels: int, tiles_size: int, discard_partial: bool = True
) -> Profile:
    """Update the provide reference profile to account for the 10 bands + 3 indices channels and
    the possible final W and H reduction due to not complete tile clearing.

    Parameters
    ----------
    profile : Profile
        Reference profile to update.
    num_channels: int
        Number of channels in the final raster.
    tiles_size : int
        Size of the tile block used to read the image raster.
    discard_partial: bool
        Wether non full tiles have to be discarded or not.

    Returns
    -------
    Profile
        The updated profile to use for each season median scene.
    """
    # We take into account that the windowed read of the raster can have window that are not
    # full size.
    if discard_partial:
        new_width = profile["width"] - profile["width"] % tiles_size
        new_height = profile["height"] - profile["height"] % tiles_size
    else:
        new_width = profile["width"]
        new_height = profile["height"]

    if new_width != profile["width"] or new_height != profile["height"]:
        logger.warning(
            f"AOI extent is not a multiple of tiles_size ({tiles_size}), "
            f"dropping the trailing {profile['width'] - new_width}x{profile['height'] - new_height} px "
            "(right/bottom edge) from the seasonal composite",
        )

    # The transform's origin (upper-left corner of pixel (0, 0)) is unaffected:
    # block_windows() always starts at (0, 0), so only trailing right/bottom
    # blocks are ever skipped, never the leading ones.
    profile.update(
        count=num_channels, dtype="float32", width=new_width, height=new_height
    )

    return profile


def create_masked_bands_and_indices_tile(
    cfg: Config, src: rasterio.DatasetReader, window: Window
) -> np.ndarray:
    """Apply an SCL mask to a windowed tile of the source file and stack bands and indices. The
    SCL channel is removed from the tile.

    Parameters
    ----------
    cfg : Config
        Configuration structure.
    src : rasterio.DatasetReader
        Rasterio DatasetReader with tiling read capability. Each tile of the raster has size
        [C, H_t, W_t].
    window : Window
        Window to read from the raster.

    Returns
    -------
    np.ndarray
        An array containing the masked bands and indices with size
        [C_t, H_t, W_t] where C_t is C - 1 (SCL band) + C_i (number of indices).
    """
    # Windowed read.
    data = src.read(window=window)

    mask = create_scl_mask(data, cfg.msi.get_scl_band_index(), cfg.msi.scl_mask_classes)
    masked_bands = get_masked_bands(data, cfg.msi.get_scl_band_index(), mask)

    rescaled_bands = rescale_reflectances(masked_bands)

    # NOTE: we know that SCL was at the end of the data so indices positions are still valid
    # but it should be improved.
    indices = compute_indices(cfg.indices, rescaled_bands)

    return np.concatenate([rescaled_bands, indices], axis=0)
