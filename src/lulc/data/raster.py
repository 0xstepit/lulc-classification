"""Collection of functions and classes to work with rasters."""

import contextlib
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.profiles import Profile
from rasterio.windows import Window

from lulc.config import Config
from lulc.data.sentinel2 import rescale_reflectances
from lulc.preprocessing.indices import compute_indices
from lulc.preprocessing.masking import (
    create_scl_mask,
    get_masked_bands,
)

logger = logging.getLogger(__name__)


def create_seasonal_profile(
    profile: Profile, num_channels: int, tiles_size: int, discard_partial: bool = True
) -> Profile:
    """Update the provided reference profile for the seasonal composite.

    The updated profile is created taking into account for the 10 reflectance bands and
    3 indices channels and the possible final W and H reduction due to not complete
    tile clearing.

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


# TODO: this function should be splitted
def create_masked_bands_and_indices_tile(
    cfg: Config, src: rasterio.DatasetReader, window: Window
) -> np.ndarray:
    """Create a raster with SCL mask applied and stacked spectral indices.

    The raster is created only for the window provided and the SCL channel is removed
    from the bands in the tile.

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

    mask = create_scl_mask(data, cfg.msi.scl_band_index, cfg.msi.scl_mask_classes)
    masked_bands = get_masked_bands(data, cfg.msi.scl_band_index, mask)

    rescaled_bands = rescale_reflectances(masked_bands)

    # NOTE: we know that SCL was at the end of the data so indices positions are still valid
    # but it should be improved.
    indices = compute_indices(cfg.indices, rescaled_bands)

    return np.concatenate([rescaled_bands, indices], axis=0)


class SeasonalStack:
    """Class manager for raster reads with many underlying raster sources.

    This class is used to read from multiple source patches of rasters associated with the same
    window. The class can be used to genreate an all seasons composite raaster out of single seasons.
    """

    def __init__(self, files: list[Path]) -> None:
        if not files:
            raise ValueError("no seasonal files provided")

        self._files = files
        self._exit_stack = contextlib.ExitStack()
        self._sources: list[rasterio.DatasetReader] = []

    def __enter__(self) -> SeasonalStack:
        """Openthe contextes associated with underlying sources."""
        self._sources = [
            self._exit_stack.enter_context(rasterio.open(f)) for f in self._files
        ]
        self._validate_alignment()
        return self

    def __exit__(self, *exec_info) -> None:
        """Close all the contextes associated with opened sources."""
        self._exit_stack.close()

    def _validate_alignment(self) -> None:
        reference = self._sources[0]

        for source in self._sources[1:]:
            if (source.width, source.height) != (reference.width, reference.height):
                raise ValueError(
                    f"seasonal scene ({source.name}) has shape "
                    f"({source.width}, {source.height}), expected "
                    f"({reference.width}, {reference.height})."
                )

            if source.crs != reference.crs:
                raise ValueError(
                    f"seasonal scene ({source.name}) has CRS {source.crs}, "
                    f"expected ({reference.crs})."
                )

            if not source.transform.almost_equals(reference.transform):
                raise ValueError(
                    f"seasonal scene ({source.name}) is not aligned with scene({reference.name})"
                )

    @property
    def width(self) -> int:
        """Return the underlying rasters width."""
        return self._sources[0].width

    @property
    def height(self) -> int:
        """Return the underlying rasters height."""
        return self._sources[0].height

    @property
    def shape(self) -> tuple[int, int]:
        """Return the underlying rasters shape."""
        return self._sources[0].shape

    @property
    def crs(self):
        """Return the CRS used."""
        return self._sources[0].crs

    @property
    def transform(self):
        """Return the transformation used."""
        return self._sources[0].transform

    @property
    def count(self):
        """Return the number of sources used."""
        return sum(source.count for source in self._sources)

    def block_window(self, band_index: int):
        """Return the block window used for the provided band index."""
        return self._sources[0].block_windows(band_index)

    def read(self, window: Window | None = None) -> np.ndarray:
        """Read a possibly patched window seasonal composite raster.

        Read from all the sources a possibly windowed patch, and return their
        combination as a stacked array.
        """
        return np.concatenate(
            [source.read(window=window) for source in self._sources], axis=0
        )
