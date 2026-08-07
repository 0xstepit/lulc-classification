import numpy as np


def create_scl_mask(
    data: np.ndarray, scl_band_index: int, mask_classes: list[int]
) -> np.ndarray:
    """Create a mask based on the specified SCL map classes.

    Parameters
    ----------
    data : np.ndarray
        The all bands raster.
    scl_band_index : int
        The index of the channel associated with the SCL map
    mask_classes : list[int]
        The ID of the classes we want to filter out.

    Returns
    -------
    np.ndarray
        A boolean array containing True where the pixel has to be filtered out.
    """
    scl = data[scl_band_index]
    scl_mask = np.isin(scl, mask_classes)

    return scl_mask


def get_masked_bands(
    data: np.ndarray, scl_band_index: int, mask: np.ndarray
) -> np.ndarray:
    """Apply a mask to the multi-band raster. The function first remove the SCL map from the raster,
    then apply the mask. Since masked-out pixels are casted to np.nan, the entire raster is converted
    to np.float32.

    Parameters
    ----------
    data : np.ndarray
        The raster containing the bands and map.
    scl_band_index : int
        The index in the raster associated with the SCL map.
    mask : np.ndarray
        The mask to apply.

    Returns
    -------
    np.ndarray
        A raster containing only filtered multispectral bands. If the shape of the input array is
        (C, H, W), the shape of the output is (C-1, H, W).

    """
    if scl_band_index > data.shape[0] - 1:
        raise ValueError(f"scl_band_index {scl_band_index} is out of bands")

    if data.shape[0] == 1:
        raise ValueError(f"provided array must have >1 dimension")

    # Since NaN is a float, we have to cast the raster data.
    bands = np.concatenate(
        [data[:scl_band_index], data[scl_band_index + 1 :]], axis=0
    ).astype(np.float32)
    bands[:, mask] = np.nan

    return bands
