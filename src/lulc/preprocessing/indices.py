import numpy as np

from lulc.config import IndicesConfig


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Difference Vegetation Index (NDVI).

    Parameters
    ----------
    nir : np.ndarray
        Near-infrared band (Sentinel-2 B08).
    red : np.ndarray
        Red band (Sentinel-2 B04).

    Returns
    -------
    np.ndarray
        NDVI values in [-1, 1]. Pixels where NIR + red = 0 are set to NaN.

    Notes
    -----
    NDVI = (NIR - red) / (NIR + red)
    """
    num = nir - red
    den = nir + red

    return np.divide(
        num, den, out=np.full_like(num, np.nan, dtype=np.float32), where=den != 0
    )


# NOTE: what was the difference if we used the other SWIR instead?
def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Difference Build-Up Index (NDBI).

    Parameters
    ----------
    swir : np.ndarray
        Swir band (Sentinel-2 B11).
    nir : np.ndarray
        Near-infrared band (Sentinel-2 B08).

    Returns
    -------
    np.ndarray
        NDBI values in [-1, 1]. Pixels where SWIR + NIR = 0 are set to NaN.

    Notes
    -----
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    """
    num = swir - nir
    den = swir + nir

    return np.divide(
        num, den, out=np.full_like(num, np.nan, dtype=np.float32), where=den != 0
    )


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Difference Water Index (NDWI).

    Parameters
    ----------
    green : np.ndarray
        Green band (Sentinel-2 B03).
    nir : np.ndarray
        Near-infrared band (Sentinel-2 B08).

    Returns
    -------
    np.ndarray
        NDWI values in [-1, 1]. Pixels where Green + NIR = 0 are set to NaN.

    Notes
    -----
    NDWI = (Green - NIR) / (Green + NIR)
    """
    num = green - nir
    den = green + nir

    return np.divide(
        num, den, out=np.full_like(num, np.nan, dtype=np.float32), where=den != 0
    )


def compute_indices(cfg: IndicesConfig, data: np.ndarray) -> np.ndarray:
    """Computes a numpy array containing the NDVI, NDBI, and NDWI for the provided raster.

    Parameters
    ----------
    cfg : Indices
        A configuration class providing the position in a raster for each band used in the indices.
    data : np.ndarray
        The raster used for indices computation of size [H, W]

    Returns
    -------
    np.ndarray
        An array of size [NUM_INDICES, H, W] contianin the stacked indices.

    """
    # Compute NDVI index
    # we add a channel axis needed for the concatenation later.
    nir = data[cfg.get_channel("ndvi", "nir"), :, :]
    red = data[cfg.get_channel("ndvi", "red"), :, :]
    ndvi = compute_ndvi(nir, red)[np.newaxis]

    # Compute NDBI index
    swir = data[cfg.get_channel("ndbi", "swir"), :, :]
    nir = data[cfg.get_channel("ndbi", "nir"), :, :]
    ndbi = compute_ndbi(swir, nir)[np.newaxis]

    # Compute NDWI index
    green = data[cfg.get_channel("ndwi", "green"), :, :]
    nir = data[cfg.get_channel("ndwi", "nir"), :, :]
    ndwi = compute_ndwi(green, nir)[np.newaxis]

    return np.concatenate([ndvi, ndbi, ndwi], axis=0)
