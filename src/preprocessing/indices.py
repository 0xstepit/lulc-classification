import numpy as np


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
