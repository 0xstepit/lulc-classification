import numpy as np


def create_seasonal_composite(scenes: list[np.ndarray]) -> np.ndarray:
    """Compute the pixel-wise NaN-aware median across scenes.

    Parameters
    ----------
    scenes : list[np.ndarray]
        List of 2D arrays of shape [H, W], one per scene.

    Returns
    -------
    np.ndarray
        Median array of shape [H, W].
    """
    stack = np.stack(scenes, axis=0)
    return np.nanmedian(stack, axis=0)
