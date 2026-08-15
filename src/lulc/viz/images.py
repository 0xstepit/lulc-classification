"""Collection of functions to simplify plotting with uniform style."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from lulc.io import IMGS_DIR


def store_figure(
    title: str,
    img_dir: Path | None = None,
    transparent: bool = True,
    fig: Figure | None = None,
) -> Path:
    """Store the the active figure using the provided title as file name.

    Parameters
    ----------
    title : str
        The title of the figure that is converter into the file name without spaces.
    transparent : bool
        Whether the backgeound of the figure must be transparent or not
    """
    dir = IMGS_DIR if img_dir is None else img_dir
    dir.mkdir(parents=True, exist_ok=True)

    _title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    (fig if fig is not None else plt.gcf()).savefig(
        (dir / _title).with_suffix(".png"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=transparent,
    )

    return dir


def robust_plot(data, bounds: list[float], cmap: str = "gray") -> None:
    """Plot the provided data with using P2 and P98 percentiles."""
    _, ax = plt.subplots(figsize=(6, 6))
    vmin, vmax = np.nanpercentile(data, [2, 98])
    left, bottom, right, top = bounds
    ax.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        extent=(left, right, bottom, top),
        cmap=cmap,
    )
