from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.figure import Figure

from src.config.viz import VizConfig
from src.io import IMGS_DIR


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


def set_matplotlib_global_config(cfg: VizConfig) -> None:
    """Set the matplotlib parameters in the provided configuration as global params.

    Parameters
    ----------
    cfg : VizConfig
        The config instance containing global params.
    """
    prop_cycle_colors = cfg.colors.prop_cycle
    colors = [prop_cycle_colors[idx] for idx in range(len(prop_cycle_colors.keys()))]
    mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)
    mpl.rcParams["figure.titlesize"] = cfg.font.size["title"]
    mpl.rcParams["axes.titlesize"] = cfg.font.size["subtitle"]
    mpl.rcParams["axes.labelsize"] = cfg.font.size["labels"]
    return
