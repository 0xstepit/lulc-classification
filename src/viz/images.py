import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from src.config.viz import VizConfig
from src.io import IMGS_DIR

IMGS_DIR.mkdir(parents=True, exist_ok=True)


def store_figure(title: str, transparent: bool = True):
    """Store the the active figure using the provided title as file name.

    Parameters
    ----------
    title : str
        The title of the figure that is converter into the file name without spaces.
    transparent : bool
        Whether the backgeound of the figure must be transparent or not
    """
    _title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(
        (IMGS_DIR / _title).with_suffix(".png"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=transparent,
    )


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
