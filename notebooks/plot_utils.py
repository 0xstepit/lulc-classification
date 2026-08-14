import matplotlib.pyplot as plt

from lulc.io import IMGS_DIR

IMGS_DIR.mkdir(parents=True, exist_ok=True)


def store_figure(title: str):
    plt.savefig(
        (IMGS_DIR / title.lower().replace(" ", "_")).with_suffix(".png"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
    )
