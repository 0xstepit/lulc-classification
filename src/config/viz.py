import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Colors:
    text: str
    facecolor: str
    prop_cycle: dict[int, str]

    def __post_init__(self):
        object.__setattr__(
            self,
            "prop_cycle",
            {int(key): value for key, value in self.prop_cycle.items()},
        )


@dataclass(frozen=True)
class Font:
    size: dict[str, int]


@dataclass(frozen=True)
class VizConfig:
    colors: Colors
    font: Font


def load_viz_config(file_path: Path) -> VizConfig:
    """Load the visualizaqtion type configuration from disk.

    Parameters
    ----------
    file_path : Path
        The path pointing to the configuration file in `.toml` format.

    Returns
    -------
    VizConfig
        The configuration file for the visualization.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"file config {file_path} does not exists")

    with open(file_path, "rb") as f:
        _cfg = tomllib.load(f)
        colors = _cfg["colors"]
        font = _cfg["font"]

    return VizConfig(colors=Colors(**colors), font=Font(**font))
