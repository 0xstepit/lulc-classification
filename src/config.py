import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

CONFIG_DIR = Path(__file__).parent.parent / "config"

# Helper alias for  w | s | e | n
BBOX: TypeAlias = tuple[float, float, float, float]


@dataclass
class AoIConfig:
    name: str
    year: int
    max_cloud_coverage: float
    min_scenes: int
    min_scenes_per_season: int
    candidates: dict[str, list[float]]
    bounding_box: BBOX | None = None
    size: float = 50

    def __post_init__(self) -> None:
        if len(self.candidates) == 0:
            raise ValueError("There should be at least one area of interest specified")

        for k, v in self.candidates.items():
            if len(v) != 2:
                raise ValueError(
                    f"An area of interested is defined by lat/lon, received {len(v)} values for {k}"
                )


@dataclass(frozen=True)
class StacConfig:
    url: str
    collection: str

    def __post_init__(self) -> None:
        if self.url == "":
            raise ValueError("STAC endpoint URL cannot be empty")

        if self.collection == "":
            raise ValueError("STAC collection cannot be empty")


@dataclass
class Sentinel2Config:
    aoi: AoIConfig
    stac: StacConfig


# def deep_merge(base: dict, overrides: dict) -> dict:
#     """
#     Right join two configuration dictionaries.
#
#     Parameters
#     ----------
#     base: dict
#         Base configuration dictionary.
#     overrides: dict
#         The customized configuration dictionary.
#
#     Returns
#     -------
#     merged: dict
#         The right join of the two provided configuration dictionaries.
#
#     """
#     merged = base.copy()
#     for key, value in overrides.items():
#         # If the ooverride value is a dict and the base is dict, call recurrently. If not,
#         # just copy and override.
#         if isinstance(value, dict) and isinstance(merged.get(key), dict):
#             merged[key] = deep_merge(merged[key], value)
#         else:
#             merged[key] = value
#
#     return merged


def load_sentinel2_config(path: Path = CONFIG_DIR) -> Sentinel2Config:
    file_name = "sentinel2.toml"
    file_path = CONFIG_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File named {file_name} does not exists in {path}")

    with open(file_path, "rb") as f:
        cfg_ = tomllib.load(f)

        aoi = AoIConfig(**cfg_.pop("aoi"))
        stac = StacConfig(**cfg_.pop("stac"))

    return Sentinel2Config(aoi, stac)


# def load_config(files: list[str]) -> Config:
#     """
#     Main configurations loader.
#
#     Parameters
#     ----------
#     files: list[str]
#         The names of the configuration files to load
#
#     Returns
#     -------
#     config: dict
#         The merged configuration containing all the fields of provided files.
#
#     Raises
#     ------
#     FileNotFoundError
#         If one of the file in the input does not exist.
#
#     """
#     config = Config()
#
#     for file in files:
#         file_path = CONFIG_DIR / file
#         if not file_path.exists():
#             raise FileNotFoundError(
#                 f"File named {file} does not exists in {CONFIG_DIR}"
#             )
#
#         with open(file_path, "rb") as f:
#             tmp_config = tomllib.load(f)
#
#             stem = file_path.stem
#             if stem == "sentinel2":
#                 config.aoi = AoIConfig(**tmp_config.pop("aoi"))
#                 config.stac = StacConfig(**tmp_config.pop("stac"))
#             else:
#                 raise NotImplemented("The file is not supported yet")
#
#     return config
