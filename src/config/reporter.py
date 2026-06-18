import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatetimeConfig:
    format: str


@dataclass
class JSONConfig:
    indent: int


@dataclass(frozen=True)
class ReporterConfig:
    writer: str
    file_config: JSONConfig
    dt: DatetimeConfig


def load_reporter_config(file_path: Path) -> ReporterConfig:
    """Load the reported type configuration from disk.

    Parameters
    ----------
    file_path : Path
        The path pointing to the configuration file in `.toml` format.

    Returns
    -------
    ReporterConfig
        The configuration file for the reporter.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"file config {file_path} does not exists")

    with open(file_path, "rb") as f:
        _cfg = tomllib.load(f)
        writer = _cfg["writer"]
        dt = DatetimeConfig(**_cfg.pop("dt"))

    if writer == "json":
        file_config = JSONConfig(**_cfg.pop("json"))
    else:
        raise ValueError(f"only JSON writer is supported")

    return ReporterConfig(writer=writer, dt=dt, file_config=file_config)
