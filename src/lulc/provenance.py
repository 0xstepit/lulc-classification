"""Provenance information stamped into every raster the pipeline writes."""

import logging
import subprocess
from datetime import UTC, datetime

from lulc.config import Config

logger = logging.getLogger(__name__)


def git_revision() -> dict[str, str]:
    """Create a git revision with a sha and a status."""
    try:
        # Returns the sha of the current HEAD
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return {"git_sha": "unknown", "git_dirty": "unknown"}

    return {"git_sha": sha, "git_dirty": str(bool(dirty)).lower()}


def print_git_revision() -> None:
    """Print the git revision to stdout."""
    print(f"{git_revision()}")


def raster_tags(cfg: Config, channels: list[str], **extra: str) -> dict[str, str]:
    """Build the GDAL tag set written into every raster.

    Parameters
    ----------
    cfg : Config
        Supplies the AOI identity.
    channels : list[str]
        Channel names of *this* raster, not the acquisition band order: a scene
        carries 11 bands including SCL, a seasonal composite carries 13 because
        masking consumes SCL and three indices are appended.
    **extra : str
        Additional tags. GDAL stores tags as strings only.

    Returns
    -------
    dict[str, str]
        The tags to hand to `rasterio.DatasetWriter.update_tags`.
    """
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "channel_names": ",".join(channels),
        "n_channels": str(len(channels)),
        "aoi": cfg.aoi.selected.name,
        **git_revision(),
        **extra,
    }


def stamp(dst, cfg: Config, channels: list[str], **extra: str) -> None:
    """Write channel names and provenance into an open rasterio dataset."""
    dst.descriptions = tuple(channels)
    dst.update_tags(**raster_tags(cfg, channels, **extra))
