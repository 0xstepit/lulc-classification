"""Project logger configuration functions."""

import logging

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO, force: bool = False) -> None:
    """Configure root logging for an enrty point.

    Call this once, from `main()`, before anything is logged and
    library modules under `lulc/` must never call this.

    Parameters
    ----------
    level : int
        Level of the logger.
    force : bool
        Specifies if any existing handler attached to the root logger must
        be removed or not.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
        force=force,
    )
    logging.captureWarnings(
        True
    )  # used to capture rasterio warnings in the same stream
