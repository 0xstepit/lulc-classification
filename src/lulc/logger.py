import logging

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO, force: bool = False) -> None:
    """Common logging configuration handler. This function has to be called
    before every logger instantiation in all the files that require logs.

    Parameters
    ----------
    level : int
        Level of the logger.

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
