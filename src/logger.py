import logging

from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> None:
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
        handlers=[RichHandler(rich_tracebacks=True)],
    )
