"""Shared value objects for the project.

This module sits at the bottom of the dependency graph: it must never import
from any other `lulc` subpackage apart from `lulc.constants`, and must never
import a third-party library that pulls in IO (rasterio, pystac, boto3).

Types that are exchanged between subpackages live here so that, for example,
the reporting layer can describe a scene count without importing the
Sentinel-2 acquisition layer and everything it depends on.
"""

from dataclasses import dataclass, field

from lulc.constants import SEASON_MONTHS

# Helper alias for  w | s | e | n
type BoundingBox = tuple[float, float, float, float]


@dataclass
class SceneCounts:
    """Contains information associated with the number of scenes found for each season.

    Attributes
    ----------
    total : int
        The overall number of scenes.
    by_season : dict[str, int]
        The number of available scenes per season.
    """

    total: int = 0
    by_season: dict = field(
        default_factory=lambda: dict.fromkeys(
            SEASON_MONTHS.keys(), 0
        )  # {"DJF": 0, "MAM": 0, "JJA": 0, "SON": 0}
    )

    def increment_counter(self, season: str):
        """Increment the number of total scenes and scenes for the provided season."""
        if season not in self.by_season.keys():
            raise KeyError(
                f"season {season} is not valid; valid seasons are {self.by_season.keys()}"
            )
        self.by_season[season] += 1
        self.total += 1
