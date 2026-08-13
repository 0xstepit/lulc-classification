from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from lulc.domain import BoundingBox, SceneCounts


@dataclass
class ReportEntry:
    """A generic report entry.

    Attributes
    ----------
    step : str
        A name used to identify the phase for which the report has been generated.
    data : Any
        Data that is added into the report.
    timestamp : str
        Timestamp of the report generation.
    metadata : dict
        Additional metadata to add along with the data.

    """

    step: str
    data: Any
    # We should consider to use datetime in config instead of the string so
    # Ruff can catch function calls.
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class CandidateResult:
    """Defines the analysis result for a single AOI candidate. Since the bounding box can
    intersect with different tiles, it is important to specify the grid code used in the count.

    Attributes
    ----------
    grid_codes : list[str]
        Grid codes of the tiles used to count the scenes.
    bbox : BoundingBox
        Bounding box of the AOI.
    scene_counts : SceneCounts
        Counts of the available scenes for the AOI.
    """

    grid_code: list[str]
    bbox: BoundingBox
    scene_counts: SceneCounts


@dataclass
class PreliminaryAnalysisResult:
    """Defines the information of the preliminary AOI analysis.

    Attributes
    ----------
    valid_candidates : list[str]
        Contains the name of valid AOI. The AOI name is a string identifier defined in the
        `.toml` configuration file.
    candidates_results : dict[str, CandidateResult]
        Contains result for each candidate AOI.
    """

    valid_candidates: list[str] = field(default_factory=list)
    candidates_results: dict[str, CandidateResult] = field(default_factory=dict)
