from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.data.sentinel2 import SceneCounts
from src.geometry import BoundingBox


@dataclass
class ReportEntry:
    step: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class CandidateResult:
    """Defines the analysis result for a single AOI candidate. Since the bounding box can be
    enclosed by different tiles, it is important to specify the grid code used in the count.

    Attributes
    ----------
    scene_counts : SceneCounts
        Counts of the available scenes for the AOI.
    bbox : BoundingBox
        Bounding box of the AOI.
    grid_code : str
        Grid code of the tile used to count the scenes.
    """

    scene_counts: SceneCounts
    bbox: BoundingBox
    grid_code: str


@dataclass
class PreliminaryAnalysisResult:
    """Defines the information of the preliminary AOI analysis.

    Attributes
    ----------
    candidates_results : dict[str, CandidateResult]
        Contains result for each candidate AOI.
    valid_candidates : list[str]
        Contains the name of valid AOI.
    """

    candidates_results: dict[str, CandidateResult] = field(default_factory=dict)
    valid_candidates: list[str] = field(default_factory=list)
