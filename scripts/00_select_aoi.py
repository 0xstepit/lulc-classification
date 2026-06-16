"""
This script is used to run a preliminary investigation of the candidate Area of Interests (AoIs).
The result of this script is a `.json` file containing relevant information to select which AoI
use in the project.
"""

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Config, load_sentinel2_config
from src.data.sentinel2 import (
    SceneCounts,
    SentinelClient,
    count_scenes_by_seasons,
    get_tile_id,
)
from src.geometry import GCS, BoundingBox, create_bbox
from src.io import ANALYSIS_DIR
from src.logger import setup_logging

setup_logging()
logger = logging.getLogger("select_aoi")


RESULTS_FILENAME = "preliminary_aoi_results.json"


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


def evaluate_candidate_validity(
    cfg: Config, candidate_analysis: CandidateResult
) -> bool:
    """Check if an AOI is a valid area based on the provided result.
    The validity is based on the number of scenes available.

    Parameters
    ----------
    cfg : Sentinel2Config
        Sentinel2 analysis configuration.
    candidate_analysis : CandidateResult
        Result of the candidate analysis used to evaluate its validity.


    Returns
    -------
    bool
        True if the AOI is a valid candidate, False otherwise.

    """
    # Evaluate minimum overall scenes.
    if candidate_analysis.scene_counts.total < cfg.aoi.min_scenes:
        return False
    # Evaluate minimum scenes per season.
    for _, count in candidate_analysis.scene_counts.by_season.items():
        if count < cfg.aoi.min_scenes_per_season:
            return False

    return True


def save_results(path: Path, data: PreliminaryAnalysisResult):
    """Save the data as a JSON formatted string.

    Parameters
    ----------
    path : Path
        The path of the file where the data has to be stored.
    data : PreliminaryAnalysisResult


    """
    logger.info("saving results to file")
    with open(path, "w") as file:
        json.dump(dataclasses.asdict(data), file, indent=4, sort_keys=True)


def main():
    # Create folder to store preliminary analysis if it does not exist yet.
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load config and create Sentinel client for CDSE.
    cfg = load_sentinel2_config()
    client = SentinelClient(cfg)

    preliminary_analysis_result = PreliminaryAnalysisResult()

    for name, point in cfg.aoi.candidates.items():
        logger.info(f"starting STAC requests for {name}")

        bbox = create_bbox(GCS(point[1], point[0]), cfg.aoi.size)

        # Get the scenes of the most frequent tile associated with the bounding box.
        scenes = client.search_items(bbox)

        scene_counts = count_scenes_by_seasons(scenes)

        # In the ID we have something like T<CODE>. The STAC filter requires only the <CODE> and the
        # prefix `MGRS-` which stands for Militaty Grid Reference System.
        candidate_result = CandidateResult(
            scene_counts, bbox, "MGRS-" + get_tile_id(scenes[0])[1:]
        )
        preliminary_analysis_result.candidates_results[name] = candidate_result
        if evaluate_candidate_validity(cfg, candidate_result):
            preliminary_analysis_result.valid_candidates.append(name)

        logger.info(f"completed STAC requests and analysis for region {name}")

    save_results(ANALYSIS_DIR / RESULTS_FILENAME, preliminary_analysis_result)


if __name__ == "__main__":
    main()
