"""
This script is used to run a preliminary investigation of the candidate Area Of Interests (AOI)s.
The result of this script is a `.json` file containing relevant information to select which AOI
use in the project.
"""

import dataclasses
import json
import logging
from pathlib import Path

from src.config import load_config
from src.data.sentinel2 import (
    MGRS_PREFIX,
    SentinelClient,
    count_scenes_by_seasons,
    evaluate_candidate_validity,
    get_tile_id,
)
from src.geometry import GCS, create_bbox
from src.io import ANALYSIS_DIR, GLOBAL_CONFIG
from src.logger import setup_logging
from src.reporter.models import CandidateResult, PreliminaryAnalysisResult

setup_logging()
logger = logging.getLogger(__name__ if __name__ != "__main__" else Path(__file__).stem)


REPORT_FILEPATH = ANALYSIS_DIR / "preliminary_aoi_results.json"


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

    # Load config and create the Sentinel client for CDSE.
    cfg = load_config(GLOBAL_CONFIG)
    client = SentinelClient(cfg)

    preliminary_analysis_result = PreliminaryAnalysisResult()

    # Iterate through all the area of interests in the configuration.
    for name, point in cfg.aoi.candidates.items():
        logger.info(f"starting STAC requests for candidate [{name}]")

        bbox = create_bbox(GCS(point[1], point[0]), cfg.aoi.size)

        scenes = client.search_items(bbox, single_tile=cfg.aoi.single_tile)
        logger.info(f"completed STAC request and collected scenes")

        logger.info(f"starting scenes analysis")
        scene_counts = count_scenes_by_seasons(scenes)

        mgrs = set(
            [MGRS_PREFIX + tid for s in scenes if (tid := get_tile_id(s)) is not None]
        )

        candidate_result = CandidateResult(list(mgrs), bbox, scene_counts)
        preliminary_analysis_result.candidates_results[name] = candidate_result

        if evaluate_candidate_validity(cfg, candidate_result.scene_counts):
            preliminary_analysis_result.valid_candidates.append(name)

        logger.info(f"completed scenes analysis for [{name}]")

    # TODO: use the new reporter class.
    save_results(ANALYSIS_DIR / RESULTS_FILE, preliminary_analysis_result)
    logger.info(f"report saved at [{REPORT_FILEPATH}]")

    logger.info(f"completed preliminary analysis of area of interests")


if __name__ == "__main__":
    main()
