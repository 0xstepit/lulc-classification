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
logger = logging.getLogger("select_aoi")


RESULTS_FILE = "preliminary_aoi_results.json"
# Specifies if we want only the scenes for the most frequent tile.
IS_SINGLE_TILE = True


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

    for name, point in cfg.aoi.candidates.items():
        logger.info(f"starting STAC requests for candidate {name}")

        bbox = create_bbox(GCS(point[1], point[0]), cfg.aoi.size)

        # Get the scenes of the most frequent tile associated with the bounding box.
        scenes = client.search_items(bbox, single_tile=IS_SINGLE_TILE)
        logger.info(f"completed STAC requests")

        logger.info(f"starting scenes analysis")
        scene_counts = count_scenes_by_seasons(scenes)

        # In the ID we have something like T<CODE>. The STAC filter requires only the <CODE> and
        # the prefix `MGRS-` which stands for Militaty Grid Reference System.
        mgrs = "MGRS-" + get_tile_id(scenes[0])[1:] if IS_SINGLE_TILE else "multiple"

        candidate_result = CandidateResult(scene_counts, bbox, mgrs)
        preliminary_analysis_result.candidates_results[name] = candidate_result
        if evaluate_candidate_validity(cfg, candidate_result.scene_counts):
            preliminary_analysis_result.valid_candidates.append(name)

        logger.info(f"completed scenes analysis")

    # TODO: use the new reporter class.
    save_results(ANALYSIS_DIR / RESULTS_FILE, preliminary_analysis_result)


if __name__ == "__main__":
    main()
