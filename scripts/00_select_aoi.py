"""Run a preliminary investigation of the candidate Area Of Interests (AOI)s.

The result of this script is a `.json` file containing relevant information to select which AOI
use in the project analysis.
"""

import logging
from pathlib import Path

from lulc.config import load_config, load_reporter_config
from lulc.data.sentinel2 import (
    MGRS_PREFIX,
    StacClient,
    count_scenes_by_seasons,
    evaluate_candidate_validity,
    get_tile_id,
)
from lulc.geometry import GCS, create_bbox
from lulc.io import GLOBAL_CONFIG, REPORTER_CONFIG, REPORTS_DIR
from lulc.logger import setup_logging
from lulc.reporter.models import CandidateResult, PreliminaryAnalysisResult
from lulc.reporter.reporter import Reporter

logger = logging.getLogger(__name__ if __name__ != "__main__" else Path(__file__).stem)

REPORT_NAME = "preliminary_aoi_results.json"


def main():  # noqa: D103
    setup_logging()

    # Load config and create the Sentinel client for CDSE.
    cfg = load_config(GLOBAL_CONFIG)
    client = StacClient(cfg)

    # Load and instantiate analysis reporter.
    reporter = Reporter(REPORTS_DIR, load_reporter_config(REPORTER_CONFIG))

    preliminary_analysis_result = PreliminaryAnalysisResult()

    # Iterate through all the area of interests in the configuration.
    for name, point in cfg.aoi.candidates.items():
        logger.info(f"starting STAC requests for candidate [{name}]")

        bbox = create_bbox(GCS(point[1], point[0]), cfg.aoi.size)

        scenes = client.search_items(bbox, single_tile=cfg.aoi.single_tile)
        logger.info("completed STAC request and collected scenes")

        logger.info("starting scenes analysis")
        scene_counts = count_scenes_by_seasons(scenes)

        mgrs = {
            MGRS_PREFIX + tid for s in scenes if (tid := get_tile_id(s)) is not None
        }

        candidate_result = CandidateResult(list(mgrs), bbox, scene_counts)
        preliminary_analysis_result.candidates_results[name] = candidate_result

        if evaluate_candidate_validity(cfg, candidate_result.scene_counts):
            preliminary_analysis_result.valid_candidates.append(name)

        logger.info(f"completed scenes analysis for [{name}]")

    reporter.add("candidates_analysis", preliminary_analysis_result.candidates_results)
    reporter.add("valid_candidates", preliminary_analysis_result.valid_candidates)
    reporter.save(REPORT_NAME)

    logger.info(f"report saved in [{REPORT_NAME}]")

    logger.info("completed preliminary analysis of area of interests")


if __name__ == "__main__":
    main()
