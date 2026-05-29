"""
This script is used to run a preliminary investigation of the candidate Area of Interests (AoIs).
"""

import json
import logging
from math import cos, pi, radians
from ntpath import exists
from pathlib import Path

from pystac_client import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("select_aoi")

from src.config import BBOX, Sentinel2Config, load_sentinel2_config

ANALYSIS_DIR = Path(__file__).parent.parent / "data/analysis"
RESULTS_FILE_NAME = "preliminary_aoi_results.json"
SEASON_MONTHS = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
}


def deg_per_km(size_km: float, center_lat: float) -> tuple[float, float]:
    """
    Convert with approximation latitude and longitude from degrees into km

    Ref: https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-and-km-distance
    """
    lat_deg = size_km / 110_574
    lon_deg = size_km / (111_320 * cos(radians(center_lat)))

    return lon_deg, lat_deg


def create_bbox(center_lon: float, center_lat: float, size_km: float) -> BBOX:
    """
    Create a bounding box around the provided center of the specified size.
    """
    half_size_km = size_km / 2

    lon_deg, lat_deg = deg_per_km(half_size_km, center_lat)

    return (
        center_lon - lon_deg,
        center_lat - lat_deg,
        center_lon + lon_deg,
        center_lat + lat_deg,
    )


def get_scene_counts(client, cfg: Sentinel2Config, bbox: BBOX) -> dict:
    search = client.search(
        collections=[cfg.stac.collection],
        bbox=list(bbox),
        datetime=f"{cfg.aoi.year}-01-01/{cfg.aoi.year}-12-31",
        query={"eo:cloud_cover": {"lt": cfg.aoi.max_cloud_coverage}},
    )

    # It seems that mathced does not work here ???
    # total = search.matched()
    total = 0
    by_season = {"DJF": 0, "MAM": 0, "JJA": 0, "SON": 0}

    for item in search.items():
        month = item.datetime.month
        for season, months in SEASON_MONTHS.items():
            if month in months:
                by_season[season] += 1
        total += 1

    return {"total": total, "by_season": by_season}


def evaluate_candidate_validity(cfg: Sentinel2Config, analysis: dict) -> bool:
    if analysis["total"] < cfg.aoi.min_scenes:
        return False
    for _, count in analysis["by_season"].items():
        if count < cfg.aoi.min_scenes_per_season:
            return False

    return True


def save_results(data: dict):
    logger.info("saving results to file")
    with open(ANALYSIS_DIR / RESULTS_FILE_NAME, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_sentinel2_config()
    client = Client.open(cfg.stac.url)

    valid_candidates = []
    candidate_analysis = {}
    for name, point in cfg.aoi.candidates.items():
        logger.info(f"starting STAC requests for {name}")

        bbox = create_bbox(point[0], point[1], cfg.aoi.size)
        result = get_scene_counts(client, cfg, bbox)
        result["bbox"] = bbox
        candidate_analysis[name] = result
        if evaluate_candidate_validity(cfg, result):
            valid_candidates.append(name)

        logger.info(f"completed STAC requests for {name}")

    save_results(
        {
            "candidate_analysis": candidate_analysis,
            "valid_candidates": valid_candidates,
        }
    )


if __name__ == "__main__":
    main()
