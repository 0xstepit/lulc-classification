"""
This script is used to run a preliminary investigation of the candidate Area of Interests (AoIs).
"""

import json
import logging
from math import cos, radians
from ntpath import exists
from pathlib import Path

from pystac_client import Client

from src.data.sentinel2 import SentinelClient
from src.io import ANALYSIS_DIR
from src.logger import setup_logging

setup_logging()
logger = logging.getLogger("select_aoi")

from src.config import BBox, Sentinel2Config, load_sentinel2_config

PRELIMINARY_AOI_FILENAME = "preliminary_aoi_results.json"
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


def create_bbox(center_lon: float, center_lat: float, size_km: float) -> BBox:
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


def get_scene_counts(client: SentinelClient, bbox: BBox) -> dict:
    items = client.search_scenes(bbox)

    # It seems that mathced does not work here ???
    # total = search.matched()
    total = 0
    by_season = {"DJF": 0, "MAM": 0, "JJA": 0, "SON": 0}

    for item in items:
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
    with open(ANALYSIS_DIR / PRELIMINARY_AOI_FILENAME, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)


def main():
    # Create folder to store preliminary analysis if it does not exist yet.
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_sentinel2_config()

    client = SentinelClient(cfg)

    valid_candidates = []
    candidate_analysis = {}
    for name, point in cfg.aoi.candidates.items():
        logger.info(f"starting STAC requests for {name}")

        bbox = create_bbox(point[0], point[1], cfg.aoi.size)
        result = get_scene_counts(client, bbox)
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
