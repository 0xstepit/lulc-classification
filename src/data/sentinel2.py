from pystac_client import Client

from src.config import BBOX, Sentinel2Config


class SentinelClient:
    def __init__(self, cfg: Sentinel2Config) -> None:
        self._client = Client.open(cfg.stac.url)
        self._cfg = cfg

    def search_scenes(self, bbox: BBOX) -> list:
        year = self._cfg.aoi.year

        search = self._client.search(
            collections=[self._cfg.stac.collection],
            bbox=list(bbox),
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": self._cfg.aoi.max_cloud_coverage}},
        )

        return list(search.items())
