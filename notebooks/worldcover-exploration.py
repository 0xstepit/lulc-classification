# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: lulc-classification
#     language: python
#     name: lulc-classification
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import geopandas as gpd

# %%
# load natural earth low res shapefile
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
# Reads directly from URL (downloads and extracts automatically)
ne = gpd.read_file(url)

# %%
ne.head()

# %%
ne.columns.tolist()[:8]

# %%
sum(ne["SOVEREIGNT"] == "Italy")

# %%
s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# get AOI geometry (Italy in this case)
country = "Italy"
geom = ne[ne["SOVEREIGNT"] == country].iloc[0].geometry

# %%
type(geom)

# %%
# load worldcover grid
url = f"{s3_url_prefix}/esa_worldcover_grid.geojson"
grid = gpd.read_file(url)

# get grid tiles intersecting AOI
tiles = grid[grid.intersects(geom)]

# use requests library to download them
import requests
from tqdm.auto import tqdm  # provides a progressbar
from pathlib import Path

year = 2021  # setting this to 2020 will download the v100 product instead

# select version tag, based on the year
version = {2020: "v100", 2021: "v200"}[year]

# output_folder = '.'  # use current directory or set a different one to store downloaded files
# for tile in tqdm(tiles.ll_tile):
#    url = f"{s3_url_prefix}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
#    r = requests.get(url, allow_redirects=True)
#    out_fn = Path(output_folder) / Path(url).name
#    with open(out_fn, 'wb') as f:
#        f.write(r.content)

# %%
type(grid)

# %%
tiles["ll_tile"].values

# %%
grid.crs

# %%
from shapely.geometry import box

bounding_box = [
    -6.362565950090263,
    37.00620706676072,
    -5.798434049909736,
    37.45839293323928,
]

aoi_geom = box(*bounding_box)  # box(minx, miny, maxx, maxy) -> Polygon
print(aoi_geom)

# %%
aoi_geom

# %%
geom = gpd.GeoSeries(aoi_geom, crs="EPSG:4326").to_crs(grid.crs)
aoi_grids = grid[grid.intersects(geom)]

# %%
type(list(aoi_grids["ll_tile"].values))

# %%
geom

# %%
tiles = grid[grid.intersects(aoi_geom)]
tiles

# %%
tiles.geometry


# %%
def get_worldcover_url(tile_id: str) -> str:
    """e.g. N36W006 → WorldCover S3 URL"""
    return (
        f"https://esa-worldcover.s3.amazonaws.com/v200/2021/map/"
        f"ESA_WorldCover_10m_2021_v200_{tile_id}_Map.tif"
    )


# %%
import rioxarray
import matplotlib.pyplot as plt

url = get_worldcover_url("N36W009")

ds = rioxarray.open_rasterio(url)  # streams from S3, no download needed

# %%
aaa = ds.squeeze("band").isel(x=slice(None, None, 40), y=slice(None, None, 40))

# %%
aaa.plot()

# %%
import numpy as np

# %%
np.unique(aaa)

# %%
from urllib.parse import urlparse
import os

# %%
parsed_url

# %%
filename = os.path.basename(parsed_url.path)

# %%
filename

# %%
raw = Path("../data/labels/worldcover_raw")

# %%
list(raw.glob("*.tif"))

# %%
