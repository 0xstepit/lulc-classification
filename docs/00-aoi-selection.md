# Area of Interest Selection

## Introduction

The scope of this project is to investigate and develop two data-driven models
for the classification of the Land Usage/Land Cover (LULC) of a selected
European region. The region will be selected by comparing data available under
specific constraints for the quality and availability of 3 different regions. In
particular, we want that the area selected has enough scene collected from the
Sentinel-2 mission for the L2A product in all the 4 climatic seasons and that
the pixels covered by clouds are limited to not impact the quality of the
generated dataset.

The approach used for the project is to divide the end-to-end analysis into
multiple configurable steps. Each step is parametrized by variables defined in a
`config.toml` file, and the code is implemented in a script for reproducibility.
Each phase is accompanied by a document highlighting the approach and the
decisions taken, along with one or more IPython notebook for data exploration
and visualization.

The overall goal will be the definition and training of two models, a random
forest and a U-net, for the classification of pixels into a specific land usage
class. The two models will be trained using scenes collected from the L2A
product of the Sentinel-2 mission that will be collected and processed into a 52
channels composite image. The composite will contains both spectral data and
spectral indexes for the 4 climatic seasons characterizing the area of interest.
We are going to use temporal data because it is what provides the real signature
of the land usage. For example, both a cropland and a grassland look green in
July, but their signatures diverge over time. The ground-truth value for the
class of each pixel will be obtained from the WorldCover dataset which has been
statistically validated.

The goal of the project is not the creation of a perfect model capable of
classifying lands. This goal is hardly possible given that the class label
dataset has been created in 2022, and we will work with images with a 10m meters
resolution. Despite that, what we want to obtain with this project is a well
designed and clean end-to-end Earth observation pipeline that allows to
configure a research with constraints and parameters and to validate the
obtained results. More precisely, we will:

- Analyze data to select scenes satisfying data quality and availability
  constraints.
- Pre-process reflectance data associated with multiple spectral frequencies
  into a specific bounding box. The data will be collected, cleaned, and used to
  create a composite image spanning an entire solar season with the aim to
  extract spatio-temporal patterns.
- Pre-process the ground-truth dataset to create a raster image matching the
  same geographic boundaries and resolution of the composite image.
- Creation of a dataset composed by chips of training features and labels. The
  dataset creation will take into account possible bias sources, like the
  spatial-correlation, to create statistically correct data for models
  validation.
- Train a random forest and a U-Net neural network to perform classification on
  unseen data.
- Validate the results generated from the data-driven models with focus on
  failures mode and quality of the classification.

## Methodology

This document describes the process undergoing the selection of the Area of
Interested (AoI) for the project. Three regions have been included as possible
candidates in the analysis, and the selection of the are used, will be based on
the availability and quality of the Sentinel-2 L2A scenes in the selected time
ranges.

The regions considered for the scene search are identified by specifying a point
described in the `EPSG:4325` (by longitude and latitude), around which a
bounding box is drawn. All the parameters for this analysis are defined in the
`config.toml` file inside the [aoi] section. In particular, the analysis
considered the following parameters and constraints:

- Year 2022 to have a common and high quality dataset for the labels associated
  with each pixel from the [ESA WorldCover V2][zanagaesaworldcover102022].
- The maximum cloud percentage of 10% to avoid having big regions of clouds that
  would results in chunk of raster to remove during the project dataset
  creation.
- 30 minimum number of scenes with at least 5 scenes per season to have enough
  data to create a composite raster with each season properly represented.
- Only the MGRS tile with the highest number of scenes satisfying the
  constraints has been selected to simplify the pre-processing phase.

The considered candidates are:

- Brandeburg (Germany)
- Andalusia (Spain)
- Lombardy (Italy)

And the WorldCover classification for them can be seen in the image below:

<p align="center" width="100%">
    <img width="32%" src="../assets/andalusia.png" alt="Andalusia">
    <img width="32%" src="../assets/lombardy.png" alt="Lombardy">
    <img width="32%" src="../assets/brandeburg.png" alt="Brandeburg">
</p>

The selection of the candidates didn't undergo any particular selection process,
I simply chose 3 regions in Europe that from a visual analysis had enough
classes from the WorldCover viewer.

## Analysis

The analysis for the AOI selection is contained in the
`/scripts/00_select_aoi.py` scripts. The scripts load the configuration from the
TOML file and uses the Python STAC client `pystac_client` to fetch Sentinel-2
scenes metadata from Copernicus database. The results from the search, with
applied filters, is then processed to evaluate the number and quality of
available scenes for each season. One approximation that has been done in this
phase is that the number of available scenes has been evaluated on a solar year,
so the January and February will be associated with winter different from the
one of December. Since this is just an preliminary analysis the approximation is
acceptable and in the subsequent composite generate full seasons will be
considered.

The result of the analysis is written into a file that reports for each
candidate:

- The bounding box generated around the center point.
- The grid codes associated with the considered tiles.
- The scenes count per season and the total number.
- A final section containing the names of the candidates that satisfy all the
  requirements.

Below we can see the data collected in the report for the Andalusia:

```json
"andalusia": {
    "bbox": [
        -6.362565950090263,
        37.00620706676072,
        -5.798434049909736,
        37.45839293323928
    ],
    "grid_code": [
        "MGRS-30STG"
    ],
    "scene_counts": {
        "by_season": {
            "DJF": 9,
            "JJA": 13,
            "MAM": 7,
            "SON": 6
        },
        "total": 35
    }
},
```

## STAC search

The search of the STAC scenes is performed by using the `pystac_client`. During
the development of the code, it happened that I received this error:

```sh
$ make run-script script=00_select_aoi
===================================================================
Running "00_select_aoi" script...
[07/28/26 11:26:18] INFO     starting STAC requests for candidate [brandeburg]                                                                                                                                                                                                                                                              00_select_aoi.py:63
Traceback (most recent call last):
  File "/Users/stepit/Repositories/projects/lulc-classification/scripts/00_select_aoi.py", line 96, in <module>
    main()
    ~~~~^^
  File "/Users/stepit/Repositories/projects/lulc-classification/scripts/00_select_aoi.py", line 67, in main
    scenes = client.search_items(bbox, single_tile=cfg.aoi.single_tile)
  File "/Users/stepit/Repositories/projects/lulc-classification/src/data/sentinel2.py", line 84, in search_items
    items = list(search.items())
  File "/Users/stepit/Repositories/projects/lulc-classification/.venv/lib/python3.14/site-packages/pystac_client/item_search.py", line 785, in items
    for item in self.items_as_dicts():
                ~~~~~~~~~~~~~~~~~~~^^
  File "/Users/stepit/Repositories/projects/lulc-classification/.venv/lib/python3.14/site-packages/pystac_client/item_search.py", line 796, in items_as_dicts
    for page in self.pages_as_dicts():
                ~~~~~~~~~~~~~~~~~~~^^
  File "/Users/stepit/Repositories/projects/lulc-classification/.venv/lib/python3.14/site-packages/pystac_client/item_search.py", line 826, in pages_as_dicts
    for page in self._stac_io.get_pages(
                ~~~~~~~~~~~~~~~~~~~~~~~^
        self.url, self.method, self.get_parameters()
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/stepit/Repositories/projects/lulc-classification/.venv/lib/python3.14/site-packages/pystac_client/stac_api_io.py", line 314, in get_pages
    page = self.read_json(link, parameters=parameters)
  File "/Users/stepit/Repositories/projects/lulc-classification/.venv/lib/python3.14/site-packages/pystac/stac_io.py", line 200, in read_json
    txt = self.read_text(source, *args, **kwargs)
  File "/Users/stepit/Repositories/projects/lulc-classification/.venv/lib/python3.14/site-packages/pystac_client/stac_api_io.py", line 161, in read_text
    return self.request(
           ~~~~~~~~~~~~^
        href, method=method, headers=headers, parameters=parameters
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/stepit/Repositories/projects/lulc-classification/.venv/lib/python3.14/site-packages/pystac_client/stac_api_io.py", line 219, in request
    raise APIError.from_response(resp)
pystac_client.exceptions.APIError: <html>
<head><title>504 Gateway Time-out</title></head>
<body>
<center><h1>504 Gateway Time-out</h1></center>
<hr><center>nginx</center>
</body>
</html>

make: *** [run-script] Error 1
```

The error code **504**, associated with gateway timeout, tells us that the
Server we are interacting with is probably under too heavy load, and this causes
our request to time out. To overcome this issue, the initially plain client
creation has been replaced with a customized
[StacApiIO](https://pystac-client.readthedocs.io/en/latest/api.html#stac-api-io)
configuration to perform paginated request of small items such that we can retry
the single HTTP request if the server is too slow in replying. The customization
of the call has been done by using the standard
[`urllib3.util.Retry`](https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html)
class. The configuration of the STAC client-server is defined in the section
[stac] of the config file, and in particular defines:

- The maximum number of request retry we allow.
- A backoff factor to increase exponentially the time between retry requests.

## Open questions

Below a reported some of the questions that came up during the preliminary
analysis and that could be investigated in a subsequent phase:

- Is 30 scene enough?
- Can we correct noise from clouds by using Setninel-1 or other data sources?
- How accuracy change with the increase in cloud coverage?

## References

1. [ESA WorldCover]
1. [WorldCover V2 - Product User Manual]

[worldcover v2 - product user manual]: https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf
[zanagaesaworldcover102022]: https://zenodo.org/record/7254221 "ESA WorldCover 10 m 2021 V200"
