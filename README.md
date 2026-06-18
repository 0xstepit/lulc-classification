# Land Usage/Land Coverage Classification with U-Net

[![Unit Tests](https://img.shields.io/github/actions/workflow/status/0xstepit/lu-lc-classification-with-unet/tests.yml?style=flat&logo=githubactions&logoColor=white&label=tests&labelColor=black)](https://github.com/0xstepit/lu-lc-classification-with-unet/actions/workflows/tests.yml)

## Description

This project explores Machine Learning (ML) assisted Land Use and Land Cover
(LULC) classification for a selected Area Of Interest (AOI). The classification
process evaluates and compares the results obtained from a classic ML approach
based on Random Forest and a more advanced one based on deep learning via the
U-Net architecture.

The goal is to assess if it is possible to train models to learn the
spectrotemporal patterns of land through optical data from satellite imagery.

## Land Use and Land Cover

Land Use and Land Cover (LULC) are two different types of analysis that can be
performed on a piece of land. Land Cover concerns the physical classification of
the land surface, while Land Use determines how people are using that particular
land.

When land is analyzed for land cover, pixels are classified into generic
physical classes such as water, forest, built-up, and grassland. Land cover maps
can be used, for example, to assess the effect of climate change, for disaster
or wildfire management, and for local or regional planning.

Land cover classification can use more specific subclasses, such as cropland,
grassland, and wetland. Changes in land use are mainly driven by humans through
processes such as deforestation or urbanization, and are one of the main sources
of carbon dioxide emissions.

## Process

The project is divided in three steps:

- AOI selection and generation of a seasonal image based on optical data
  obtained from Sentinel-2 satellites.
- Creation of the dataset based on the seasonal image created on the previous
  step and the WorldCover dataset provided by the European Space Agency (ESA).
- Training and evaluation of a random forest and a U-Net models.

Each step is then divided into finer tasks that are described in detail in the
[docs](./docs/).

After selecting the area of interest, the data is preprocessed to create all
spectral bands and seasonal composites.

The ESA WorldCover 2021 dataset is used to create the training labels for the
selected area.

The AOI is divided into tiles of a selected size and models are evaluated on
held-out tiles. The model performs classification based on the classes present
in the WorldCover dataset.

## Structure

The project is structured in the following main folders:

- `./src/`: contains the source code for the framework including configuration
  loader, data models, and core functions.
- `./scripts/`: contains Python scripts that are the entrypoints for the
  workflow execution. The order of the scripts is defined by the number
  prefixing the script name.
- `./config/`: contains configuration files in [TOML](https://toml.io/en/)
  format that define the main variables of the analysis, such as the AOI, the
  optical bands used, and other parameters.
- `./tests/`: contains unit tests for the core functions.
- `./data/`: stores raw and processed data (not tracked by git).

Other relevant folders are `./notebooks/`, containing exploration files for
visual inspection, and `./docs/`, containing detailed documentation for each
phase.

## Usage

The project dependencies are managed with the [uv](https://docs.astral.sh/uv/)
package manager. Please, refer to the uv documentation to understand how to
install it. After you have uv available on your machine:

```sh
uv sync
```

A Makefile is provided to simplify the usage of the source code.

### Tests

Unit tests can be executed with:

```sh
make unit-tests
```

### Env

```sh
mv .env.examples .env
```

Add your Copernicus Data Space Ecosystem (CDSE) credentials.

### Scripts

Scripts are the main entrypoints for running the pipeline. They must be executed
in order, as each step depends on the outputs of the previous one. Before
running, review and adjust `config/config.toml` to set the AOI and any other
parameters.

| Script                         | Description                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------- |
| `00_select_aoi`                | Query Sentinel-2 scene counts for candidate AOIs and creates a report           |
| `01_download_sentinel2`        | Download all Sentinel-2 L2A scenes for the selected AOI                         |
| `02_create_seasonal_composite` | Apply cloud masking, compute spectral indices, and build the seasonal composite |
| `03_download_world_cover`      | Download ESA WorldCover 2021 tiles, mosaic, and remap class labels              |

Run a script with:

```sh
make run-script script=00_select_aoi
```

## Notebooks

The project is accompanied with small Jupyter Notebooks to help in the
visualization of the geospatial operations performed in the scripts.

To use them, please first install a new IPython kernel:

```sh
make create-notebook-kernel
```

Start Jupyter to access the notebooks:

```sh
make start-notebook
```

Then select the just created kernel when opening the notebook.

## Scope

The goal of this project is to develop an end-to-end LULC classification
pipeline for learning purpose by using Sentinel-2 satellite imagery, random
forests, and U-net machine learning models.

The development of the framework has been focused around:

- Understanding how to reproduce an end-to-end geospatial ML pipeline from
  scratch.
- Understanding how to handle geospatial blocks for a real-world scenario and
  properly consider memory requirements for raster.
- Handling real-world messy geospatial data with masking, coordinates
  alignments, and seasonal analysis.

It is not part of the scope to create a general purpose LULC classifier but only
a model capable of predicting pixel classes for held-out tiles in the AOI.

## Future Works

- Create the AOI with size multiple of the tiled reading blocks.
- Improve management of the band references.
- Parallelize raster tile reading and writing with positional queue.

## References

1. [Land cover - Wikipedia](https://en.wikipedia.org/wiki/Land_cover)
1. [Land use - Wikipedia](https://en.wikipedia.org/wiki/Land_use)
1. [U-Net: Convolutional Networks for Biomedical Image Segmentation - Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597)
1. [ESA WorldCover 10 m 2021 v200 - Zanaga et al. 2022](https://zenodo.org/records/7254221)
1. [Review on Convolutional Neural Networks (CNN) in vegetation remote sensing - Kattenborn et al. 2022](https://www.sciencedirect.com/science/article/abs/pii/S0924271620303488)
