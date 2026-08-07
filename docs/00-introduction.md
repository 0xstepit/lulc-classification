# Introduction

The scope of this project is the design and implementation of a data retrieval
and processing pipeline for the classification of the Land Usage/Land Cover
(LULC) of a selected European region.

The pipeline is divided into 3 milestones, each associated with a full
definition and implementation of an automated workflow for processing of the
data involved:

1. Investigation and decision of the target area of interest.
1. Generation of a statistically sound dataset.
1. Implementation of data-driven solution for the automatic LULC classification.

The region will be selected by comparing data available under specific
constraints for the quality and availability of a fixed number of candidate
regions. In particular, we want that the area selected has enough scene
collected from the Sentinel-2 mission for the L2A product in all the 4 climatic
seasons, and that the pixels covered by clouds are limited to not impact the
quality of the generated dataset.

The approach used for the project is to divide the end-to-end analysis into
multiple configurable steps. Each step is then parametrized by variables defined
in a `.toml` file, and the code is wrapped into a script for reproducibility
purposes. The execution of the scripts will be accompanied by reports that will
be stored in the associated folders specified in the `/src/io.py` file. Each
report will contain a summary of the data processed, like statistics, number of
valid and discarded scenes, counting of classes per patches, and other relevant
information depending on the executed step.

The overall goal will be the definition and training of two data-driven models,
a random forest and a U-net, for the classification of pixels into a specific
land usage class. The two models will be trained using scenes obtained from the
L2A product of the Sentinel-2 mission that will be collected and processed into
a 52 channels composite image. The composite will contains both spectral data
and spectral indexes for the 4 climatic seasons characterizing the area of
interest. We are going to use temporal data because it is what provides the real
signature of the land usage. Without it, it would be difficult to properly
categorize usages with similar fingerprint in a specific season. For example,
both a cropland and a grassland look green in July, but their signatures diverge
over time. The ground-truth value for the class of each pixel will be obtained
from the WorldCover dataset which has been statistically validated and can than
be considered a valid source of truth.

The goal of the project is not the creation of a perfect model capable of
classifying LULC. This goal is hardly possible given that the class label
dataset has been created in 2022, and we will work with images with a 10m meters
resolution. Despite that,

As a quick summary, what we want to achieve with this project is a well designed
and clean end-to-end Earth observation pipeline that allows to configure a
research with defined constraints and validate the results. In particular, we
will:

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
  spatial-correlation, to create statistically sound data for models validation.
- Train a random forest and a U-Net neural network to perform classification on
  unseen data.
- Validate the results generated from the data-driven models with focus on
  failures mode and quality of the classification.
