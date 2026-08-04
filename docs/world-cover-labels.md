# WorldCover Labels

This document describes the process involved in the generation of the training
labels from the WorldCover V2 dataset.

## The WorldCover Project

The WorldCover project has been initiate by the European Space Agency (ESA) and
freely released. The scope of the project was to create a global land cover map
at $10 ; m$ resolution to enable applications such as biodiversity, food
security, carbon assessment and climate modelling.

We will use the WorldCover V2 dataset released on 28 October 2022 and that uses
an algorithm trained on 319415 locations with 115 features with an overall
accuracy of 76.7%. The important aspect of this dataset is that it has been
*statistically validated* following the Committee on Earth Observation
Satellites (CEOS) validation guidelines

The ESA WorldCover products are provided as $3 \\times 3$ degree tiles for a
total of 2651 elements. Each tile is composed of 2 Cloud Optimized GeoTIFF (COG)
files:

- A land cover map with 11 classes defined using the Land Cover Classification
  System (LCCS).
- An indicator of the quality of the Sentinel-2 and Sentinel-1 data used into
  classifying each pixel.

The 11 classes are: tree cover, shrubland, grassland, cropland, build-up,
bare/sparse vegetation, snow and ice, permanent water bodies, herbaceous
wetland, mangroves, moss and lichen.

The products are delivered in the classic latitude/longitude grid with cog in
`EPSG:4326` projection with the ellipsoid WGS 1984 and the grid resolution is
approximately 10m at equator. For this reason, the final categorical raster has
to be re-projected onto the EPSG CRS used by the composite.

## Analysis

The generation of the WorldCover dataset is configured with the parameters in
the `[worldcover]` section. Here, we define the URL were the data is retrieved
and additional information associated with the labels.

WorldCover classify each pixel in one of the 11 classes mentioned in the
previous section, and each class is associated with a number that is not the
classic sequential identifier used in classification tasks. More specifically,
the numbers $10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100$ are used. We also
added a class `0` to identify missing data.

The first step in the analysis is the retrieval of the dataset tiles associated
with our AOI. This step is performed with GeoPandas and requires the conversion
of the provided tiles from the `EPSG:4326` to the local `EPSG:32630` of the AOI
to properly evaluate intersections of the geometric objects.

From the tile IDs, we iteratively download the categorical rasters from the S3
buckets and merge them into a single entity with the `rioxarray` library. After
the single raster is created, we have to reproject it to match the resolution
and boundaries of our previously generated composite raster. The next e final
phase is the reclassing, which is done to convert all the WorldCover classes
into sequential integers.

The associated scripts compute some useful statistics to better understand the
output of the process, which is a GeoTIFF image with a single band containing
the categorical variables. The report highlights that the raster has been
created out of two WorldCover tiles, the `N36W006` and the `N36W009` and that it
contains 8 classes out of 11. The resulting raster is fully defined and there
are no missing data.

## Classes Distribution

In a classification problem, one of the most important statistics for the
training phase is the distribution of each of the classes in a dataset. If one
class is dominating the labels in the dataset, the final model will be most
likely biased towards that class. For this reason, most of the models are based
under the assumption that the dataset is balanced.

To assess the quality of the label set, the script computes the **imbalance
ratio** (IR), a measure of the distribution of the data considering only the
most frequent and least frequent classes, i.e the **majority** and **minority
classes**. The IR is defined as:

$$ \\text{IR} =
\\frac{pixels(most_frequent_class)}{pixels(least_frequent_class)} $$

The imbalance value for the WorldCover raster over our AOI is $18.47$,
indicating that for every pixel in the least represented class, there are around
18 pixels in the most represented. There is no general agreement on how to
interpret precisely this value, but the rule of thumb is that an imbalance of
1:10 is considered as a modestly imbalanced dataset, which is our case. The
naive solution to solve this condition is to sample with rejection until we
obtain a balanced dataset, but this is done at the expenses of a reduced
dataset. Other more advanced methods can be employed and we will see how to
proceed based on the results of the training phase.
