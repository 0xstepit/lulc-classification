# Composite Construction

In the third phase of the LULC classification problem, we are going to

We will consider:

- 4 seasons: DJF, MAM, JJA, SON.
- 10 spectral bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12.
- 3 indices: NDVI, NDBI, NDWI

Since both spectral bands and indices are defined per pixel, we will create a
composite image with $4 \\times (3 + 10) = 52$ channels.

What we are going to do in this phase is the following:

1. Mask cloudy/shadowed pixels using the SCL band.
1. Compute NDVI, NDBI, NDWI per scene.
1. Build the seasonal median composite (52 channels).
1. Download and remap WorldCover 2021 labels.
1. Ties it all together with a script.

## Multispectral Bands

| Band ID | Description | Frequency | | ------- | --------------------- |
--------- | | B02 | Blue | | | B03 | Green | | | B04 | Red | | | B05 | Red Edge
1 | | | B06 | Red Edge 2 | | | B07 | Red Edge 3 | | | B08 | Near Infrared | | |
B8A | Red Edge 4 | | | B11 | Short Wave Infrared 1 | | | B12 | Short Wave
Infrared 2 | |

## Vegetation Indices

The Normalized Difference Vegetation Index is defined as follow:

$$ NDVI = \\frac{NIR - RED}{NIR + RED} $$

This index is used to display biomass in an image by leveraging the chlorophyll
pigment absorption in the red band and the high reflectivity of plant materials
in the NIR band.

The Normalized Difference Built-up Index is used to highlight manufactured
built-up areas and is defined as:

$$ NDBI = \\frac{SWIR - NIR}{SWIR + NIR} $$

The **Normalized Difference Water Index (NDWI)** is used to highlight open water
features by measuring moisture content. It is used to both sharpen and monitor
changes in water.

$$ NDWI = \\frac{GREEN - NIR}{GREEN + NIR} $$

This index is used because beyond the visible towards the infrared, water
reflect almost no light.

It is important to notice that this index is particularly sensitive to
built-structures and could overestimate water bodies.

| Range      | Classification                         |
| ---------- | -------------------------------------- |
| 0.2 - 1.0  | Water surface                          |
| 0.0 - 0.2  | Flooding humidity                      |
| -0.3 - 0.0 | Moderate drought, non-aqueous surfaces |
| -1 - -0.3  | Drought, non-aqueous surfaces          |

## Scene Classification

The Scene CLassification (SCL) map is used to clean the rasters we created in
phase 2 to account for disturbances in the image. The SCL raster provides
information abound cloudy, clear, or water pixels by using twelve classes:

| Class | Description              |
| ----- | ------------------------ |
| 0     | No Data                  |
| 1     | Defective Pixel          |
| 2     | Topographic Shadow       |
| 3     | Cloud Shadow             |
| 4     | Vegetation               |
| 5     | Non-vegetated            |
| 6     | Water                    |
| 7     | Unclassified             |
| 8     | Cloud Medium Probability |
| 9     | Cloud High Probability   |
| 10    | Thin Cirrus              |
| 11    | Snow or Ice              |

Amongst these classification classes, we are interested only in those that can
be considered noise in the data: No Data, Defective Pixel, Cloud Shadow, Cloud
Medium Probability, Cloud High Probability, Thin Cirrus, and Snow or Ice.

The class we want to filter out from our scenes are defined in the configuration
file to enable possible parametric analysis on them.

In applying the SCL mask, we will convert pixels associated with the filter mask
to NaN, requiring the conversion of the whole raster from `uint16` to `float32`.

## Memory Usage

To create a seasonal composite image, we will need to use all available images
for each season. This process needs to be carefully designed since it requires
to load up to 10 images into memory. Loading 10 scenes into memory requires
roughly

$$ \\underbrace{10}_{\\text{max images}} \\times
\\underbrace{13}_{\\text{channels}} \\times \\underbrace{5{,}000}_{\\text{approx
height}} \\times \\underbrace{5{,}000}_{\\text{approx width}} \\times
\\underbrace{4}\_{\\text{bytes per value}} = 1.300.000.000 ; \\text{Bytes}
\\simeq 12 ; GB $$

Given that operations performed to compute the median require to copy this
array, with the naive approach of loading all rasters in memory we can arrive to
bloat the RAM close to 50 GB.

We can improve on the naive solution by processing subrasters of images instead
of all the scenes together. This way, the memory burden can be reduced following
the formula:

$$ \\underbrace{10}_{\\text{max images}} \\times
\\underbrace{13}_{\\text{channels}} \\times \\underbrace{C_h}_{\\text{chunk
height}} \\times \\underbrace{C_w}_{\\text{chunk width}} \\times
\\underbrace{4}\_{\\text{bytes per value}} $$

Which for chunks of $512 \\times 512$ results in roughly $260 MB$ of required
RAM.

Now that I discovered that we need to use tiled read, it was better to store
images already with the correct shape as multiple of the tile block. If we
window-read an image now, we end-up with the last tile in a row that has.

## What I've learned

Hard to work with all raster in memory. Even though compressed each all band
image is just half giga, when uncompressed it is around 2 GB. Then, we add other
3 channels and the data is casted to float32. We easily arrive at around 30GB in
memory..not good. Then we duplicate everything with the stack. no bueno.

Given that we have to know where each band sits in the raster, probably was
better to use xarray instead of a fragile configuration pointing each band to a
position.

How to handle nan values:

- Median of the band, is independent on the patch size.
- Median of the entire image if all the bands are nan. Better doing it at patch
  level and not at block level.

## References

- https://eos.com/make-an-analysis/ndwi/
