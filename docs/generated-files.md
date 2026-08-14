# Generated files

This document provides a brief description of the files generated at the various
steps of the pipeline.

## All bands

The first type of file generated is an all-bands raster for each scene used in
the seasonal composites. Since data at different frequencies are provided at
different endpoints in the STAC catalog, we group all of them for each scene to
make their analysis easier.

From the folder defined by the `RAW_DATA_DIR` variable, we can select one of the
scene files and set an env variable to make the commands cleaner:

```sh
FILE=S2A_MSIL2A_20220903T110631_N0510_R137_T30STG_20240729T160319_ALLBANDS.tif
```

and execute:

```sh
gdalinfo -stats $FILE
```

With this command we can get a lot of information out of our raster: the CRS
used, the origin and pixel size, the band statistics, and other metadata.

We can get the info associated with a single band with:

```json
$ gdalinfo -json $FILE | jq '.bands[6]'

{
  "band": 7,
  "block": [1024, 1024],
  "type": "UInt16",
  "colorInterpretation": "Undefined",
  "description": "nir",
  "min": 0.0,
  "max": 16768.0,
  "minimum": 0.0,
  "maximum": 16768.0,
  "mean": 3802.716,
  "stdDev": 754.129,
  "metadata": {
    "": {
      "STATISTICS_MAXIMUM": "16768",
      "STATISTICS_MEAN": "3802.7164438392",
      "STATISTICS_MINIMUM": "0",
      "STATISTICS_STDDEV": "754.1285125272",
      "STATISTICS_VALID_PERCENT": "100"
    }
  }
}
```

Notice that GDAL starts counting from 1 and not from 0 when assigning the band
index, so index 0 in the query is associated with band number 1.

As we can see, in the generation of this all-bands raster, we:

- Updated the name of the band to a generic `nir` instead of keeping the
  CDSE-specific value `B08_10m`.
- Kept the rescaled reflectance values unchanged, to optimise memory usage with
  the `UInt16` type.

We can also see that the data is stored in blocks of $1024 \\times 1024$ pixels,
and that we forgot to add the `colorInterpretation` for this band, which instead
is set to `Gray` for the green band.

We can see the metadata where we stored the information for each specific
analysis with:

```sh
gdalinfo -json $FILE | jq '.metadata'
```

In particular, we added to the metadata of each image some provenance
information that associates each raster with a specific state of the project,
the commit and the worktree. One example is:

```json
 $ gdalinfo -json $FILE | jq '.metadata."" | {git_sha, git_dirty} '

{
  "git_sha": "fb4759c9d95f6194090aa048b2caa68d33200529",
  "git_dirty": "true"
}
```

Notice that we had to use the `.""` because GDAL organizes metadata in domains,
and the empty string is the default domain where all the custom metadata are
added.

## Seasonal composites

In the folder indicated by `SEASONAL_SCENE_DIR` we can see the seasonal
composites created from the all-bands files. We can get the stats:

```sh
FILE=JJA_SEASONAL.tif
gdalinfo -stats $FILE
```

We see that we have 13 bands instead of 11 because we removed the SCL band and
added three spectral indices. In this case the reflectance values are the true
ones, rescaled from the values stored by Sentinel-2.

As a quick check we can see that all the indices are bounded within the correct
ranges, and that probably all of them had saturated values that have been
clipped to $[-1, +1]$, except NDBI, which has a minimum at $-0.834$.

We can also compare the NDVI for all the seasons:

```sh
$ gdalinfo -json DJF_SEASONAL.tif | jq '.bands[10].metadata."".STATISTICS_MEAN'
"0.30221024972717"

$ gdalinfo -json MAM_SEASONAL.tif | jq '.bands[10].metadata."".STATISTICS_MEAN'
"0.32304774749566"

$ gdalinfo -json JJA_SEASONAL.tif | jq '.bands[10].metadata."".STATISTICS_MEAN'
"0.2565132392462"

$ gdalinfo -json SON_SEASONAL.tif | jq '.bands[11].metadata."".STATISTICS_MEAN'
"0.080534817009171"
```

We can notice that, as expected, the maximum of the NDVI is in the spring
season, and that it drops during the hot summer.

We can investigate the reflectance values for the visible frequencies with:

```json
$ gdalinfo -json SON_SEASONAL.tif | jq '.bands[0, 1, 2] | {band, min, max, mean}'

{ "band": 1, "min": 0.0, "max": 0.806, "mean": 0.083 }
{ "band": 2, "min": 0.0, "max": 0.905, "mean": 0.117 }
{ "band": 3, "min": 0.001, "max": 0.962, "mean": 0.153 }
```

Although the three bands have min and max touching the physical boundaries, all
of them have an average reflectance value that is skewed towards the lower end.

## WorldCover raster

The WorldCover raster is stored in the directory defined by the variable
`LABELS_DIR`. We can execute the GDAL commands:

```sh
FILE=JJA_SEASONAL.tif
gdalinfo -stats $FILE
```

We can see that this raster has far more information attached to it than the
previous files. In particular, when we run the commands, we will see a very long
list of numbers like:

```sh
Color Table (RGB with 256 entries)
  ...
  248: 0,0,0,255
  249: 0,0,0,255
  250: 0,0,0,255
  251: 0,0,0,255
  252: 0,0,0,255
  253: 0,0,0,255
  254: 0,0,0,255
  255: 0,0,0,255
```

This long list contains the color table that has to be used to visualize the
raster values. With this information, when we open the file with the open-source
software QGIS, we will directly see the classes with the colors used in the
official WorldCover paper. From the stats we see that our tile only contains
values from class 1 to class 9. Since this raster is not associated with the
standard tile system of the WorldCover project, we added a specific metadata
entry to record which tiles have been used to compose this raster:

```json
$ gdalinfo -json $FILE | jq '.metadata."".worldcover_tiles'

"['N36W009', 'N36W006']"
```

This file also contains the copyright notice from the ESA WorldCover project:
"ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021)
processed by ESA WorldCover consortium".

## Patches

The final products generated from the data preparation pipeline are the dataset
patches. These files are stored in the folder defined by `PATCHES_DIR`. Patches
are nothing but the rasters for the labels and the stacked seasons divided into
small chunks, with each chunk associated with a dataset subset. Since these data
will be used directly by PyTorch or another Python library, they are stored in
`.npy` files, and we cannot use GDAL to extract information from them.
