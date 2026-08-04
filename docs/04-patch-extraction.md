# Patch extraction

This document describes the process of of creating the labeled dataset from the
composite seasonal image and the WorldCover labels raster.

We can quickly look at both the data by using GDAL from the command line. After
removing some of the returned info to not bloat the document, for the labels we
have:

```sh
$ gdalinfo ./data/labels/worldcover_classes.tif

Driver: GTiff/GeoTIFF
Files: ./data/labels/worldcover_classes.tif
Size is 5120, 5120
Coordinate System is:
PROJCRS["WGS 84 / UTM zone 30N",
    ...
    ID["EPSG",32630]]
Data axis to CRS axis mapping: 1,2
Origin = (200780.000000000000000,4151040.000000000000000)
Pixel Size = (10.000000000000000,-10.000000000000000)
Metadata:
  algorithm_version=V2.0.0
  copyright=ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium
  creation_time=2022-10-21 07:35:40.236136
  legend=10  Tree cover 20  Shrubland 30  Grassland 40  Cropland 50  Built-up 60  Bare/sparse vegetation 70  Snow and ice 80  Permanent water bodies 90  Herbaceous wetland 95  Mangroves 100 Moss and lichen
  license=CC-BY 4.0 - https://creativecommons.org/licenses/by/4.0/
  product_crs=EPSG:4326
  product_grid=3x3 degree tiling grid
  product_tile=N36W009
  product_type=LandCover Map
  product_version=V2.0.0
  reference=https://esa-worldcover.org
  time_end=2021-12-31T23:59:59Z
  time_start=2021-01-01T00:00:00Z
  title=ESA WorldCover product at 10m resolution for year 2021
  AREA_OR_POINT=Area
Image Structure Metadata:
  INTERLEAVE=BAND
Corner Coordinates:
Upper Left  (  200780.000, 4151040.000) (  6d22'58.10"W, 37d27'28.23"N)
Lower Left  (  200780.000, 4099840.000) (  6d21'44.15"W, 36d59'49.64"N)
Upper Right (  251980.000, 4151040.000) (  5d48'16.80"W, 37d28'22.78"N)
Lower Right (  251980.000, 4099840.000) (  5d47'15.45"W, 37d 0'43.29"N)
Center      (  226380.000, 4125440.000) (  6d 5' 3.62"W, 37d14' 7.27"N)
Band 1 Block=5120x1 Type=Byte, ColorInterp=Gray
  NoData Value=0
```

Use `gdalinfo -json ...` if you prefer the JSON formatted output. For the
composite image instead:

```sh
$ gdalinfo ./data/processed/multiseasonal/composite.tif

Driver: GTiff/GeoTIFF
Files: ./data/processed/multiseasonal/composite.tif
Size is 5120, 5120
Coordinate System is:
PROJCRS["WGS 84 / UTM zone 30N",
    ...
    ID["EPSG",32630]]
Data axis to CRS axis mapping: 1,2
Origin = (200780.000000000000000,4151040.000000000000000)
Pixel Size = (10.000000000000000,-10.000000000000000)
Metadata:
  AREA_OR_POINT=Area
Image Structure Metadata:
  COMPRESSION=LZW
  INTERLEAVE=BAND
Corner Coordinates:
Upper Left  (  200780.000, 4151040.000) (  6d22'58.10"W, 37d27'28.23"N)
Lower Left  (  200780.000, 4099840.000) (  6d21'44.15"W, 36d59'49.64"N)
Upper Right (  251980.000, 4151040.000) (  5d48'16.80"W, 37d28'22.78"N)
Lower Right (  251980.000, 4099840.000) (  5d47'15.45"W, 37d 0'43.29"N)
Center      (  226380.000, 4125440.000) (  6d 5' 3.62"W, 37d14' 7.27"N)
Band 1 Block=1024x1024 Type=Float32, ColorInterp=Gray
...
Band 52 Block=1024x1024 Type=Float32, ColorInterp=Undefined
```

And from the returned information we can check that for both the raster the data
is provided with respect to the same `EPSG:32630` CRS, they have the same origin
and resolution.

For the label dataset we could also print some stats with:

```sh
$ gdalinfo -stats -hist ./data/labels/worldcover_classes.tif

  Minimum=1.000, Maximum=9.000, Mean=4.000, StdDev=2.064
0...10...20...30...40...50...60...70...80...90...100 - done.
  256 buckets from -0.5 to 255.5:
  0 3549385 620974 5580828 10780100 1838293 1000586 0 583744 2260490 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  NoData Value=0
  Metadata:
    STATISTICS_MINIMUM=1
    STATISTICS_MAXIMUM=9
    STATISTICS_MEAN=4.000227355957
    STATISTICS_STDDEV=2.0642413436946
    STATISTICS_VALID_PERCENT=100
```

One thing we can notice is that we have the `Block=1024x1024` but we want to
create patches for the training with a different size. How this is impacting the
I/O of the patch creation?

The other important information is that our data is stored with
`INTERLEAVE=BAND`, which means that the data is laid out in memory with all the
values for the first band 1 sequential, then band 2, and so on. Is it the best
approach for our dataset creation or would it be better to have the
`INTERLEAVE=PIXEL`?
