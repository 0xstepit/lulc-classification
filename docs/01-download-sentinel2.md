## What I've Learned

- Not all providers uses the same id to reference to a specific band. The
  Copernicus database access them with \<BAND_NUMBER>\_<RESOLUTION>m while the
  Element.. uses directly the name of the band.
- When up or downsampling, we have to update the affine transformation!
- When creating a new file, we have to bring along the profile if we don't want
  to specify all the info.

When we create a multi-band raster, we can store pixels in two ways:

- **Band sequential mode**: we store all pixels of the first band, then all the
  pixels of the second, and so on until the last band.
  `R1 R2 R3 ... G1 G2 G3 ... B1 B2 B3 ...`
- **Band interleaved by Pixel**: we store for each pixel all its band values
  contiguously. We start writing all the band values for the pixel (0,0) then we
  move to second (0, 1), and so on. `R1 G1 B1 R2 G2 B2 R3 G3 B3`

If we are using tiles, we concept is the same but for each tile. Executing the
sentinel script (01) with the two approaches, result in two completely different
size of the resulting file. In the former, the size is around 514MB, while in
the second is almost 4GB. A possible explanation for it is that, given the
sequential nature of GDAL writing approach, all the time new data is added in
the sequential approach. If we have to touch something that has already been
written, like a tile in the pixel approach when we add a new band, the memory
layout cannot be enough to accommodate the new values, and for this reason, a
new allocation is made. With the band approach, when we add a new band, we
always touch the next memory available without having to keep dead space for the
reallocation.
