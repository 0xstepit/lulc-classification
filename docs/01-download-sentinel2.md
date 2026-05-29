## What I've Learned

- Not all providers uses the same id to reference to a specific band. The
  Copernicus database access them with \<BAND_NUMBER>\_<RESOLUTION>m while the
  Element.. uses directly the name of the band.
- When up or downsampling, we have to update the affine transformation!
- When creating a new file, we have to bring along the profile if we don't want
  to specify all the info.
