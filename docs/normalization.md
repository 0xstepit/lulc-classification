- Percentile clipping over z-scoring, even though your JSON carries mean/std.
  Reflectance distributions here are right-skewed with bright outliers; clipping
  to p1/p99 bounds the input to [0,1] exactly, which pairs better with the
  BatchNorm in the U-Net. The mean/std stay in the file if you later want to
  compare.

- NaN → normalized median, not 0. Zero is a legitimate reflectance value and
  would read as "very dark pixel"; the median is the least informative signal
  for that band. Your extraction found ~1e-5 NaN, so this fires rarely — but it
  must not fire wrongly.

- Raw on disk, normalized on read. Costs a little CPU per __getitem__, buys you
  a scheme change without re-extracting 3.5 GB.

- Design notes: sorted() fixes the order so a run is reproducible and a given
  index always means the same patch across runs; the label path is derived from
  the feature path rather than globbed separately, so a missing or extra file
  fails loudly at construction instead of silently misaligning features and
  labels; and the transform takes the pair together, because the spatial
  augmentations have to be applied identically to both.
