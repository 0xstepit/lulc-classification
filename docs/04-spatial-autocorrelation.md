# Spatial autocorrelation

When designing how to create our dataset out of the two tiles we created, the
$54$ channels composite image and the WorldCover labels raster, we need to keep
into consideration a phenomenon called
[spatial autocorrelation][kattenbornspatiallyautocorrelatedtraining2022]. The
goal of the rasters splitting, is to be able to train a model that is able to to
learn from the training data parameters capable to generalize the prediction to
unseen data. When we talk about unseen data, we actually desire that the model
is capable of making prediction not only for unseen input samples, but also, to
samples that are not correlated with the data used during training. This is the
real goal of creating multiple sets during training. If the sets are correlated,
then our validation of the model performance is not fair.

We are fine if the model uses the spatial correlation inside the training set,
because then, we evaluate its real generalization capabilities on something that
is independent from the data used during training. If we don't do so, then our
model will just learn to assign to geographically close enough pixels the same
class. So, the training, testing, and validation set should be independent not
only in the sense that they have different samples, but also in terms of
geolocation, avoiding this way leakage of information associated with geographic
coordinates. The spatial correlation on the training set has the only
consequence that the real informative content in the dataset is reduced, because
spatially correlated pixels provides less useful content than non correlated
one.

So, we safeguard the information leakage by creating a wall, the buffer, between
the regions used for the different purpose.

```text
   T T T V V        T = train block
   T T T V V        V = val block
   T T R R V        R = test block
   T T R R T        (each cell = 4×4 patches ≈ 10×10 km)
   V T T T T
```

Also in the limit of infinite dataset, buffering each chips should not buy
anything since we will just have sparse data over the same distribution.

But it is interesting to have mutually independent test chips so we have
calibrated confidence intervals.

## Patches

We have two rasters with the same size of $5120 \\times 5120$ pixels. We can
split them in patches of:

- $256$: $5120 / 256 = 20$ obtaining 400 patches.

We also have to divide each rasters into block to account for the spatial
correlation.

- $1024$: $5120 / 1024 = 5$ obtaining 25 blocks.

By having the block size as a multiple of the patch size, we can create the
buffer between the set used for the three different phases (train, test, val)
directly by removing patches on the boundaries of each block.

There is no a specific formula that tells you how much buffer you have to insert
between each set. This specific choice is strongly dependent on the area of
interest considered, and also on the spatial resolution of the data you have at
hand. Remember that this phase is essential for the proper assessment of spatial
models.

Since we are using the Sentinel-2 data at $10 , m$ resolution, each pixel is
associated with a $10 \\times 10$ meter area. If we want to have a buffer of $1
, km$, we have to use $100$ pixels. However, since a single patch is composed of
$256$ pixels, we have to remove it completely because we will not be able to
properly use patches on the border otherwise (without further data
manipulation).

Assuming we use a train/val/test split of $0.7/0.15/0.15$, we can compute the
total number of patches we will have based on the spatial buffer chosen. With
this separation, we will have $16$ blocks for the training and $4$ for both the
val and the test. How can we create the buffers now?

The simplest approach is probably to create a function that given the position
of a pixels, assign it to one of the three sets. Then, once we have the label
for each block, we can get the blocks associated with the train and val, and
create a buffer around them by removing patches from the training set. Given
that we will have $8$ blocks not in the training, and each block has a size of
$1024$ pixels, we will have to remove in the worst case scenario:

$$ 8 \\times 4 \\times 1024 \\ 256 = 128 , \\text{patches} $$

## Normalization

The normalization of the dataset is done by clipping each band to its p1 and p99
computed only over the training blocks and then rescaling everything within
$\[0, 1\]$.

[kattenbornspatiallyautocorrelatedtraining2022]: https://linkinghub.elsevier.com/retrieve/pii/S2667393222000072 "Spatially Autocorrelated Training and Validation Samples Inflate Performance Assessment of Convolutional Neural Networks"
