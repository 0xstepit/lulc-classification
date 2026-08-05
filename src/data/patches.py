from math import ceil

import numpy as np

from src.config.dataset import SET_NAMES

SETS_TO_LABEL = {k: v for k, v in enumerate(SET_NAMES)}
LABELS_TO_SET = {v: k for k, v in SETS_TO_LABEL.items()}


def validate_block_size(raster_size: int, block_size: int):
    """Check that the provided block size is a factor of both the raster size. This check
    assumes that both the raster and the block are squared.

    Parameters
    ----------
    raster_size : int
        The number of pixels along each reaster dimension.
    block_size : int
        The number of pixels along each block dimension.
    """
    if raster_size % block_size != 0:
        raise ValueError(
            f"block_size ({block_size}) should be a factor of the raster_size ({raster_size})"
        )


def compute_buffer_radius(
    buffer_m: float, patch_size: int, pixel_resolution_m: float
) -> int:
    """Returns the buffer radius to apply to each blocks as a fraction of the patch_size.

    Parameters
    ----------
    buffer_m : float
        The desired buffer expressed in meters.
    patch_size : int
        Number of pixels along each patch dimension.
    pixel_size : float
        Spatial resolution of the each patch pixel expressed in meters.

    Returns
    -------
    int
        The number of patch to use as buffer for the block.

    Examples
    --------
    If we have patch_size=10, pixel_resolution_m=2, and buffer_m=10, the resulting buffer radius
    will be 1, meaning that an entire patch is used as a buffer.
    """
    buffer_pixel = buffer_m / pixel_resolution_m
    return ceil(buffer_pixel / patch_size)


def create_buffer_mask(patch_labels: np.ndarray, buffer_radius: int) -> np.ndarray:
    """Create a boolean buffer mask indicating which patch has to be retained (1) and
    which one has to be removed (0).

    Parameters
    ----------
    patch_labels : np.ndarray
        An array indicating the label of the set each patch belongs to.
    buffer_radius : int
        The size of the buffer as a fraction of the patch size.

    Returns
    -------
    np.ndarray


    """
    # Let's start by assuming we keep all the patches.
    keep = np.ones(patch_labels.shape, dtype=np.bool)
    if buffer_radius < 1:
        return keep

    if len(patch_labels.shape) != 2:
        raise ValueError(
            f"patches dimension must be 2D, obtained a {len(patch_labels.shape)}D dimension"
        )

    # Number of patches along the two dimensions.
    rows, cols = patch_labels.shape

    if buffer_radius > rows or buffer_radius > cols:
        raise ValueError(
            f"buffer_radius ({buffer_radius}) is bigger than patches number in the block ({rows}, {cols})"
        )

    # Add a pad around the raster such that border patches can be
    # compared. The comparison is done against their own
    # class so they will be retained.
    padded = np.pad(patch_labels, buffer_radius, mode="edge")

    # Slide the original patch_labels as a kernel on its padded version.
    for row_shift in range(-buffer_radius, buffer_radius + 1):
        for col_shift in range(-buffer_radius, buffer_radius + 1):
            # The patch labels is over itself so alway full of 1.
            if row_shift == 0 and col_shift == 0:
                continue

            # Index position of the padded patch_labels. Since row_shift and col_shift can be
            # negative, we have to add the buffer_radius to start from the index 0.
            row_start = row_shift + buffer_radius
            col_start = col_shift + buffer_radius

            padded_patch = padded[
                row_start : row_start + rows, col_start : col_start + cols
            ]

            # Keep values only where the patch label is equal to the
            # padded patch, and it was a kept pixel also in the old keep mask.
            keep &= patch_labels == padded_patch

    return keep


def assign_blocks(
    seed: int, grid_size: int, set_fractions: dict[str, float]
) -> np.ndarray:
    """Divide the the raster into blocks and assign each block to one set based on the
    associated fraction. Randomness in the set dispatching is controlled by the provided seed.

    Parameters
    ----------
    seed : int
        Seed used in the random number generation.
    grid_size : int
        Number of blocks along each raster direction.
    set_fractions : dict[str, float]
        Mapping between a set name and its dataset fraction.

    Returns
    -------
    np.ndarray

    Examples
    --------
    If we have 3 sets, and the grid is a 3x3 blocks, we will obtain something like:
        | 0, 1, 0 |
        | 0, 2, 2 |
        | 0, 1, 0 |
    """
    rng = np.random.default_rng(seed)

    num_blocks = grid_size**2

    # The rounding due to fraction computation is absorbed by the training set.
    val_blocks = ceil(num_blocks * set_fractions["val"])
    test_blocks = ceil(num_blocks * set_fractions["test"])

    block_labels = np.zeros(num_blocks, dtype=np.uint8)
    block_labels[:test_blocks] = LABELS_TO_SET["test"]
    block_labels[test_blocks : test_blocks + val_blocks] = LABELS_TO_SET["val"]

    rng.shuffle(block_labels)

    return block_labels.reshape(grid_size, grid_size)


def create_labelled_patches(labels: np.ndarray, patches_per_block: int) -> np.ndarray:
    return np.repeat(
        np.repeat(labels, patches_per_block, axis=0), patches_per_block, axis=1
    )


def counts_classes_per_patch(
    labels: np.ndarray, patch_size: int, num_classes: int
) -> np.ndarray:
    """Count the number of pixels associated with the classes in each patch.

    Parameters
    ----------
    labels : np.ndarray
        The rasters with pixel labels of dimension R1xR2.
    patch_size : int
        The number of pixels along each patch dimension.
    num_classes : int
        The total number of classes.

    Returns
    -------
    np.ndarray
        A 3D array with shape R1xR2x(num_classes). The size of the first
        two dimensions equal to the number of patches along x and y
        in the raster, and the third dimension contains the counting
        for each class in the patch.
    """
    # Number of patches along each direction.
    rows = labels.shape[0] // patch_size
    cols = labels.shape[1] // patch_size

    # We need to separate pixels without breaking the patch structure.
    # Then, we change the dimension so that the first two indexes are
    # associated with the directions (x, y) of a patch.
    # We are just regrouping basically having rows and subrows, cols and subcols, since patch_size is a factor of rows, and reshape, never
    # change the memory layout.
    #
    # I initially tried an approach with a transpose, but it was worse in terms of memory usage.
    # The swapaxes is virtual, and does not allocate new memory. Transpose instead, directly
    # alloccate new memory to have the values contiguous in memory. The realloccation is done
    # only inside the loop for the swapaxes, but with a smaller size.
    patch_grid = labels.reshape(
        rows, patch_size, cols, patch_size
    )  # .transpose(1, 3, 0, 2)
    patch_grid = patch_grid.swapaxes(1, 2)

    counts = np.zeros((rows, cols, num_classes), dtype=np.int64)

    for i in range(rows):
        for j in range(cols):
            counts[i, j] = np.bincount(patch_grid[i, j].ravel(), minlength=num_classes)

    return counts.reshape((rows, cols, num_classes))


# def select_seed(
#     grid_shape: np.ndarray, num_candidates: int, set_fractions: dict[str, float]
# ):
#
#     ranked = []
#     for seed in range(num_candidates):
#         raster_blocks = assign_blocks(seed, grid_shape, set_fractions)
