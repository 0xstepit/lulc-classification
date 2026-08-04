from math import ceil

import numpy as np

from src.config.dataset import SET_NAMES

SETS_TO_LABEL = {k: v for k, v in enumerate(SET_NAMES)}
LABELS_TO_SET = {v: k for k, v in SETS_TO_LABEL.items()}


def validate_block_size(raster_size: int, block_size: int):
    if raster_size % block_size != 0:
        raise ValueError("[block_size] should be a factor of the AOI raster size")


def buffer_radius(buffer_m: float, patch_size: int, pixel_size: float) -> int:
    """Returns the buffer radius expressed in terms of patch size."""
    buffer_pixel = buffer_m / pixel_size
    return ceil(buffer_pixel / patch_size)


def create_buffer_mask(patch_labels: np.ndarray, buffer_radius: int) -> np.ndarray:
    keep = np.ones(patch_labels.shape, dtype=np.bool)
    if buffer_radius < 1:
        return keep

    # Number of patches along the two dimensions.
    rows, cols = patch_labels.shape

    # Add a pad around the raster such that border patches can be compared. The comparison is done
    # against their own class so they will be retained.
    padded = np.pad(patch_labels, buffer_radius, mode="edge")

    for row_shift in range(-buffer_radius, buffer_radius + 1):
        for col_shift in range(-buffer_radius, buffer_radius + 1):
            # The patch labels is over itself so alway full of 1.
            if row_shift == 0 and col_shift == 0:
                continue

            # We have to create the index positions on the padded raster.
            row_start = row_shift + buffer_radius
            col_start = col_shift + buffer_radius

            padded_patch = padded[
                row_start : row_start + rows, col_start : col_start + cols
            ]

            # Keep values only where the patch label is equal to the padded patch, and it was a kept
            # pixel also in the old keep mask.
            keep &= patch_labels == padded_patch

    return keep


def assign_blocks(
    grid_shape: tuple[int, int], set_fractions: dict[str, float], seed: int
) -> np.ndarray:
    """

    Parameters
    ----------
    grid_shape : tuple[int, int]
        How many blocks are along the rows and columns of the raster.

    set_fractions : dict[str, float]

    seed : int


    Returns
    -------
    np.ndarray

    """
    rng = np.random.default_rng(seed)

    rows, cols = grid_shape
    num_blocks = rows * cols

    # TODO: sets name is not checked in the config.
    val_blocks = ceil(num_blocks * set_fractions["val"])
    test_blocks = ceil(num_blocks * set_fractions["test"])

    # By default zeros uses flaot64 but we need only 3 classes.
    block_labels = np.zeros(num_blocks, dtype=np.uint8)
    block_labels[:test_blocks] = LABELS_TO_SET["test"]
    block_labels[test_blocks : test_blocks + val_blocks] = LABELS_TO_SET["val"]

    rng.shuffle(block_labels)

    return block_labels.reshape(rows, cols)


def create_labelled_patches(labels: np.ndarray, patches_per_block: int) -> np.ndarray:
    return np.repeat(
        np.repeat(labels, patches_per_block, axis=0), patches_per_block, axis=1
    )


def compute_patch_class_counts(
    labels: np.ndarray, patch_size: int, num_classes: int
) -> np.ndarray:
    # Number of patches along each direction.
    rows = labels.shape[0] // patch_size
    cols = labels.shape[1] // patch_size

    # We need to separate pixels without breaking the patch structure.
    # Then, we change the dimension so that the first two indexes are
    # associated with the (x, y) of a patch.
    # We are just regrouping basically having rows and subrows, cols and subcols, since patch_size is a factor of rows, and reshape, never
    # change the memory layout.
    #
    # I initially tried an approach with a transpose, but it was worse in terms of memory usage.
    # The swapaxes is virtual, and does not allocate new memory. Transpose instead, directly
    # alloccate new memory to have the values contiguous in memory. The realloccation is done
    # only inside the loop for the swapaxes, but with a smaller size.
    tiles = labels.reshape(rows, patch_size, cols, patch_size)  # .transpose(1, 3, 0, 2)
    tiles = tiles.swapaxes(1, 2)

    counts = np.zeros((rows, cols, num_classes), dtype=np.int64)

    for i in range(rows):
        for j in range(cols):
            counts[i, j] = np.bincount(tiles[i, j].ravel(), minlength=num_classes)

    return counts.reshape((rows, cols, num_classes))
