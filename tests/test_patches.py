import numpy as np
import pytest

from src.data.patches import (
    LABELS_TO_SET,
    assign_blocks,
    buffer_radius,
    compute_patch_class_counts,
    create_buffer_mask,
    create_labelled_patches,
    validate_block_size,
)

EVEN_SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}


class TestValidateBlockSize:
    @pytest.mark.parametrize("block_size", [1, 2, 256, 1024, 10240])
    def test_accepts_block_size_factor_of_raster(self, block_size):
        assert validate_block_size(10240, block_size) is None

    @pytest.mark.parametrize("block_size", [3, 300, 1023, 20480])
    def test_raises_when_block_size_is_not_a_factor(self, block_size):
        with pytest.raises(ValueError, match="factor of the AOI raster size"):
            validate_block_size(10240, block_size)


class TestBufferRadius:
    @pytest.mark.parametrize(
        ("buffer_m", "patch_size", "pixel_size", "expected"),
        [
            (0, 256, 10.0, 0),  # No buffer requested.
            (100, 256, 10.0, 1),  # Below one patch, still rounded up to one.
            (1000, 256, 10.0, 1),  # 100 pixels, the project default.
            (2560, 256, 10.0, 1),  # Exactly one patch.
            (2570, 256, 10.0, 2),  # Just above one patch.
            (1000, 256, 20.0, 1),  # Coarser pixels shrink the radius.
            (10000, 256, 10.0, 4),  # 1000 pixels.
        ],
    )
    def test_converts_metres_to_patch_units(
        self, buffer_m, patch_size, pixel_size, expected
    ):
        assert buffer_radius(buffer_m, patch_size, pixel_size) == expected


class TestCreateBufferMask:
    def test_returns_boolean_mask_with_input_shape(self):
        labels = np.zeros((4, 6), dtype=np.uint8)

        keep = create_buffer_mask(labels, 1)

        assert keep.shape == labels.shape
        assert keep.dtype == np.bool_

    @pytest.mark.parametrize("radius", [-1, 0])
    def test_keeps_everything_when_radius_is_not_positive(self, radius):
        labels = np.array([[0, 1], [2, 0]], dtype=np.uint8)

        keep = create_buffer_mask(labels, radius)

        assert keep.all()

    @pytest.mark.parametrize("radius", [1, 2, 5])
    def test_keeps_everything_when_all_blocks_share_one_set(self, radius):
        labels = np.full((5, 5), LABELS_TO_SET["train"], dtype=np.uint8)

        keep = create_buffer_mask(labels, radius)

        assert keep.all()

    def test_drops_patches_next_to_a_vertical_boundary(self):
        # Left half train, right half test.
        labels = np.array([[0, 0, 2, 2]] * 4, dtype=np.uint8)

        keep = create_buffer_mask(labels, 1)

        expected = np.array([[True, False, False, True]] * 4)
        np.testing.assert_array_equal(keep, expected)

    def test_drops_patches_next_to_a_horizontal_boundary(self):
        # Top half train, bottom half test — the transpose of the case above.
        labels = np.array([[0, 0, 0, 0]] * 2 + [[2, 2, 2, 2]] * 2, dtype=np.uint8)

        keep = create_buffer_mask(labels, 1)

        expected = np.array([[True] * 4, [False] * 4, [False] * 4, [True] * 4])
        np.testing.assert_array_equal(keep, expected)

    def test_wider_radius_drops_more_patches(self):
        labels = np.array([[0, 0, 0, 2, 2, 2]] * 6, dtype=np.uint8)

        keep_one = create_buffer_mask(labels, 1)
        keep_two = create_buffer_mask(labels, 2)

        np.testing.assert_array_equal(
            keep_one, np.array([[True, True, False, False, True, True]] * 6)
        )
        np.testing.assert_array_equal(
            keep_two, np.array([[True, False, False, False, False, True]] * 6)
        )

    def test_considers_diagonal_neighbours(self):
        # Only the bottom-right corner has a fully homogeneous neighbourhood, and
        # it survives only because the diagonal neighbour (1, 1) matches it.
        labels = np.array([[0, 0, 2], [0, 2, 2], [0, 2, 2]], dtype=np.uint8)

        keep = create_buffer_mask(labels, 1)

        expected = np.zeros((3, 3), dtype=bool)
        expected[2, 2] = True
        np.testing.assert_array_equal(keep, expected)

    def test_borders_are_padded_and_not_dropped(self):
        # Without edge padding the outer ring would be compared against
        # out-of-grid values and dropped.
        labels = np.zeros((3, 3), dtype=np.uint8)

        keep = create_buffer_mask(labels, 1)

        assert keep.all()


class TestAssignBlocks:
    def test_returns_grid_shaped_uint8_labels(self):
        labels = assign_blocks((10, 10), EVEN_SPLIT, seed=3)

        assert labels.shape == (10, 10)
        assert labels.dtype == np.uint8
        assert set(np.unique(labels)) <= set(LABELS_TO_SET.values())

    def test_block_counts_follow_the_split_fractions(self):
        labels = assign_blocks((10, 10), EVEN_SPLIT, seed=3)

        counts = np.bincount(labels.ravel(), minlength=3)

        assert counts[LABELS_TO_SET["test"]] == 15
        assert counts[LABELS_TO_SET["val"]] == 15
        assert counts[LABELS_TO_SET["train"]] == 70

    def test_fractions_are_rounded_up(self):
        # 7 blocks * 0.15 = 1.05 -> 2 blocks each for val and test.
        labels = assign_blocks((1, 7), EVEN_SPLIT, seed=3)

        counts = np.bincount(labels.ravel(), minlength=3)

        assert counts[LABELS_TO_SET["test"]] == 2
        assert counts[LABELS_TO_SET["val"]] == 2
        assert counts[LABELS_TO_SET["train"]] == 3

    def test_assigns_everything_to_train_when_other_fractions_are_zero(self):
        labels = assign_blocks((4, 4), {"train": 1.0, "val": 0.0, "test": 0.0}, seed=3)

        assert (labels == LABELS_TO_SET["train"]).all()

    def test_is_deterministic_for_a_given_seed(self):
        first = assign_blocks((8, 8), EVEN_SPLIT, seed=3)
        second = assign_blocks((8, 8), EVEN_SPLIT, seed=3)

        np.testing.assert_array_equal(first, second)

    def test_different_seeds_shuffle_differently_but_keep_the_counts(self):
        first = assign_blocks((10, 10), EVEN_SPLIT, seed=3)
        second = assign_blocks((10, 10), EVEN_SPLIT, seed=4)

        assert not np.array_equal(first, second)
        np.testing.assert_array_equal(
            np.bincount(first.ravel(), minlength=3),
            np.bincount(second.ravel(), minlength=3),
        )

    def test_handles_non_square_grids(self):
        labels = assign_blocks((4, 25), EVEN_SPLIT, seed=3)

        assert labels.shape == (4, 25)
        assert np.bincount(labels.ravel(), minlength=3).sum() == 100


class TestCreateLabelledPatches:
    def test_is_the_identity_for_one_patch_per_block(self):
        labels = np.array([[0, 2], [1, 0]], dtype=np.uint8)

        patches = create_labelled_patches(labels, 1)

        np.testing.assert_array_equal(patches, labels)

    def test_expands_each_block_into_a_square_tile(self):
        labels = np.array([[0, 2], [1, 0]], dtype=np.uint8)

        patches = create_labelled_patches(labels, 2)

        expected = np.array(
            [
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(patches, expected)

    def test_scales_the_shape_by_patches_per_block(self):
        labels = np.zeros((3, 5), dtype=np.uint8)

        patches = create_labelled_patches(labels, 4)

        assert patches.shape == (12, 20)

    def test_preserves_the_label_dtype(self):
        labels = np.zeros((2, 2), dtype=np.uint8)

        patches = create_labelled_patches(labels, 4)

        assert patches.dtype == np.uint8


class TestComputePatchClassCounts:
    def test_returns_counts_per_patch_and_class(self):
        labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=np.uint8,
        )

        counts = compute_patch_class_counts(labels, patch_size=2, num_classes=2)

        expected = np.array(
            [
                [[4, 0], [0, 4]],
                [[0, 4], [4, 0]],
            ],
            dtype=np.int64,
        )
        assert counts.shape == (2, 2, 2)
        np.testing.assert_array_equal(counts, expected)

    def test_counts_mixed_patches(self):
        labels = np.array(
            [
                [0, 1, 2, 2],
                [1, 1, 2, 0],
            ],
            dtype=np.uint8,
        )

        counts = compute_patch_class_counts(labels, patch_size=2, num_classes=3)

        np.testing.assert_array_equal(counts[0, 0], [1, 3, 0])
        np.testing.assert_array_equal(counts[0, 1], [1, 0, 3])

    def test_each_patch_sums_to_the_patch_area(self):
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 5, size=(24, 16), dtype=np.uint8)

        counts = compute_patch_class_counts(labels, patch_size=8, num_classes=5)

        assert counts.shape == (3, 2, 5)
        assert (counts.sum(axis=-1) == 64).all()

    def test_reserves_a_column_for_classes_absent_from_the_labels(self):
        labels = np.zeros((4, 4), dtype=np.uint8)

        counts = compute_patch_class_counts(labels, patch_size=4, num_classes=6)

        assert counts.shape == (1, 1, 6)
        np.testing.assert_array_equal(counts[0, 0], [16, 0, 0, 0, 0, 0])

    def test_counts_are_int64(self):
        labels = np.zeros((2, 2), dtype=np.uint8)

        counts = compute_patch_class_counts(labels, patch_size=2, num_classes=2)

        assert counts.dtype == np.int64
