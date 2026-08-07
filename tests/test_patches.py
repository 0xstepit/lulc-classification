import numpy as np
import pytest

from src.data.patches import (
    LABELS_TO_SET,
    assign_blocks,
    compute_buffer_radius,
    counts_classes_per_patch,
    create_buffer_mask,
    create_labelled_patches,
    select_seed,
    validate_block_size,
)

EVEN_SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}


class TestValidateBlockSize:
    @pytest.mark.parametrize("block_size", [1, 2, 256, 1024, 10240])
    def test_accepts_block_size_factor_of_raster(self, block_size):
        assert validate_block_size(10240, block_size) is None

    @pytest.mark.parametrize("block_size", [3, 300, 1023, 20480])
    def test_raises_when_block_size_is_not_a_factor(self, block_size):
        with pytest.raises(ValueError, match="factor of the raster_size"):
            validate_block_size(10240, block_size)


class TestComputeBufferRadius:
    @pytest.mark.parametrize(
        ("buffer_m", "patch_size", "pixel_resolution_m", "expected"),
        [
            (0.0, 256, 10.0, 0),  # No buffer requested.
            (100.0, 256, 10.0, 1),  # Below one patch, still rounded up to one.
            (1000.0, 256, 10.0, 1),  # 100 pixels, the project default.
            (2560.0, 256, 10.0, 1),  # Exactly one patch.
            (2570.0, 256, 10.0, 2),  # Just above one patch.
            (1000.0, 256, 20.0, 1),  # Coarser pixels shrink the radius.
            (10000.0, 256, 10.0, 4),  # 1000 pixels.
        ],
    )
    def test_converts_meters_to_patch_units(
        self, buffer_m, patch_size, pixel_resolution_m, expected
    ):
        assert (
            compute_buffer_radius(buffer_m, patch_size, pixel_resolution_m) == expected
        )


class TestCreateBufferMask:
    def test_returns_boolean_mask_with_input_shape(self):
        labels = np.zeros((4, 6), dtype=np.uint8)

        keep = create_buffer_mask(labels, 1)

        assert keep.shape == labels.shape
        assert keep.dtype == np.bool

    def test_raises_when_radius_is_too_big(self):
        labels = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="is bigger than patches number"):
            create_buffer_mask(labels, 3)

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
        labels = assign_blocks(3, 10, EVEN_SPLIT)

        assert labels.shape == (10, 10)
        assert labels.dtype == np.uint8
        assert set(np.unique(labels)) <= set(LABELS_TO_SET.values())

    def test_block_counts_follow_the_split_fractions(self):
        labels = assign_blocks(3, 10, EVEN_SPLIT)

        counts = np.bincount(labels.ravel(), minlength=3)

        assert counts[LABELS_TO_SET["test"]] == 15
        assert counts[LABELS_TO_SET["val"]] == 15
        assert counts[LABELS_TO_SET["train"]] == 70

    def test_assigns_everything_to_train_when_other_fractions_are_zero(self):
        labels = assign_blocks(3, 4, {"train": 1.0, "val": 0.0, "test": 0.0})

        assert (labels == LABELS_TO_SET["train"]).all()

    def test_is_deterministic_for_a_given_seed(self):
        first = assign_blocks(3, 8, EVEN_SPLIT)
        second = assign_blocks(3, 8, EVEN_SPLIT)

        np.testing.assert_array_equal(first, second)

    def test_different_seeds_shuffle_differently_but_keep_the_counts(self):
        first = assign_blocks(3, 10, EVEN_SPLIT)
        second = assign_blocks(4, 10, EVEN_SPLIT)

        assert not np.array_equal(first, second)
        np.testing.assert_array_equal(
            np.bincount(first.ravel(), minlength=3),
            np.bincount(second.ravel(), minlength=3),
        )


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


class TestCountsClassesPerPatch:
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

        counts = counts_classes_per_patch(labels, patch_size=2, num_classes=2)

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

        counts = counts_classes_per_patch(labels, patch_size=2, num_classes=3)

        np.testing.assert_array_equal(counts[0, 0], [1, 3, 0])
        np.testing.assert_array_equal(counts[0, 1], [1, 0, 3])

    def test_each_patch_sums_to_the_patch_area(self):
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 5, size=(24, 16), dtype=np.uint8)

        counts = counts_classes_per_patch(labels, patch_size=8, num_classes=5)

        assert counts.shape == (3, 2, 5)
        assert (counts.sum(axis=-1) == 64).all()

    def test_reserves_a_column_for_classes_absent_from_the_labels(self):
        labels = np.zeros((4, 4), dtype=np.uint8)

        counts = counts_classes_per_patch(labels, patch_size=4, num_classes=6)

        assert counts.shape == (1, 1, 6)
        np.testing.assert_array_equal(counts[0, 0], [16, 0, 0, 0, 0, 0])

    def test_counts_are_int64(self):
        labels = np.zeros((2, 2), dtype=np.uint8)

        counts = counts_classes_per_patch(labels, patch_size=2, num_classes=2)

        assert counts.dtype == np.int64


def _split_for_seed(seed, grid_size, patches_per_block, buffer_radius, set_fractions):
    """Rebuild the patch labels and buffer mask a given seed produces."""
    patch_labels = create_labelled_patches(
        assign_blocks(seed, grid_size, set_fractions), patches_per_block
    )
    return patch_labels, create_buffer_mask(patch_labels, buffer_radius)


def _smallest_set_size(patch_labels, keep_mask):
    """The quantity select_seed maximises: kept patches in the thinnest set."""
    return min(
        int((keep_mask & (patch_labels == label)).sum())
        for label in LABELS_TO_SET.values()
    )


class TestSelectSeed:
    GRID_SIZE = 4
    PATCHES_PER_BLOCK = 2
    NUM_PATCHES = GRID_SIZE * PATCHES_PER_BLOCK  # 8 patches per side.

    def _counts(self, num_classes=3, fill=1):
        return np.full(
            (self.NUM_PATCHES, self.NUM_PATCHES, num_classes), fill, dtype=np.int64
        )

    def _select(self, patch_class_count, buffer_radius=0, num_candidates=20):
        return select_seed(
            patch_class_count,
            self.GRID_SIZE,
            buffer_radius,
            self.PATCHES_PER_BLOCK,
            num_candidates,
            EVEN_SPLIT,
        )

    def test_returns_seed_labels_and_mask(self):
        seed, patch_labels, keep_mask, _ = self._select(self._counts())

        assert 0 <= seed < 20
        assert patch_labels.shape == (self.NUM_PATCHES, self.NUM_PATCHES)
        assert keep_mask.shape == (self.NUM_PATCHES, self.NUM_PATCHES)
        assert keep_mask.dtype == np.bool_

    def test_returned_arrays_match_the_returned_seed(self):
        seed, patch_labels, keep_mask, _ = self._select(self._counts(), buffer_radius=1)

        expected_labels, expected_mask = _split_for_seed(
            seed, self.GRID_SIZE, self.PATCHES_PER_BLOCK, 1, EVEN_SPLIT
        )
        np.testing.assert_array_equal(patch_labels, expected_labels)
        np.testing.assert_array_equal(keep_mask, expected_mask)

    def test_returned_block_labels_belong_to_the_returned_seed(self):
        # Regression: block_labels used to leak out of the ranking loop, so it
        # described the last candidate seed instead of the selected one.
        seed, patch_labels, _, block_labels = self._select(
            self._counts(), buffer_radius=1
        )

        np.testing.assert_array_equal(
            block_labels, assign_blocks(seed, self.GRID_SIZE, EVEN_SPLIT)
        )
        np.testing.assert_array_equal(
            create_labelled_patches(block_labels, self.PATCHES_PER_BLOCK), patch_labels
        )

    def test_is_deterministic(self):
        first_seed, first_labels, first_mask, _ = self._select(self._counts())
        second_seed, second_labels, second_mask, _ = self._select(self._counts())

        assert first_seed == second_seed
        np.testing.assert_array_equal(first_labels, second_labels)
        np.testing.assert_array_equal(first_mask, second_mask)

    def test_every_set_contains_every_present_class(self):
        rng = np.random.default_rng(3)
        counts = rng.integers(
            0, 5, size=(self.NUM_PATCHES, self.NUM_PATCHES, 3), dtype=np.int64
        )

        _, patch_labels, keep_mask, _ = self._select(counts)

        present = counts.sum(axis=(0, 1)) > 0
        for label in LABELS_TO_SET.values():
            selected = keep_mask & (patch_labels == label)
            assert (counts[selected].sum(axis=0)[present] > 0).all()

    def test_ignores_classes_absent_from_the_whole_raster(self):
        # Class 2 has no pixel anywhere, so it must not block the selection.
        counts = self._counts(num_classes=3)
        counts[:, :, 2] = 0

        seed, patch_labels, keep_mask, _ = self._select(counts)

        assert 0 <= seed < 20
        assert patch_labels.shape == keep_mask.shape
        totals = counts[keep_mask].sum(axis=0)
        assert totals[0] > 0
        assert totals[2] == 0

    def test_prefers_the_covering_seed_with_the_largest_smallest_set(self):
        counts = self._counts()

        seed, _, _, _ = self._select(counts, buffer_radius=1)

        # Every seed covers the classes here, so the choice is purely the rank.
        sizes = {
            candidate: _smallest_set_size(
                *_split_for_seed(
                    candidate,
                    self.GRID_SIZE,
                    self.PATCHES_PER_BLOCK,
                    1,
                    EVEN_SPLIT,
                )
            )
            for candidate in range(20)
        }
        best = max(sizes.values())
        assert sizes[seed] == best
        # Ties are broken by the lowest seed.
        assert seed == min(c for c, size in sizes.items() if size == best)

    def test_skips_seeds_that_leave_a_class_out_of_a_set(self):
        # Class 1 lives in a single patch, so at most one set can ever hold it.
        counts = self._counts(num_classes=2)
        counts[:, :, 1] = 0
        counts[0, 0, 1] = 5

        with pytest.raises(ValueError, match="no seed in the range"):
            self._select(counts)

    def test_raises_when_there_are_no_candidates(self):
        with pytest.raises(ValueError, match="no seed in the range"):
            self._select(self._counts(), num_candidates=0)

    def test_zero_buffer_keeps_every_patch(self):
        _, _, keep_mask, _ = self._select(self._counts(), buffer_radius=0)

        assert keep_mask.all()

    def test_buffer_removes_patches_along_set_boundaries(self):
        _, _, keep_mask, _ = self._select(self._counts(), buffer_radius=1)

        assert not keep_mask.all()
        assert keep_mask.any()
