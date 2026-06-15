import numpy as np
import pytest

from src.preprocessing.masking import apply_scl_mask, create_scl_mask


def _scene(n_bands: int, height: int, width: int, fill: int = 0) -> np.ndarray:
    return np.full((n_bands, height, width), fill, dtype=np.uint16)


class TestCreateSclMask:
    def test_no_pixels_masked_when_no_class_matches(self):
        data = _scene(11, 3, 3, fill=3)
        mask = create_scl_mask(data, scl_band_index=0, mask_classes=[0, 1, 2, 10])
        assert not mask.any()

    def test_all_pixels_masked_when_all_match(self):
        data = _scene(11, 3, 3, fill=3)
        mask = create_scl_mask(data, scl_band_index=0, mask_classes=[0, 1, 2, 3])
        assert mask.all()

    def test_partial_pixels_masked_when_some_pixels_match(self):
        data = _scene(11, 3, 3, fill=3)

        # Two pixels doen't match the mask filter.
        data[0, 0, 0] = 100
        data[0, 2, 2] = 100

        mask = create_scl_mask(data, scl_band_index=0, mask_classes=[0, 1, 2, 3])

        num_pixels = mask.size
        total_matched = np.sum(mask)

        assert not mask[0, 0]
        assert not mask[2, 2]
        assert total_matched == num_pixels - 2

    def test_partial_pixels_masked_multiple_classes(self):
        data = _scene(11, 3, 3, fill=0)

        data[0, 1, 1] = 3
        data[0, 0, 0] = 1
        data[0, 1, 2] = 1
        data[0, 2, 2] = 1

        mask = create_scl_mask(data, scl_band_index=0, mask_classes=[1, 2, 3])

        total_matched = np.sum(mask)

        assert total_matched == 4

    def test_empty_mask_classes_masks_nothing(self):
        data = _scene(11, 3, 3, fill=3)
        mask = create_scl_mask(data, scl_band_index=0, mask_classes=[])
        assert not mask.all()


class TestApplySclMask:
    def test_scl_band_removed_from_output(self):
        data = _scene(2, 3, 3, fill=3)
        mask = np.zeros((3, 3), dtype=bool)

        result = apply_scl_mask(data, scl_band_index=1, mask=mask)

        assert result.shape[0] == 1

    def test_scl_at_first_index_removed(self):
        data = _scene(4, 3, 3, fill=1)
        mask = np.zeros((3, 3), dtype=bool)

        result = apply_scl_mask(data, scl_band_index=0, mask=mask)

        assert result.shape == (3, 3, 3)

    def test_scl_at_last_index_removed(self):
        data = _scene(4, 3, 3, fill=1)
        mask = np.zeros((3, 3), dtype=bool)

        result = apply_scl_mask(data, scl_band_index=3, mask=mask)

        assert result.shape == (3, 3, 3)

    def test_masked_pixels_are_nan_in_all_bands(self):
        data = _scene(4, 3, 3, fill=100)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        result = apply_scl_mask(data, scl_band_index=3, mask=mask)

        assert np.all(np.isnan(result[:, 1, 1]))

    def test_unmasked_pixels_preserve_values(self):
        data = _scene(4, 3, 3, fill=500)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        result = apply_scl_mask(data, scl_band_index=3, mask=mask)

        unmasked = result[:, mask == False]
        assert np.all(unmasked == 500)

    def test_output_dtype_is_float32(self):
        data = _scene(4, 3, 3, fill=1)
        mask = np.zeros((3, 3), dtype=bool)

        result = apply_scl_mask(data, scl_band_index=3, mask=mask)

        assert result.dtype == np.float32

    def test_fully_masked_scene_all_nan(self):
        data = _scene(4, 3, 3, fill=1)
        mask = np.ones((3, 3), dtype=bool)

        result = apply_scl_mask(data, scl_band_index=3, mask=mask)

        assert np.all(np.isnan(result))

    def test_empty_mask_no_nans(self):
        data = _scene(4, 3, 3, fill=1)
        mask = np.zeros((3, 3), dtype=bool)

        result = apply_scl_mask(data, scl_band_index=3, mask=mask)

        assert not np.any(np.isnan(result))

    def test_raises_when_scl_index_out_of_bounds(self):
        data = _scene(4, 3, 3)
        mask = np.zeros((3, 3), dtype=bool)

        with pytest.raises(ValueError):
            apply_scl_mask(data, scl_band_index=4, mask=mask)

    def test_raises_when_single_band_input(self):
        data = _scene(1, 3, 3)
        mask = np.zeros((3, 3), dtype=bool)

        with pytest.raises(ValueError):
            apply_scl_mask(data, scl_band_index=0, mask=mask)
