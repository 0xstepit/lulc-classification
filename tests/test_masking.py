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
