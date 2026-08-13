import numpy as np
import pytest

from lulc.preprocessing.indices import compute_ndbi, compute_ndvi, compute_ndwi


@pytest.mark.parametrize("fn", [compute_ndvi, compute_ndbi, compute_ndwi])
class TestSharedBehavior:
    def test_output_dtype_is_float32(self, fn):
        result = fn(np.array([0.5]), np.array([0.1]))
        assert result.dtype == np.float32

    def test_shape_preserved(self, fn):
        a = np.ones((4, 5))
        b = np.ones((4, 5)) * 0.5
        assert fn(a, b).shape == (4, 5)

    def test_zero_denominator_is_nan(self, fn):
        result = fn(np.array([0.0]), np.array([0.0]))
        assert np.isnan(result).all()

    def test_nan_input_propagates(self, fn):
        result = fn(np.array([np.nan]), np.array([0.5]))
        assert np.isnan(result).all()

    def test_output_bounded_in_minus_one_to_one(self, fn):
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 1, (20, 20)).astype(np.float32)
        b = rng.uniform(0, 1, (20, 20)).astype(np.float32)
        result = fn(a, b)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0) and np.all(valid <= 1.0)

    def test_equal_bands_is_zero(self, fn):
        a = np.array([0.4])
        result = fn(a, a.copy())
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_only_zero_denominator_pixel_is_nan(self, fn):
        a = np.array([0.5, 0.0])
        b = np.array([0.1, 0.0])
        result = fn(a, b)
        assert not np.isnan(result[0])
        assert np.isnan(result[1])


class TestNdvi:
    def test_known_value(self):
        nir = np.array([0.5])
        red = np.array([0.1])
        np.testing.assert_allclose(compute_ndvi(nir, red), [2 / 3], atol=1e-5)

    def test_nir_dominant_returns_positive(self):
        np.testing.assert_allclose(
            compute_ndvi(np.array([1.0]), np.array([0.0])), [1.0], atol=1e-6
        )

    def test_red_dominant_returns_negative(self):
        np.testing.assert_allclose(
            compute_ndvi(np.array([0.0]), np.array([1.0])), [-1.0], atol=1e-6
        )


class TestNdbi:
    def test_known_value(self):
        swir = np.array([0.3])
        nir = np.array([0.2])
        np.testing.assert_allclose(compute_ndbi(swir, nir), [0.2], atol=1e-5)

    def test_swir_dominant_returns_positive(self):
        np.testing.assert_allclose(
            compute_ndbi(np.array([1.0]), np.array([0.0])), [1.0], atol=1e-6
        )

    def test_nir_dominant_returns_negative(self):
        np.testing.assert_allclose(
            compute_ndbi(np.array([0.0]), np.array([1.0])), [-1.0], atol=1e-6
        )


class TestNdwi:
    def test_known_value(self):
        green = np.array([0.3])
        nir = np.array([0.1])
        np.testing.assert_allclose(compute_ndwi(green, nir), [0.5], atol=1e-5)

    def test_green_dominant_returns_positive(self):
        np.testing.assert_allclose(
            compute_ndwi(np.array([1.0]), np.array([0.0])), [1.0], atol=1e-6
        )

    def test_nir_dominant_returns_negative(self):
        np.testing.assert_allclose(
            compute_ndwi(np.array([0.0]), np.array([1.0])), [-1.0], atol=1e-6
        )
