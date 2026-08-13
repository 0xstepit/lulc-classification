import json

import numpy as np
import pytest

from lulc.preprocessing.normalization import NormalizationParams, apply_normalization


@pytest.fixture
def params() -> NormalizationParams:
    return NormalizationParams(
        low=np.array([0.0, 10.0], dtype=np.float32),
        high=np.array([1.0, 20.0], dtype=np.float32),
        median=np.array([0.5, 15.0], dtype=np.float32),
    )


class TestNormalizationParams:
    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError):
            NormalizationParams(
                low=np.zeros(3, dtype=np.float32),
                high=np.ones(2, dtype=np.float32),
                median=np.ones(2, dtype=np.float32),
            )

    def test_inverted_percentiles_are_rejected(self):
        with pytest.raises(ValueError):
            NormalizationParams(
                low=np.ones(1, dtype=np.float32),
                high=np.zeros(1, dtype=np.float32),
                median=np.zeros(1, dtype=np.float32),
            )

    def test_normalized_median_is_centred(self, params):
        np.testing.assert_allclose(params.normalized_median, [0.5, 0.5], atol=1e-6)

    def test_round_trips_through_json(self, tmp_path, params):
        path = tmp_path / "normalization_params.json"
        path.write_text(
            json.dumps(
                {
                    "low": params.low.tolist(),
                    "high": params.high.tolist(),
                    "median": params.median.tolist(),
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                    "percentiles": [1, 99],
                }
            )
        )
        loaded = NormalizationParams.from_json(path)

        np.testing.assert_allclose(loaded.high, params.high)
        assert loaded.num_channels == 2


class TestApplyNormalization:
    def test_output_is_bounded_and_float32(self, params):
        rng = np.random.default_rng(42)
        patch = np.stack(
            [
                rng.uniform(-1.0, 2.0, (8, 8)),
                rng.uniform(5.0, 25.0, (8, 8)),
            ]
        ).astype(np.float32)

        result = apply_normalization(patch, params)

        assert result.dtype == np.float32
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_percentile_bounds_map_to_zero_and_one(self, params):
        patch = np.stack([np.full((2, 2), 0.0), np.full((2, 2), 20.0)]).astype(
            np.float32
        )

        result = apply_normalization(patch, params)

        np.testing.assert_allclose(result[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[1], 1.0, atol=1e-6)

    def test_values_beyond_percentiles_are_clipped(self, params):
        patch = np.stack([np.full((2, 2), -5.0), np.full((2, 2), 999.0)]).astype(
            np.float32
        )

        result = apply_normalization(patch, params)

        np.testing.assert_allclose(result[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[1], 1.0, atol=1e-6)

    def test_nan_is_replaced_by_the_normalized_median(self, params):
        patch = np.full((2, 2, 2), np.nan, dtype=np.float32)
        print(patch)

        result = apply_normalization(patch, params)
        print(result)

        assert not np.isnan(result).any()
        np.testing.assert_allclose(result, 0.5, atol=1e-6)

    def test_constant_band_maps_to_zero(self):
        constant = NormalizationParams(
            low=np.array([3.0], dtype=np.float32),
            high=np.array([3.0], dtype=np.float32),
            median=np.array([3.0], dtype=np.float32),
        )
        result = apply_normalization(
            np.full((1, 4, 4), 3.0, dtype=np.float32), constant
        )

        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_channel_mismatch_is_rejected(self, params):
        with pytest.raises(ValueError):
            apply_normalization(np.zeros((5, 4, 4), dtype=np.float32), params)

    def test_non_three_dimensional_input_is_rejected(self, params):
        with pytest.raises(ValueError):
            apply_normalization(np.zeros((2, 4), dtype=np.float32), params)
