import dataclasses

import pytest

from lulc.config.dataset import SET_NAMES, PatchesConfig

VALID_SPLIT = {"train": 0.7, "test": 0.15, "val": 0.15}


def _patches_config(**overrides) -> PatchesConfig:
    kwargs = {
        "seed": 3,
        "seed_candidates": 200,
        "size": 256,
        "block_size": 1024,
        "buffer": 1000,
        "max_nan_fraction": 0.5,
        "normalization_percentiles": [1, 99],
        "stats_retention_fraction": 0.25,
        "split": dict(VALID_SPLIT),
    }
    kwargs.update(overrides)
    return PatchesConfig(**kwargs)


class TestPatchesConfig:
    def test_builds_valid_config(self):
        cfg = _patches_config()

        assert cfg.seed == 3
        assert cfg.seed_candidates == 200
        assert cfg.size == 256
        assert cfg.block_size == 1024
        assert cfg.buffer == 1000
        assert cfg.max_nan_fraction == 0.5
        assert cfg.normalization_percentiles == [1, 99]
        assert cfg.stats_retention_fraction == 0.25
        assert cfg.split == VALID_SPLIT

    def test_is_frozen(self):
        cfg = _patches_config()

        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.size = 128  # type: ignore[misc]

    def test_raises_when_size_is_zero(self):
        with pytest.raises(ValueError, match="patch size"):
            _patches_config(size=0)

    @pytest.mark.parametrize("block_size", [256, 512, 1024, 2048])
    def test_accepts_block_size_multiple_of_patch_size(self, block_size):
        cfg = _patches_config(size=256, block_size=block_size)

        assert cfg.block_size == block_size

    @pytest.mark.parametrize("block_size", [200, 257, 1000])
    def test_raises_when_block_size_is_not_multiple_of_patch_size(self, block_size):
        with pytest.raises(ValueError, match="integer multiple"):
            _patches_config(size=256, block_size=block_size)

    @pytest.mark.parametrize(
        ("size", "block_size", "expected"),
        [(256, 256, 1), (256, 1024, 4), (128, 1024, 8)],
    )
    def test_patches_per_block(self, size, block_size, expected):
        cfg = _patches_config(size=size, block_size=block_size)

        assert cfg.patches_per_block == expected

    @pytest.mark.parametrize(
        "split",
        [
            {},
            {"train": 1.0},
            {"train": 0.8, "test": 0.2},
            {"train": 0.6, "test": 0.15, "val": 0.15, "holdout": 0.1},
            {"train": 0.7, "test": 0.15, "validation": 0.15},
        ],
    )
    def test_raises_when_split_names_do_not_match(self, split):
        with pytest.raises(ValueError, match="split names"):
            _patches_config(split=split)

    def test_accepts_split_names_in_any_order(self):
        split = {name: 1 / len(SET_NAMES) for name in reversed(SET_NAMES)}

        cfg = _patches_config(split=split)

        assert cfg.split == split

    @pytest.mark.parametrize(
        "split",
        [
            {"train": 0.7, "test": 0.15, "val": 0.10},
            {"train": 0.8, "test": 0.15, "val": 0.15},
        ],
    )
    def test_raises_when_split_does_not_sum_to_one(self, split):
        with pytest.raises(ValueError, match="must be 1.0"):
            _patches_config(split=split)

    def test_accepts_split_summing_to_one_up_to_float_error(self):
        # 0.7 + 0.2 + 0.1 is 0.9999999999999999 in binary floating point.
        split = {"train": 0.7, "test": 0.2, "val": 0.1}

        cfg = _patches_config(split=split)

        assert cfg.split == split

    @pytest.mark.parametrize("percentiles", [[], [50], [1, 50, 99], [0, 25, 75, 100]])
    def test_raises_when_normalization_percentiles_are_not_a_pair(self, percentiles):
        with pytest.raises(ValueError, match=r"\[low, high\]"):
            _patches_config(normalization_percentiles=percentiles)

    @pytest.mark.parametrize("percentiles", [[0, 100], [1, 99], [2.5, 97.5]])
    def test_accepts_percentiles_within_bounds(self, percentiles):
        cfg = _patches_config(normalization_percentiles=percentiles)

        assert cfg.normalization_percentiles == percentiles

    @pytest.mark.parametrize("percentiles", [[-1, 99], [1, 101], [-5, 200]])
    def test_raises_when_percentiles_are_out_of_bounds(self, percentiles):
        with pytest.raises(ValueError, match="normalization percentiles"):
            _patches_config(normalization_percentiles=percentiles)

    @pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
    def test_accepts_max_nan_fraction_within_bounds(self, fraction):
        cfg = _patches_config(max_nan_fraction=fraction)

        assert cfg.max_nan_fraction == fraction

    @pytest.mark.parametrize("fraction", [-0.1, 1.1])
    def test_raises_when_max_nan_fraction_is_out_of_bounds(self, fraction):
        with pytest.raises(ValueError, match="max_nan_fraction"):
            _patches_config(max_nan_fraction=fraction)

    @pytest.mark.parametrize("fraction", [0.0, 0.25, 1.0])
    def test_accepts_stats_retention_fraction_within_bounds(self, fraction):
        cfg = _patches_config(stats_retention_fraction=fraction)

        assert cfg.stats_retention_fraction == fraction

    @pytest.mark.parametrize("fraction", [-0.1, 1.1])
    def test_raises_when_stats_retention_fraction_is_out_of_bounds(self, fraction):
        with pytest.raises(ValueError, match="stats_retention_fraction"):
            _patches_config(stats_retention_fraction=fraction)
