import dataclasses

import pytest

from src.config.dataset import PatchesConfig

VALID_SPLIT = {"train": 0.7, "test": 0.15, "val": 0.15}


def _patches_config(**overrides) -> PatchesConfig:
    kwargs = {
        "seed": 3,
        "size": 256,
        "block_size": 1024,
        "buffer": 1000,
        "split": dict(VALID_SPLIT),
    }
    kwargs.update(overrides)
    return PatchesConfig(**kwargs)


class TestPatchesConfig:
    def test_builds_valid_config(self):
        cfg = _patches_config()

        assert cfg.seed == 3
        assert cfg.size == 256
        assert cfg.block_size == 1024
        assert cfg.buffer == 1000
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
        "split",
        [
            {},
            {"train": 1.0},
            {"train": 0.8, "test": 0.2},
            {"train": 0.6, "test": 0.15, "val": 0.15, "holdout": 0.1},
        ],
    )
    def test_raises_when_split_has_not_three_sets(self, split):
        with pytest.raises(ValueError, match="three different sets"):
            _patches_config(split=split)

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

    def test_accepts_any_split_summing_to_one(self):
        split = {"train": 0.5, "test": 0.25, "val": 0.25}

        cfg = _patches_config(split=split)

        assert cfg.split == split
