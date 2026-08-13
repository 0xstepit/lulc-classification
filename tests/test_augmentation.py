import dataclasses

import numpy as np
import pytest
import torch

from lulc.training.augmentation import AugmentationConfig, PatchAugmentation

NUM_CHANNELS = 3
HEIGHT = 4
WIDTH = 5


def make_pair(height=HEIGHT, width=WIDTH):
    """Build a pair whose channel 0 is the label."""
    labels = torch.arange(height * width, dtype=torch.int64).reshape(height, width)
    features = torch.stack(
        [labels.to(torch.float32) + channel for channel in range(NUM_CHANNELS)]
    )

    return features, labels


def generator(seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def distinct_outcomes(augment, draws=64):
    seen = {}
    for _ in range(draws):
        features, labels = augment(*make_pair())
        seen[features.numpy().tobytes()] = (features, labels)
    return list(seen.values())


ALL_OFF = AugmentationConfig(
    horizontal_flip=False, vertical_flip=False, rotate_90=False
)


class TestAugmentationConfig:
    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            AugmentationConfig().horizontal_flip = False  # pyright: ignore[reportAttributeAccessIssue]

    def test_defaults_enable_the_full_dihedral_groud(self):
        cfg = AugmentationConfig()

        assert cfg.horizontal_flip and cfg.vertical_flip and cfg.rotate_90


class TestFeatureLabelCoupling:
    def test_channel_zero_tracks_the_label(self):
        # The test that matters most. Channel 0 starts equal to the label, so a
        # mismatched `dims` between the two flips shows up immediately.
        for seed in range(32):
            augment = PatchAugmentation(AugmentationConfig(), generator(seed))
            features, labels = augment(*make_pair(height=4, width=4))

            np.testing.assert_array_equal(
                features[0].numpy(), labels.numpy().astype(np.float32)
            )

    def test_channel_offsets_are_preserved(self):
        # Every channel must receive the *same* spatial transform, not just
        # channel 0.
        for seed in range(16):
            augment = PatchAugmentation(AugmentationConfig(), generator(seed))
            features, _ = augment(*make_pair(height=4, width=4))

            for channel in range(1, NUM_CHANNELS):
                np.testing.assert_array_equal(
                    features[channel].numpy(), features[0].numpy() + channel
                )


class TestDeterminism:
    def test_the_same_seed_reproduces_the_sequence(self):
        first = [
            f
            for f, _ in (
                PatchAugmentation(AugmentationConfig(), generator(123))(*make_pair())
                for _ in range(1)
            )
        ]
        second = [
            f
            for f, _ in (
                PatchAugmentation(AugmentationConfig(), generator(123))(*make_pair())
                for _ in range(1)
            )
        ]

        np.testing.assert_array_equal(first[0].numpy(), second[0].numpy())

    def test_different_seeds_eventually_diverge(self):
        outputs = set()
        for seed in range(16):
            augment = PatchAugmentation(AugmentationConfig(), generator(seed))
            features, _ = augment(*make_pair(height=4, width=4))
            outputs.add(features.numpy().tobytes())

        assert len(outputs) > 1

    def test_an_absent_generator_still_works(self):
        # Production relies on per-worker seeding rather than an explicit
        # generator, so the None path must not raise.
        augment = PatchAugmentation(AugmentationConfig())

        features, labels = augment(*make_pair(height=4, width=4))

        assert features.shape == (NUM_CHANNELS, 4, 4)
        assert labels.shape == (4, 4)


class TestShapesAndDtypes:
    def test_square_patches_keep_their_shape(self):
        augment = PatchAugmentation(AugmentationConfig(), generator(0))

        for _ in range(16):
            features, labels = augment(*make_pair(height=4, width=4))

            assert features.shape == (NUM_CHANNELS, 4, 4)
            assert labels.shape == (4, 4)

    def test_rotation_transposes_non_square_patches(self):
        # Documents a real constraint rather than a bug: rotate_90 is only
        # shape-preserving because the production patches are 256x256.
        cfg = dataclasses.replace(ALL_OFF, rotate_90=True)
        augment = PatchAugmentation(cfg, generator(0))

        shapes = set()
        for _ in range(64):
            features, _ = augment(*make_pair(height=HEIGHT, width=WIDTH))
            shapes.add(tuple(features.shape))

        assert shapes == {
            (NUM_CHANNELS, HEIGHT, WIDTH),
            (NUM_CHANNELS, WIDTH, HEIGHT),
        }

    def test_dtypes_survive(self):
        augment = PatchAugmentation(AugmentationConfig(), generator(3))

        features, labels = augment(*make_pair(height=4, width=4))

        assert features.dtype == torch.float32
        assert labels.dtype == torch.int64

    def test_output_is_contiguous(self):
        # Flips and rotations return views with negative or permuted strides;
        # downstream collation and .numpy() both want contiguous memory.
        augment = PatchAugmentation(AugmentationConfig(), generator(1))

        features, labels = augment(*make_pair(height=4, width=4))

        assert features.is_contiguous() and labels.is_contiguous()
