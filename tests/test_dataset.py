import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lulc.data.dataset import LULCPatchDataset
from lulc.io import FEATURE_SUFFIX, LABEL_SUFFIX
from lulc.preprocessing.normalization import NormalizationParams, apply_normalization

NUM_CHANNELS = 3
HEIGHT = 4
WIDTH = 5
SHAPE = (NUM_CHANNELS, HEIGHT, WIDTH)


@pytest.fixture
def params() -> NormalizationParams:
    return NormalizationParams(
        low=np.array([0.0, 10.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 20.0, 1.0], dtype=np.float32),
        median=np.array([0.5, 15.0, 0.0], dtype=np.float32),
    )


@pytest.fixture
def patches_dir(tmp_path):
    directory = tmp_path / "train"
    directory.mkdir()
    return directory


def write_patch(patches_dir, index, params, label=None):
    fraction = index / 10.0
    values = params.low + fraction * (params.high - params.low)
    feature = (values[:, None, None] * np.ones(SHAPE, dtype=np.float32)).astype(
        np.float32
    )

    if label is None:
        label = np.full((HEIGHT, WIDTH), index, dtype=np.uint8)

    stem = f"patch_{index:04d}"
    np.save(patches_dir / f"{stem}{FEATURE_SUFFIX}", feature)
    np.save(patches_dir / f"{stem}{LABEL_SUFFIX}", label)

    return feature, label


class RecordingTransform:
    """Transform spy: records the shapes it saw and returns known sentinels."""

    def __init__(self):
        self.calls = []

    def __call__(self, features, labels):
        self.calls.append((tuple(features.shape), tuple(labels.shape)))
        return torch.zeros_like(features), torch.ones_like(labels)


class TestConstruction:
    def test_raises_when_the_directory_is_empty(self, patches_dir, params):
        with pytest.raises(FileNotFoundError):
            LULCPatchDataset(patches_dir, params)

    def test_raises_when_the_directory_does_not_exits(self, patches_dir, params):
        with pytest.raises(FileNotFoundError):
            LULCPatchDataset(patches_dir / "absent", params)

    def test_raises_when_a_feature_has_no_label(self, patches_dir, params):
        # Save feature and patches
        write_patch(patches_dir, 3, params)
        # Save only a feature
        np.save(
            patches_dir / f"patch_004{FEATURE_SUFFIX}",
            np.zeros(SHAPE, dtype=np.float32),
        )

        with pytest.raises(FileNotFoundError):
            LULCPatchDataset(patches_dir, params)

    def test_length_matches_number_of_samples(self, patches_dir, params):
        for index in (1, 2, 3):
            write_patch(patches_dir, index, params)

        assert len(LULCPatchDataset(patches_dir, params)) == 3

    def test_stray_labels_are_ignored(self, patches_dir, params):
        write_patch(patches_dir, 1, params)

        np.save(
            patches_dir / f"patch_004{LABEL_SUFFIX}",
            np.zeros((HEIGHT, WIDTH), dtype=np.float32),
        )

        assert len(LULCPatchDataset(patches_dir, params)) == 1

    def test_samples_are_ordered_by_index(self, patches_dir, params):
        _range = [1, 2, 3, 100]
        for index in _range:
            write_patch(patches_dir, index, params)

        dataset = LULCPatchDataset(patches_dir, params)

        assert [p.name for p in dataset.feature_paths] == [
            f"patch_{index:04d}{FEATURE_SUFFIX}" for index in _range
        ]


class TestGetItem:
    def test_returns_expected_shapes_and_dtypes(self, patches_dir, params):
        write_patch(patches_dir, 3, params)

        features, labels = LULCPatchDataset(patches_dir, params)[0]

        assert features.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert tuple(features.shape) == SHAPE
        assert tuple(labels.shape) == (HEIGHT, WIDTH)

    def test_features_are_normalized_on_read(self, patches_dir, params):
        raw, _ = write_patch(patches_dir, 3, params)

        features, _ = LULCPatchDataset(patches_dir, params)[0]

        np.testing.assert_allclose(
            features.numpy(), apply_normalization(raw, params), atol=1e-6
        )

    def test_every_channel_normalizes_to_the_encoded_fraction(
        self, patches_dir, params
    ):
        write_patch(patches_dir, 3, params)

        features, _ = LULCPatchDataset(patches_dir, params)[0]

        # A per-channel statistic broadcast against the width axis instead of
        # the channel axis would leave the channels disagreeing here.
        np.testing.assert_allclose(features.numpy(), 0.3, atol=1e-6)

    def test_labels_are_returned_unchanged(self, patches_dir, params):
        _, raw = write_patch(patches_dir, 5, params)

        _, labels = LULCPatchDataset(patches_dir, params)[0]

        np.testing.assert_array_equal(labels.numpy(), raw.astype(np.int64))

    def test_features_and_labels_stay_paired(self, patches_dir, params):
        for index in (7, 3, 5):
            write_patch(patches_dir, index, params)

        dataset = LULCPatchDataset(patches_dir, params)

        # Sorted order is 3, 5, 7 and each patch's content encodes its own
        # index, so a crossed pairing surfaces as a mismatch here rather than as
        # mediocre mIoU weeks later.
        for position, index in enumerate((3, 5, 7)):
            features, labels = dataset[position]

            np.testing.assert_allclose(features.numpy(), index / 10.0, atol=1e-6)
            assert torch.all(labels == index)

    def test_channel_count_mismatch_is_rejected(self, patches_dir, params):
        np.save(
            patches_dir / f"patch_0000{FEATURE_SUFFIX}",
            np.zeros((2, HEIGHT, WIDTH), dtype=np.float32),
        )
        np.save(
            patches_dir / f"patch_0000{LABEL_SUFFIX}",
            np.zeros((HEIGHT, WIDTH), dtype=np.uint8),
        )

        with pytest.raises(ValueError):
            LULCPatchDataset(patches_dir, params)[0]

    class TestTransform:
        def test_transform_receives_both_tensors(self, patches_dir, params):
            write_patch(patches_dir, 3, params)
            transform = RecordingTransform()

            LULCPatchDataset(patches_dir, params, transform=transform)[0]

            assert transform.calls == [(SHAPE, (HEIGHT, WIDTH))]

        def test_transform_output_is_returned(self, patches_dir, params):
            write_patch(patches_dir, 3, params)

            features, labels = LULCPatchDataset(
                patches_dir, params, transform=RecordingTransform()
            )[0]

            # The sentinels the spy returns, not the values read from disk.
            assert torch.all(features == 0.0)
            assert torch.all(labels == 1)

        def test_transform_is_optional(self, patches_dir, params):
            write_patch(patches_dir, 3, params)

            features, labels = LULCPatchDataset(patches_dir, params)[0]

            np.testing.assert_allclose(features.numpy(), 0.3, atol=1e-6)
            assert torch.all(labels == 3)

    class TestDataLoaderIntegration:
        def test_batches_collate_with_the_expected_shapes(self, patches_dir, params):
            for index in (1, 2, 3, 4):
                write_patch(patches_dir, index, params)

            loader = DataLoader(
                LULCPatchDataset(patches_dir, params), batch_size=2, shuffle=False
            )
            features, labels = next(iter(loader))

            assert tuple(features.shape) == (2, *SHAPE)
            assert tuple(labels.shape) == (2, HEIGHT, WIDTH)
            assert labels.dtype == torch.int64
