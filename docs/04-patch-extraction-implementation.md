# Patch extraction implementation

This document collects the code of the patch extraction step, in the order it
should be written. It complements
[04-patch-extraction](./04-patch-extraction.md), which describes the two input
rasters, and [04-spatial-autocorrelation](./04-spatial-autocorrelation.md),
which motivates the buffer between the splits.

The rasters are `5120x5120` at 10 m, so the geometry is fixed:

| Quantity               | Value                             |
| ---------------------- | --------------------------------- |
| Block grid             | `5120 / 1024 = 5x5`               |
| Patches per block      | `1024 / 256 = 4x4`                |
| Candidate patches      | `20x20 = 400`                     |
| Blocks per split       | 17 train / 4 val / 4 test         |
| Buffer radius          | `ceil(1000 / 10 / 256) = 1` patch |
| Patches kept (seed 86) | ~272                              |

______________________________________________________________________

## Step 1: configuration

The `[patches]` section of `config/config.toml`:

```toml
[patches]
size = 256                          # [px] side of a training patch.
block_size = 1024                   # [px] side of a spatial block, must be a multiple of size.
buffer = 1000                       # [m] minimum gap enforced between patches of different splits.
max_nan_fraction = 0.5              # patches with more NaN than this are discarded.
seed = 42
normalization_percentiles = [1, 99] # [low, high] percentiles used for the min-max scaling.
normalization_subsample = 4         # keep one pixel every N on each axis when accumulating statistics.
seed_candidates = 200               # number of seeds evaluated when searching a split assignment.

[patches.split]
train = 0.70
val = 0.15
test = 0.15
```

The matching dataclass in `src/config/dataset.py`. Validating the set names here
is what allows `assign_blocks` to index `set_fractions` without a `KeyError`:

New entries in `src/io.py`:

```python
# Patches directories and files.
# Folder containing the extracted patches, one sub-folder per split.
PATCHES_DIR = DATA_DIR / "patches"
# Per-band statistics used to normalize the patches at load time.
NORMALIZATION_PARAMS = PATCHES_DIR / "normalization_params.json"
# Split assignment, class distribution and label remapping of the extracted dataset.
PATCHES_METADATA = PATCHES_DIR / "dataset_metadata.json"

# Outputs directories.
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
```

______________________________________________________________________

## Step 3: seed selection

```python

def select_seed(
    patch_class_counts: np.ndarray,
    grid_shape: tuple[int, int],
    set_fractions: dict[str, float],
    patches_per_block: int,
    radius: int,
    num_candidates: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Search a split assignment that keeps every class in every set.

    Candidate seeds are ranked by how many patches survive the buffer in the
    smallest of the validation and test sets, then the first candidate covering
    all the classes present in the AOI is accepted.

    Parameters
    ----------
    patch_class_counts : np.ndarray
        Per-patch label counts of shape [rows, cols, num_classes].
    grid_shape : tuple[int, int]
        How many blocks are along the rows and columns of the raster.
    set_fractions : dict[str, float]
        Fraction of blocks assigned to each set.
    patches_per_block : int
        Number of patches along one side of a block.
    radius : int
        Buffer radius expressed in patches.
    num_candidates : int
        Number of seeds to evaluate.

    Returns
    -------
    tuple[int, np.ndarray, np.ndarray]
        The accepted seed, the patch split labels and the buffer keep mask.

    Raises
    ------
    ValueError
        If no candidate seed covers every class in every set.
    """
    present = patch_class_counts.sum(axis=(0, 1)) > 0

    ranked = []
    for seed in range(num_candidates):
        block_labels = assign_blocks(grid_shape, set_fractions, seed)
        patch_labels = create_labelled_patches(block_labels, patches_per_block)
        keep = create_buffer_mask(patch_labels, radius)

        kept = {
            name: int((keep & (patch_labels == label)).sum())
            for name, label in SETS_TO_LABEL.items()
        }
        ranked.append((min(kept["val"], kept["test"]), seed, patch_labels, keep))

    ranked.sort(key=lambda item: item[0], reverse=True)

    for _, seed, patch_labels, keep in ranked:
        covered = True
        for label in SETS_TO_LABEL.values():
            selected = keep & (patch_labels == label)
            counts = patch_class_counts[selected].sum(axis=0)
            if not np.all(counts[present] > 0):
                covered = False
                break

        if covered:
            return seed, patch_labels, keep

    raise ValueError(
        f"no seed among {num_candidates} candidates covers every class in every set"
    )
```

______________________________________________________________________

## Step 6: the extraction script

`scripts/04_extract_patches.py`. The composite is tiled in `1024x1024` blocks,
so reading a single `256x256` window still decompresses the enclosing block.
Grouping the patches per block keeps the whole extraction to one read per block
instead of up to sixteen:

```python
import json
import logging

import numpy as np
import rasterio
from rasterio.windows import Window

from lulc.config import load_config, load_reporter_config
from lulc.data.labels import IGNORE_INDEX, build_label_lut
from lulc.data.patches import (
    SETS_TO_LABEL,
    build_patch_specs,
    buffer_radius,
    compute_patch_class_counts,
    select_seed,
    validate_block_size,
)
from lulc.io import (
    ANALYSIS_DIR,
    GLOBAL_CONFIG,
    MULTISEASONAL_SCENE,
    NORMALIZATION_PARAMS,
    PATCHES_DIR,
    PATCHES_METADATA,
    REPORTER_CONFIG,
    WORLDCOVER_LABELS,
)
from lulc.logger import setup_logging
from lulc.reporter.reporter import Reporter

setup_logging()
logger = logging.getLogger("extract_patches")

# Label values span the WorldCover remapped codes, see [worldcover.class_mapping].
NUM_RASTER_CLASSES = 12


def main():
    cfg = load_config(GLOBAL_CONFIG)
    reporter = Reporter(ANALYSIS_DIR, load_reporter_config(REPORTER_CONFIG))

    patch_size = cfg.patches.size
    block_size = cfg.patches.block_size
    patches_per_block = cfg.patches.patches_per_block

    with (
        rasterio.open(MULTISEASONAL_SCENE) as composite,
        rasterio.open(WORLDCOVER_LABELS) as labels_src,
    ):
        if (composite.width, composite.height) != (labels_src.width, labels_src.height):
            raise ValueError("composite and labels rasters have different shapes")

        if composite.crs != labels_src.crs:
            raise ValueError("composite and labels rasters have different CRS")

        if not composite.transform.almost_equals(labels_src.transform):
            raise ValueError("composite and labels rasters are not aligned")

        validate_block_size(composite.height, block_size)
        validate_block_size(composite.width, block_size)

        grid_shape = (composite.height // block_size, composite.width // block_size)
        num_channels = composite.count
        pixel_size = abs(composite.transform.a)

        logger.info(
            "raster %dx%d, %d channels, %.1f m pixels, block grid %s",
            composite.height,
            composite.width,
            num_channels,
            pixel_size,
            grid_shape,
        )

        labels = labels_src.read(1)

        lut, mapping = build_label_lut(labels)
        logger.info("remapping raster labels %s to contiguous ids", mapping)

        radius = buffer_radius(cfg.patches.buffer, patch_size, pixel_size)
        patch_class_counts = compute_patch_class_counts(
            labels, patch_size, NUM_RASTER_CLASSES
        )

        seed, patch_labels, keep = select_seed(
            patch_class_counts,
            grid_shape,
            cfg.patches.split,
            patches_per_block,
            radius,
            cfg.patches.seed_candidates,
        )
        logger.info(
            "accepted seed %d, keeping %d of %d patches with buffer radius %d",
            seed,
            int(keep.sum()),
            keep.size,
            radius,
        )

        specs = build_patch_specs(patch_labels, keep)

        for split in SETS_TO_LABEL:
            (PATCHES_DIR / split).mkdir(parents=True, exist_ok=True)

        # Patches are grouped per block so that every tile of the composite,
        # which is stored in 1024x1024 blocks, is decompressed exactly once.
        specs_by_block: dict[tuple[int, int], list] = {}
        for spec in specs:
            block = (spec.row // patches_per_block, spec.col // patches_per_block)
            specs_by_block.setdefault(block, []).append(spec)

        counters = dict.fromkeys(SETS_TO_LABEL, 0)
        class_counts = {
            split: np.zeros(len(mapping), dtype=np.int64) for split in SETS_TO_LABEL
        }
        samples = []
        discarded = []

        for (block_row, block_col), block_specs in sorted(specs_by_block.items()):
            logger.info("processing block (%d, %d)", block_row, block_col)

            window = Window(
                block_col * block_size, block_row * block_size, block_size, block_size
            )
            block_data = composite.read(window=window)
            block_labels = lut[labels_src.read(1, window=window)]

            for spec in block_specs:
                row_off = (spec.row % patches_per_block) * patch_size
                col_off = (spec.col % patches_per_block) * patch_size

                patch = block_data[
                    :, row_off : row_off + patch_size, col_off : col_off + patch_size
                ]
                patch_label = block_labels[
                    row_off : row_off + patch_size, col_off : col_off + patch_size
                ]

                nan_fraction = float(np.isnan(patch).mean())
                if nan_fraction > cfg.patches.max_nan_fraction:
                    logger.warning(
                        "discarding patch (%d, %d), NaN fraction %.3f",
                        spec.row,
                        spec.col,
                        nan_fraction,
                    )
                    discarded.append(
                        {"row": spec.row, "col": spec.col, "nan_fraction": nan_fraction}
                    )
                    continue

                index = counters[spec.split]
                counters[spec.split] += 1

                out_dir = PATCHES_DIR / spec.split
                np.save(
                    out_dir / f"patch_{index:04d}_input.npy", patch.astype(np.float32)
                )
                np.save(out_dir / f"patch_{index:04d}_label.npy", patch_label)

                valid = patch_label != IGNORE_INDEX
                class_counts[spec.split] += np.bincount(
                    patch_label[valid].ravel(), minlength=len(mapping)
                )

                # Statistics are accumulated on the training set only, otherwise the
                # normalization would leak information from the held out regions.
                if spec.split == "train":
                    step = cfg.patches.normalization_subsample
                    samples.append(patch[:, ::step, ::step].reshape(num_channels, -1))

    logger.info("extracted patches per split: %s", counters)

    low, high = cfg.patches.normalization_percentiles
    stacked = np.concatenate(samples, axis=1)
    normalization = {
        "percentiles": [low, high],
        "low": np.nanpercentile(stacked, low, axis=1).tolist(),
        "high": np.nanpercentile(stacked, high, axis=1).tolist(),
        "median": np.nanmedian(stacked, axis=1).tolist(),
    }

    NORMALIZATION_PARAMS.write_text(json.dumps(normalization, indent=4))

    PATCHES_METADATA.write_text(
        json.dumps(
            {
                "seed": seed,
                "buffer_radius": radius,
                "grid_shape": list(grid_shape),
                "patch_size": patch_size,
                "block_size": block_size,
                "num_channels": num_channels,
                "label_mapping": {str(k): v for k, v in mapping.items()},
                "class_names": {
                    str(index): cfg.worldcover.class_names[value]
                    for value, index in mapping.items()
                },
                "patch_labels": patch_labels.tolist(),
                "keep": keep.astype(int).tolist(),
                "patches_per_split": counters,
                "class_counts_per_split": {
                    split: counts.tolist() for split, counts in class_counts.items()
                },
                "discarded": discarded,
            },
            indent=4,
        )
    )

    reporter.add("patch_split", {"seed": seed, "patches_per_split": counters})
    for split, counts in class_counts.items():
        reporter.add("patch_class_distribution", counts.tolist(), split=split)
    reporter.save("extract_patches.json")


if __name__ == "__main__":
    main()
```

Run it with:

```sh
make run-script script=04_extract_patches
```

______________________________________________________________________

## Step 7: normalization

The patches are stored raw and normalized at load time, so the scheme can be
changed without extracting the dataset again, and the random forest and the
U-Net share a single code path.

`src/preprocessing/normalization.py`:

```python
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalizationParams:
    """Per-channel statistics accumulated on the training split.

    Attributes
    ----------
    low : Value of the low percentile for each channel.
    high : Value of the high percentile for each channel.
    median : Median of each channel, used to fill the missing observations.
    """

    low: np.ndarray
    high: np.ndarray
    median: np.ndarray

    @classmethod
    def load(cls, file_path: Path) -> "NormalizationParams":
        """Read the statistics produced by the patch extraction."""
        params = json.loads(file_path.read_text())

        return cls(
            low=np.asarray(params["low"], dtype=np.float32),
            high=np.asarray(params["high"], dtype=np.float32),
            median=np.asarray(params["median"], dtype=np.float32),
        )


def apply_normalization(patch: np.ndarray, params: NormalizationParams) -> np.ndarray:
    """Scale a patch to [0, 1] and fill its missing observations.

    Values are clipped to the percentile range before scaling, because a handful
    of specular pixels would otherwise compress the whole reflectance range into
    a thin slice.

    Parameters
    ----------
    patch : np.ndarray
        Raw patch of shape [C, H, W].
    params : NormalizationParams
        Statistics accumulated on the training split.

    Returns
    -------
    np.ndarray
        Normalized patch of shape [C, H, W], float32, without NaN.
    """
    low = params.low[:, None, None]
    high = params.high[:, None, None]

    # Constant channels would divide by zero.
    span = np.where(high > low, high - low, 1.0)

    normalized = (np.clip(patch, low, high) - low) / span

    median = np.clip(params.median[:, None, None], low, high)
    fill = np.broadcast_to((median - low) / span, normalized.shape)

    return np.where(np.isnan(normalized), fill, normalized).astype(np.float32)
```

______________________________________________________________________

## Step 8: tests

`tests/test_patches.py`:

```python
import numpy as np
import pytest

from lulc.data.patches import (
    SETS_TO_LABEL,
    assign_blocks,
    buffer_radius,
    build_patch_specs,
    create_buffer_mask,
    create_labelled_patches,
    validate_block_size,
)

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


class TestAssignBlocks:
    def test_shape_and_dtype(self):
        labels = assign_blocks((5, 5), FRACTIONS, seed=42)
        assert labels.shape == (5, 5)
        assert labels.dtype == np.uint8

    def test_is_deterministic(self):
        first = assign_blocks((5, 5), FRACTIONS, seed=42)
        second = assign_blocks((5, 5), FRACTIONS, seed=42)
        assert np.array_equal(first, second)

    def test_different_seeds_give_different_assignments(self):
        first = assign_blocks((5, 5), FRACTIONS, seed=42)
        second = assign_blocks((5, 5), FRACTIONS, seed=43)
        assert not np.array_equal(first, second)

    def test_every_set_is_represented(self):
        labels = assign_blocks((5, 5), FRACTIONS, seed=42)
        for label in SETS_TO_LABEL.values():
            assert (labels == label).sum() > 0

    def test_all_blocks_are_assigned(self):
        labels = assign_blocks((5, 5), FRACTIONS, seed=42)
        assert labels.size == 25


class TestCreateLabelledPatches:
    def test_expands_every_block(self):
        labels = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        patches = create_labelled_patches(labels, 4)
        assert patches.shape == (8, 8)

    def test_preserves_block_adjacency(self):
        labels = assign_blocks((5, 5), FRACTIONS, seed=42)
        patches = create_labelled_patches(labels, 4)

        for row in range(5):
            for col in range(5):
                block = patches[row * 4 : (row + 1) * 4, col * 4 : (col + 1) * 4]
                assert (block == labels[row, col]).all()


class TestBufferRadius:
    def test_rounds_up(self):
        assert buffer_radius(1000, 256, 10.0) == 1

    def test_larger_buffer_needs_larger_radius(self):
        assert buffer_radius(3000, 256, 10.0) == 2

    def test_rejects_invalid_pixel_size(self):
        with pytest.raises(ValueError):
            buffer_radius(1000, 256, 0.0)


class TestCreateBufferMask:
    def test_uniform_grid_keeps_everything(self):
        labels = np.zeros((4, 4), dtype=np.uint8)
        assert create_buffer_mask(labels, radius=1).all()

    def test_checkerboard_keeps_nothing(self):
        labels = np.indices((4, 4)).sum(axis=0).astype(np.uint8) % 2
        assert not create_buffer_mask(labels, radius=1).any()

    def test_zero_radius_keeps_everything(self):
        labels = np.indices((4, 4)).sum(axis=0).astype(np.uint8) % 2
        assert create_buffer_mask(labels, radius=0).all()

    def test_kept_patches_have_no_foreign_neighbour(self):
        labels = create_labelled_patches(assign_blocks((5, 5), FRACTIONS, 42), 4)
        keep = create_buffer_mask(labels, radius=1)

        rows, cols = labels.shape
        for row in range(rows):
            for col in range(cols):
                if not keep[row, col]:
                    continue

                neighbours = labels[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
                assert (neighbours == labels[row, col]).all()


class TestBuildPatchSpecs:
    def test_only_kept_patches_are_listed(self):
        labels = np.zeros((3, 3), dtype=np.uint8)
        keep = np.zeros((3, 3), dtype=bool)
        keep[1, 1] = True

        specs = build_patch_specs(labels, keep)

        assert len(specs) == 1
        assert (specs[0].row, specs[0].col, specs[0].split) == (1, 1, "train")

    def test_window_matches_the_grid_position(self):
        labels = np.zeros((3, 3), dtype=np.uint8)
        keep = np.ones((3, 3), dtype=bool)

        spec = build_patch_specs(labels, keep)[4]
        window = spec.window(256)

        assert (window.col_off, window.row_off) == (256, 256)


class TestValidateBlockSize:
    def test_accepts_exact_multiple(self):
        validate_block_size(5120, 1024)

    def test_rejects_remainder(self):
        with pytest.raises(ValueError):
            validate_block_size(5000, 1024)
```

Run them with:

```sh
make unit-tests
```

______________________________________________________________________

## Order of work

Steps 1 to 5 and step 7 are pure functions: write them together with the tests
of step 8 and their behaviour is known before touching a 3.8 GB raster. Then run
step 6, which should report around 270 patches over 25 block reads. Before
moving on, check one saved sample:

```python
import numpy as np

patch = np.load("data/patches/train/patch_0000_input.npy")
label = np.load("data/patches/train/patch_0000_label.npy")

assert patch.shape == (52, 256, 256) and patch.dtype == np.float32
assert label.shape == (256, 256) and label.max() <= 7
```

______________________________________________________________________

## Left out

Two deliverables of the phase are not covered here:

- The split map figure, `outputs/figures/spatial_split_map.png`. It only needs
  `patch_labels` and `keep`, both saved in `dataset_metadata.json`. For the
  background, a decimated read of the JJA true colour bands: season index 2
  gives a band offset of 26, so bands 29, 28 and 27 for B04, B03 and B02.
- `src/data/patch_dataset.py`, the `torch.utils.data.Dataset`, which needs torch
  added to `pyproject.toml`. Everything above is torch free and already feeds
  the random forest baseline of Phase 5.
