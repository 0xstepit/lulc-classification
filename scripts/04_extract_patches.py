"""
Extract patches from the features and labels rasters to create a train/val/test
dataset. Patches are created based on the [patch] configuration. Each raster is divided into blocks, and then a spatial buffer is applied on the boundary of the blocks
that are not associated with the same set. Since the shunting of blocks into
the three sets is randomic based on sets size specified in the [patch] config,
the script iterates over different seeds to discover a value that allows to represet all
classes in each set.
"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from lulc.config import load_config, load_reporter_config
from lulc.data.labels import IGNORE_IDX, build_remap_lookup_table
from lulc.data.patches import (
    LABELS_TO_SET,
    PatchSpec,
    build_patch_specs,
    compute_buffer_radius,
    counts_classes_per_patch,
    select_seed,
    validate_block_size,
)
from lulc.data.rasterio import SeasonalStack
from lulc.io import (
    FEATURE_SUFFIX,
    GLOBAL_CONFIG,
    LABEL_SUFFIX,
    NORMALIZATION_PARAMS,
    PATCH_PREFIX,
    PATCHES_DIR,
    PATCHES_METADATA,
    REPORTER_CONFIG,
    REPORTS_DIR,
    SEASONAL_SCENES,
    WORLDCOVER_LABELS,
)
from lulc.logger import setup_logging
from lulc.reporter.reporter import Reporter

setup_logging()
logger = logging.getLogger(__name__ if __name__ != "__main__" else Path(__file__).stem)


def main():
    reporter = Reporter(REPORTS_DIR, load_reporter_config(REPORTER_CONFIG))

    cfg = load_config(GLOBAL_CONFIG)

    patch_size = cfg.patches.size
    block_size = cfg.patches.block_size
    patches_per_block = cfg.patches.patches_per_block
    max_nan_fraction = cfg.patches.max_nan_fraction
    num_label_classes = len(cfg.worldcover.class_names.keys())

    with (
        SeasonalStack(SEASONAL_SCENES) as composite_src,
        rasterio.open(WORLDCOVER_LABELS) as labels_src,
    ):
        # We enforce square rasters for simplicity.
        if composite_src.width != composite_src.height:
            raise ValueError(
                f"composite raster width and height must be equal, "
                f"obtained {composite_src.width} and {composite_src.height}"
            )

        if (composite_src.width, composite_src.height) != (
            labels_src.width,
            labels_src.height,
        ):
            raise ValueError(
                "composite and labels rasters have different shapes:"
                f"{(composite_src.width, composite_src.height)} != {(labels_src.width, labels_src.height)}"
            )

        if composite_src.crs != labels_src.crs:
            raise ValueError(
                "composite and labels rasters have different CRS:"
                f"{composite_src.crs} != {labels_src.crs}"
            )

        if not composite_src.transform.almost_equals(labels_src.transform):
            raise ValueError("composite and labels rasters are not spatially aligned")

        validate_block_size(composite_src.shape[0], block_size)

        # The grid defines the number of blocks in each dimension.
        grid_shape = (
            composite_src.height // block_size,
            composite_src.width // block_size,
        )
        grid_size = grid_shape[0]

        num_channels = composite_src.count
        pixel_resolution = abs(composite_src.transform.a)

        logging.info(
            f"raster of size ({composite_src.shape}) with ({num_channels}) channels "
            f"and ({pixel_resolution}m) pixel resolution, "
            f"is divided into ({grid_size}, {grid_size}) blocks"
        )

        buffer_radius = compute_buffer_radius(
            cfg.patches.buffer, patch_size, pixel_resolution
        )
        logging.info(f"buffer radius is ({buffer_radius}) patches")

        # The label raster is 1 band so we can load in memory without issues.
        labels = labels_src.read(1)

        patch_class_count = counts_classes_per_patch(
            labels, patch_size, num_label_classes
        )

        seed, patch_labels, keep_mask, block_sets = select_seed(
            patch_class_count,
            grid_size,
            buffer_radius,
            patches_per_block,
            cfg.patches.seed_candidates,
            cfg.patches.split,
        )

        logger.info(
            f"best candidate seed is {seed}, keeping {int(keep_mask.sum())}/{int(keep_mask.size)}"
        )

        specs = build_patch_specs(patch_labels, keep_mask)

        for set_ in LABELS_TO_SET.keys():
            (PATCHES_DIR / set_).mkdir(parents=True, exist_ok=True)

        # We have the block size which is equal to the raster window block.
        specs_by_block: dict[tuple[int, int], list[PatchSpec]] = {}
        for spec in specs:
            # Block position
            block = (spec.row // patches_per_block, spec.col // patches_per_block)
            specs_by_block.setdefault(block, []).append(spec)

        # Mask to NaN the WorldCover no data value.
        # labels = np.where(labels == cfg.worldcover.nodata_value, labels, np.nan)
        lut, mapping = build_remap_lookup_table(
            labels, cfg.worldcover.nodata_value, 256
        )
        class_counts = {
            split: np.zeros(len(mapping), dtype=np.int64) for split in LABELS_TO_SET
        }
        patch_counters = dict.fromkeys(LABELS_TO_SET, 0)
        normalization_samples = []
        discarded = []  # discared patches

        # Iterate over all the blocks containing valid patches.
        for (block_row, block_col), block_specs in sorted(specs_by_block.items()):
            logger.info(
                f"processing block ({block_row}, {block_col}) "
                f"with ({len(block_specs)}) valid patches in ({block_specs[0].set_name}) set"
            )

            window = Window(
                col_off=block_col * block_size,
                row_off=block_row * block_size,
                width=block_size,
                height=block_size,
            )

            block_data = composite_src.read(window=window)
            block_labels = lut[labels_src.read(1, window=window)]

            for spec in block_specs:
                # The row offset is computed as the relative position of the current patch in the
                # the block multiplied by the patch size, obtainin this way the number of pixel
                # in the block after which the current patch is placed. Same for the column offset.
                row_off = (spec.row % patches_per_block) * patch_size
                col_off = (spec.col % patches_per_block) * patch_size

                # TODO: this would be much clearer using xarray
                patch = block_data[
                    :, row_off : row_off + patch_size, col_off : col_off + patch_size
                ]
                patch_labels = block_labels[
                    row_off : row_off + patch_size, col_off : col_off + patch_size
                ]

                # Now we filter out if the patch has too many nan:
                nan_fraction_features = float(np.isnan(patch).mean())
                nan_fraction_labels = float(np.isnan(patch_labels).mean())
                if (nan_fraction_features > max_nan_fraction) or (
                    nan_fraction_labels > max_nan_fraction
                ):
                    logger.warning(
                        f"discarding patch ({spec.row, spec.col}) in block ({block_row, block_col})"
                        f"NaN fraction features ({nan_fraction_features}), NaN fraction labels ({nan_fraction_labels})"
                    )
                    discarded.append(
                        {
                            "row": spec.row,
                            "col": spec.col,
                            "nan_fraction_features": {nan_fraction_features},
                            "nan_fraction_labels": {nan_fraction_labels},
                        }
                    )
                    continue

                out_dir = PATCHES_DIR / spec.set_name
                np.save(
                    out_dir
                    / f"{PATCH_PREFIX}{patch_counters[spec.set_name]:04d}{FEATURE_SUFFIX}",
                    patch.astype(np.float32),
                )
                np.save(
                    out_dir
                    / f"{PATCH_PREFIX}{patch_counters[spec.set_name]:04d}{LABEL_SUFFIX}",
                    patch_labels,
                )
                patch_counters[spec.set_name] += 1

                valid = patch_labels != IGNORE_IDX
                class_counts[spec.set_name] += np.bincount(
                    patch_labels[valid].ravel(), minlength=len(mapping)
                )

                if spec.set_name == "train":
                    random_sampler = np.random.choice(
                        [True, False],
                        size=(patch_size, patch_size),
                        p=(
                            cfg.patches.stats_retention_fraction,
                            1 - cfg.patches.stats_retention_fraction,
                        ),
                    )

                    normalization_samples.append(patch[:, random_sampler])

    low, high = cfg.patches.normalization_percentiles
    stacked = np.concatenate(normalization_samples, axis=1)
    normalization = {
        "percentiles": [low, high],
        "low": np.nanpercentile(stacked, low, axis=1).tolist(),
        "high": np.nanpercentile(stacked, high, axis=1).tolist(),
        "median": np.nanmedian(stacked, axis=1).tolist(),
        "mean": np.nanmean(stacked, axis=1).tolist(),
        "std": np.nanstd(stacked, axis=1).tolist(),
    }

    NORMALIZATION_PARAMS.write_text(json.dumps(normalization, indent=4))

    PATCHES_METADATA.write_text(
        json.dumps(
            {
                "seed": seed,
                "buffer_radius": buffer_radius,
                "grid_shape": list(grid_shape),
                "patch_size": patch_size,
                "block_size": block_size,
                "num_channels": num_channels,
                "label_mapping": {str(k): v for k, v in mapping.items()},
                "class_names": {
                    str(index): cfg.worldcover.class_names[value]
                    for value, index in mapping.items()
                },
                "patches_per_split": patch_counters,
                "class_counts_per_split": {
                    split: counts.tolist() for split, counts in class_counts.items()
                },
                "discarded": discarded,
                # We keep also the patch_labels and keep_mask so we can
                # plot the raster split.
                "block_labels": block_sets.tolist(),
                "keep_mask": keep_mask.astype(int).tolist(),
            },
            indent=4,
        )
    )
    reporter.add("patch_split", {"seed": seed, "patches_per_split": patch_counters})
    for split, counts in class_counts.items():
        reporter.add("patches_class_distribution", counts.tolist(), split=split)
    reporter.save("extract_patches.json")


if __name__ == "__main__":
    main()
