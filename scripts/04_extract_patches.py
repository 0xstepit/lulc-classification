"""
Extract patches from the composite and labels raster to create a train/val/test
dataset. Patches are created based on the [patch] configuration and after applying
a spatial buffer around each set block. Since the shunting of blocks into
dataset sets is randomic based on sets size, the script iterates over different
seeds to ensure that all sets contains all the classes present in the original
labels raster.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from src.config import load_config, load_reporter_config
from src.data.labels import build_remap_lookup_table
from src.data.patches import (
    LABELS_TO_SET,
    SETS_TO_LABEL,
    PatchSpec,
    build_patch_specs,
    compute_buffer_radius,
    counts_classes_per_patch,
    select_seed,
    validate_block_size,
)
from src.data.worldcover import compute_class_stats
from src.io import (
    ANALYSIS_DIR,
    GLOBAL_CONFIG,
    MULTISEASONAL_SCENE,
    PATCHES_DIR,
    REPORTER_CONFIG,
    WORLDCOVER_LABELS,
)
from src.logger import setup_logging
from src.reporter.reporter import Reporter

setup_logging()
logger = logging.getLogger(__name__ if __name__ != "__main__" else Path(__file__).stem)


def main():
    cfg = load_config(GLOBAL_CONFIG)

    cfg_reporter = load_reporter_config(REPORTER_CONFIG)
    reporter = Reporter(ANALYSIS_DIR, cfg_reporter)

    patch_size = cfg.patches.size
    block_size = cfg.patches.block_size
    patches_per_block = cfg.patches.patches_per_block
    max_nan_fraction = cfg.patches.max_nan_fraction

    num_label_classes = len(cfg.worldcover.class_names.keys())

    with (
        rasterio.open(MULTISEASONAL_SCENE) as composite_src,
        rasterio.open(WORLDCOVER_LABELS) as labels_src,
    ):
        if composite_src.width != composite_src.height:
            raise ValueError(
                f"composite raster width and height must be equal, obtained {composite_src.width} and {composite_src.height}"
            )

        if (composite_src.width, composite_src.height) != (
            labels_src.width,
            labels_src.height,
        ):
            raise ValueError(
                "composite and labels have different shapes:"
                f"{(composite_src.width, composite_src.height)} != {(labels_src.width, labels_src.height)}"
            )

        if composite_src.crs != labels_src.crs:
            raise ValueError(
                "composite and labels have different CRS:"
                f"{composite_src.crs} != {labels_src.crs}"
            )

        if not composite_src.transform.almost_equals(labels_src.transform):
            raise ValueError("composite and labels rasters are not aligned")

        validate_block_size(composite_src.shape[0], block_size)

        grid_shape = (
            composite_src.height // block_size,
            composite_src.width // block_size,
        )
        grid_size = composite_src.width // block_size
        num_channels = composite_src.count
        pixel_res = abs(composite_src.transform.a)

        logging.info(
            f"raster of size {composite_src.shape} with {num_channels} channels and {pixel_res}m pixel resolution"
        )

        radius = compute_buffer_radius(cfg.patches.buffer, patch_size, pixel_res)
        logging.info(f"buffer radius is {radius} patches")

        labels = labels_src.read(1)
        # labels = labels[labels != 0]
        lut, mapping = build_remap_lookup_table(labels, 0, 256)

        patch_class_count = counts_classes_per_patch(
            labels, patch_size, num_label_classes
        )

        seed, patch_labels, keep_mask = select_seed(
            patch_class_count,
            grid_size,
            radius,
            patches_per_block,
            200,
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

        patch_counters = dict.fromkeys(LABELS_TO_SET, 0)
        samples = []
        discarded = []  # discared patches

        # Iterate over all the blocks containing valid patches.
        for (block_row, block_col), block_specs in sorted(specs_by_block.items()):
            logger.info(
                f"processing block ({block_row}, {block_col})"
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
                    out_dir / f"patch_{patch_counters[spec.set_name]}_feature.npy",
                    patch.astype(np.float32),
                )
                np.save(
                    out_dir / f"patch_{patch_counters[spec.set_name]}_label.npy",
                    patch_labels,
                )
                patch_counters[spec.set_name] += 1


if __name__ == "__main__":
    main()

# def _compute_global_band_medians(files: list[Path], tiles_size: int) -> np.ndarray:
#     block_medians: list[np.ndarray] = []
#
#     with contextlib.ExitStack() as stack:
#         sources = [stack.enter_context(rasterio.open(f)) for f in files]
#
#         for _, window in sources[0].block_windows(1):
#             if (window.width, window.height) != (tiles_size, tiles_size):
#                 continue
#
#             for source in sources:
#                 block = source.read(window=window)
#                 block_medians.append(np.nanmedian(block, axis=(1, 2)))
#
#     return np.median(np.stack(block_medians, axis=0), axis=0)
#
#
# def _fill_seasonal_nans(
#     season_block: list[np.ndarray], global_band_medians: np.ndarray
# ) -> list[np.ndarray]:
#     season_stack = np.stack(season_block, axis=0)  # [n_seasons, C, H, W]
#
#     annual_median = np.nanmedian(season_stack, axis=0)  # [C, H, W]
#
#     still_nan = np.isnan(annual_median)
#     if still_nan.sum() > 0:
#         fallback = np.broadcast_to(
#             global_band_medians[:, None, None], annual_median.shape
#         )
#         annual_median[still_nan] = fallback[still_nan]
#     filled = np.where(np.isnan(season_stack), annual_median[None], season_stack)
#     return [filled[i] for i in range(filled.shape[0])]
