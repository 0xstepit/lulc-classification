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
