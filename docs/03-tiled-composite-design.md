# Tiled Seasonal Composite — Design

## Problem

The naive implementation of `02_create_seasonal_composite.py` loads all scenes
for a season into memory simultaneously before computing the median. With 10
scenes at ~480 MB each (compressed on disk), the in-memory footprint explodes:

- Each scene decompresses to ~1.3 GB in RAM (13 bands × 5000×5000 × float32)
- `scenes` list: 10 × 1.3 GB = ~13 GB
- `np.stack(scenes)` creates a full [10, 13, 5000, 5000] array while `scenes`
  still lives: another ~13 GB → **26 GB**
- `np.nanmedian` sorts a copy of the stack internally to find the median:
  another ~13 GB → **~40 GB**
- Plus transient allocations during the per-scene loop (~3 GB)

**Peak usage: ~40–50 GB for a single season.**

______________________________________________________________________

## Root Cause

The loop order is wrong. The current structure is:

```
outer = files
    read entire file into RAM
    append to scenes list
stack all scenes → median → write
```

All N scenes must live in RAM at the same time because the write happens at the
end.

______________________________________________________________________

## Solution: Swap the Loop Order

Process one spatial chunk at a time across all scenes, and write immediately:

```
open all file handles
create output file
outer = spatial windows (e.g. 512×512 tiles)
    inner = files
        read window from file
        apply SCL mask
        compute indices
        → [13, h, w] float32 chunk
    stack N chunks → nanmedian → write window to output
```

At any moment, RAM holds only `N_scenes × 13 × chunk_h × chunk_w × 4 bytes`. For
512×512 chunks: **~260 MB peak** regardless of raster size.

______________________________________________________________________

## Implementation

### Prerequisite: write source files with internal tiling

`block_windows()` yields windows aligned to the file's internal block structure.
For this to produce square 512×512 chunks — rather than full-width strips — the
source files must be written with tiling enabled. Set this in the profile when
saving each downloaded scene:

```python
profile.update(
    tiled=True,
    blockxsize=512,
    blockysize=512,
    compress="lzw",
)
```

All files in a season directory come from the same pipeline so they share the
same block layout.

### `_process_window`

Reads one window from one open source file, applies the SCL mask, computes
indices from the **masked** float32 bands, and returns a `[13, h, w]` float32
array:

```python
def _process_window(
    src,
    window: Window,
    cfg,
    scl_band_idx: int,
    mask_classes: list[int],
) -> np.ndarray:
    data = src.read(window=window)  # [11, h, w] uint16
    mask = create_scl_mask(data, scl_band_idx, mask_classes)
    masked_data = apply_scl_mask(data, scl_band_idx, mask)  # [10, h, w] float32

    nir = masked_data[cfg.indices.get_channel("ndvi", "nir")]
    red = masked_data[cfg.indices.get_channel("ndvi", "red")]
    ndvi = compute_ndvi(nir, red)[np.newaxis]

    swir = masked_data[cfg.indices.get_channel("ndbi", "swir")]
    nir = masked_data[cfg.indices.get_channel("ndbi", "nir")]
    ndbi = compute_ndbi(swir, nir)[np.newaxis]

    green = masked_data[cfg.indices.get_channel("ndwi", "green")]
    nir = masked_data[cfg.indices.get_channel("ndwi", "nir")]
    ndwi = compute_ndwi(green, nir)[np.newaxis]

    return np.concatenate([masked_data, ndvi, ndbi, ndwi], axis=0)  # [13, h, w]
```

Note: indices are now derived from `masked_data` (float32, cloudy pixels already
NaN) rather than the raw `data` array.

### Main loop

`contextlib.ExitStack` manages a variable number of open file handles.
`block_windows()` is called on the first source — all files share the same
layout so their windows are identical:

```python
import contextlib

with contextlib.ExitStack() as stack:
    sources = [stack.enter_context(rasterio.open(f)) for f in files]

    profile = sources[0].profile.copy()
    profile.update(count=13, dtype="float32")

    with rasterio.open(out_file, "w", **profile) as dst:
        for _, window in sources[0].block_windows(1):
            chunk_scenes = [
                _process_window(src, window, cfg, scl_band_idx, mask_classes)
                for src in sources
            ]
            composite_chunk = create_seasonal_composite(chunk_scenes)
            dst.write(composite_chunk, window=window)
```

`create_seasonal_composite` in `composites.py` requires **no changes** — it
already does `np.stack → np.nanmedian` and works on any spatial size.

______________________________________________________________________

## Memory Comparison

|                            | Current       | Tiled (512×512 chunks) |
| -------------------------- | ------------- | ---------------------- |
| Per scene in RAM           | ~1.3 GB       | ~13 MB                 |
| All scenes (`scenes` list) | ~13 GB        | ~130 MB                |
| `np.stack` copy            | +13 GB        | +130 MB                |
| `np.nanmedian` sort copy   | +13 GB        | +130 MB                |
| **Peak**                   | **~40–50 GB** | **~390 MB**            |
