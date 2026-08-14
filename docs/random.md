So there's no performance problem to solve here. But your instinct isn't wrong
in general, so let me separate where it applies from where it doesn't.

Why dask doesn't fit this step

The output isn't an array. dask is at its best for array → array transforms.
Here the terminal operation is "write ~270 irregular sub-selections to 270
separate .npy files." You'd end up wrapping each patch in dask.delayed and
calling compute(), at which point dask is a thread pool with extra ceremony and
a task graph you have to reason about when something goes wrong.

The percentiles get harder, not easier. dask.array.percentile only accepts 1-D
input and is approximate by design — it computes per-chunk quantiles and merges
them. You need exact p1/p99 per band over the training pixels. With dask you'd
either rechunk to a single chunk along the pixel axis (materializing everything,
defeating the point) or hand-roll a merge. The current design sidesteps this:
subsample by 4 on each axis → 171 MB in RAM → exact np.nanpercentile. That's the
reason the subsample parameter exists.

GDAL and threads need care. rioxarray.open_rasterio(..., chunks=...) under a
threaded scheduler shares a GDAL dataset handle across workers. rioxarray
manages this with a lock, but it's a well-known source of silent corruption or
serialization when you get it wrong — and a lock puts you back at
single-threaded anyway.

Consistency. 02_create_seasonal_composite.py already establishes the rasterio +
block_windows idiom, and that script does far more I/O than this one. Two I/O
paradigms in a five-script pipeline is a cost a reviewer will notice.

Where you're right

Your Future Works already says "parallelize raster tile reading and writing with
positional queue" — that's the real want, and it doesn't require dask. GDAL
releases the GIL during LZW decompression, so plain threads scale, with one
dataset handle per thread:

from concurrent.futures import ThreadPoolExecutor

def read_block(block: tuple[int, int]) -> np.ndarray: """Open a private handle
so the GDAL dataset is never shared across threads.""" block_row, block_col =
block window = Window( block_col * block_size, block_row * block_size,
block_size, block_size )

```
  with rasterio.open(MULTISEASONAL_SCENE) as src:
      return src.read(window=window)
```

with ThreadPoolExecutor(max_workers=4) as pool: for block, block_data in
zip(blocks, pool.map(read_block, blocks)): ...

That's ~10 lines, no new dependency paradigm, and it turns 18 s into maybe 6 s.
I'd argue it isn't worth writing for an 18-second one-off, but it's there if you
want the pattern.

Where dask genuinely earns its place

- Phase 6 inference. Tiling the full AOI with 50% overlap, averaging softmax
  probabilities in the overlaps, reassembling into one raster — that is array →
  array, with a real halo/overlap structure. dask.array.overlap.map_overlap is
  exactly the right tool and hand-rolling it is genuinely unpleasant.
- Script 02, in hindsight. nanmedian across a scene axis is a textbook dask
  reduction. You wrote the block loop by hand instead — which was a reasonable
  call for learning, but that's the step where dask would have paid.

So: keep rasterio for step 6, and spend the dask budget on Phase 6 inference
where the problem shape actually matches the tool. If you want, I can note this
in the implementation doc as a recorded decision so it doesn't look like the
option was never considered.
