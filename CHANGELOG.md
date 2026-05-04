# Changelog

## 1.2.0

**Spatial clustering.** New `clusters()` method returns connected-component regions of
differing pixels with bounding boxes, pixel counts, and centroids. Uses dilation (default 4px)
to merge nearby fragments into UI-level regions, then two-pass CCL with 8-connectivity and
union-find. Configurable via `dilation`, `min_pixels` (default 16), and `min_side` params.
Results sorted by `pixel_count` descending for triage UIs. ~15% overhead over count-only.

**Performance: threshold=0 fast path.** When threshold is 0 and AA detection is off, the
count-only and mask paths skip `color_delta` entirely and just count u32 mismatches. ~25%
faster across all image sizes for this scenario.

**Performance: early exit.** New `diff_count_capped(max_diffs)` stops processing once enough
diffs are found. Sub-100µs for a quick-fail check regardless of image size.

**Comparison object API.** New `Comparison` class decodes PNGs once at construction, then
exposes individual methods: `diff_count()`, `ssim()`, `clusters()`, `diff_image()`,
`current_thumbnail()`, `baseline_thumbnail()`, `diff_count_capped()`. Also available via
`Comparison.from_rgba()` and `Comparison.batch()`. Exposes `size_mismatch`, `baseline_size`,
and `current_size` properties so callers can detect padding artifacts. Exposed as frozen
`#[pyclass]` with proper `BoundingBox` and `Cluster` result types. Old function-based API
stays untouched.

**Expanded benchmarks.** 9 groups covering 2.1M / 2.5M / 18M pixel images across count-only,
diff image, SSIM, compare, clusters, early exit, identical, and small-diff scenarios.

## 1.1.0

**Diff PNG output is now 97% smaller.** Encoding switched from `Fast + NoFilter + RGBA`
to `Default + Adaptive + RGB`. The alpha channel was always 255 (fully opaque) so stripping
it is lossless. For a typical 1059×674 screenshot diff: 1,405 KB → 49 KB, with ~5ms
additional encode time.

**Thumbnail generation.** New `thumbnail()` function produces lossless WebP thumbnails with
Lanczos3 downscaling. Supports width-only and width+height (top-crop) modes for grid layouts.
Also available as an opt-in on `compare()` / `compare_rgba()` / `compare_batch()` via
`thumbnail_width` / `thumbnail_height` params — generates the thumbnail from the
already-decoded current image buffer at zero extra decode cost.

**Breaking:** `compare`, `compare_rgba`, and `compare_batch` now return a 6-tuple instead of
5-tuple. The new 6th element is `Optional[bytes]` containing the WebP thumbnail when
`thumbnail_width` is set, `None` otherwise.

## 1.0.0

Initial stable release.

- Pixel-level diffing with anti-alias detection, ported from Mapbox's pixelmatch
- SSIM (structural similarity) scoring with 11×11 windowed and global fallback
- PNG and raw RGBA entry points for all operations
- Batch APIs for processing multiple image pairs in parallel
- Combined `compare` call that runs diff + SSIM in a single decode pass
- Automatic padding when images differ in size
- Python bindings via PyO3 with full type stubs
- Validated against the canonical Mapbox pixelmatch test suite
