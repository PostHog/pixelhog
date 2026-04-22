# Changelog

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
