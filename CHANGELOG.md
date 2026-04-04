# Changelog

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
