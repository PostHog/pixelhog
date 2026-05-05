# pixelhog

Fast visual regression primitives for Python, implemented in Rust.

`pixelhog` compares screenshots in two complementary ways:
- `diff`: exact pixel-level differences (with anti-alias handling), optional diff image output
- `ssim`: perceptual similarity score in `[0.0, 1.0]`

It also provides spatial clustering (where on the page did things change), early-exit checks,
and WebP thumbnail generation — all accessible through a stateful `Comparison` object that
decodes images once and exposes methods on demand.

## Install (local dev)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -U pip maturin
maturin develop --release
```

## Quickstart

```python
from pixelhog import Comparison, diff, ssim, compare, thumbnail

# -- Stateful object API (recommended) --

cmp = Comparison(baseline_png, current_png)

count = cmp.diff_count()                     # pixel mismatch count
score = cmp.ssim()                           # perceptual similarity
png   = cmp.diff_image()                     # diff visualization (PNG bytes)
thumb = cmp.current_thumbnail(width=200)     # lossless WebP thumbnail

# Spatial clustering — where did things change?
result = cmp.clusters(dilation=8, merge_gap=60)
for cluster in result.clusters:
    print(cluster.bbox.x, cluster.bbox.y, cluster.bbox.width, cluster.bbox.height)

# Early exit — fail fast if too many diffs
capped = cmp.diff_count_capped(max_diffs=1000)

# -- Function-based API (still available) --

# Pixel-level diff + diff PNG
diff_png, diff_count, width, height = diff(baseline_png, current_png)

# Perceptual similarity
score = ssim(baseline_png, current_png)

# One-call gate: count + SSIM + optional diff + optional thumbnail
diff_count, score, width, height, diff_png, thumb = compare(
    baseline_png,
    current_png,
    return_diff=True,
    thumbnail_width=200,
    thumbnail_height=150,
)

# Standalone thumbnail (lossless WebP, Lanczos3 downscale, top-crop)
thumb = thumbnail(current_png, width=200, height=150)
```

## API at a glance

### Comparison (stateful, decode-once)

| Method | Returns | Notes |
|---|---|---|
| `Comparison(baseline_png, current_png)` | `Comparison` | Decode pair once, call methods on demand |
| `Comparison.from_rgba(...)` | `Comparison` | Pre-decoded RGBA buffers |
| `Comparison.batch(pairs)` | `list[Comparison]` | Parallel decode |
| `.diff_count(threshold, include_aa)` | `int` | Pixel mismatch count |
| `.diff_count_capped(max_diffs, ...)` | `int` | Early-exit count |
| `.ssim()` | `float` | Structural similarity |
| `.clusters(dilation, merge_gap, ...)` | `ClustersResult` | Spatial regions of change |
| `.diff_image(...)` | `bytes` (PNG) | Diff visualization |
| `.current_thumbnail(width, height, ...)` | `bytes` (WebP) | Thumbnail of current image |
| `.baseline_thumbnail(width, height, ...)` | `bytes` (WebP) | Thumbnail of baseline image |
| `.size_mismatch` | `bool` | Whether images had different dimensions |

### Function-based API

| Function | Input | Output | Use when |
|---|---|---|---|
| `thumbnail` | PNG bytes | `bytes` (WebP) | You need a preview thumbnail |
| `diff` | PNG bytes | `(diff_png, diff_count, width, height)` | You need a visual diff artifact |
| `diff_count` | PNG bytes | `(diff_count, width, height)` | You only need the mismatch count |
| `ssim` | PNG bytes | `float` | You need perceptual similarity |
| `compare` | PNG bytes | `(diff_count, ssim, w, h, diff_png?, thumb?)` | You want both metrics in one call |
| `diff_rgba` | RGBA bytes + sizes | `(diff_rgba, diff_count, width, height)` | Pre-decoded buffers |
| `diff_count_rgba` | RGBA bytes + sizes | `(diff_count, width, height)` | Count-only on pre-decoded buffers |
| `ssim_rgba` | RGBA bytes + sizes | `float` | SSIM on pre-decoded buffers |
| `compare_rgba` | RGBA bytes + sizes | `(diff_count, ssim, w, h, diff_rgba?, thumb?)` | Combined on pre-decoded buffers |
| `*_batch` | `list[(baseline, current)]` | `list[result]` | Parallel processing of multiple pairs |

## Behavior

- High-level APIs accept PNG bytes and decode internally.
- Smaller images are padded to the larger dimensions with transparent pixels.
- `diff` always generates a diff image.
- `diff_count` and `compare(..., return_diff=False)` skip diff-image generation.
- SSIM uses 11x11 uniform windows with reflect padding.
- For images smaller than 11x11, SSIM falls back to global SSIM.
- No SSIM visualization image is produced.

## Correctness and tests

The test suite is designed to validate both algorithm fidelity and practical product behavior.

- Rust unit/integration tests cover:
  - identical/completely different/partial-diff images
  - threshold behavior
  - different-size padding behavior
  - SSIM behavior (identical, slight change, large change, small-image fallback)
- Canonical pixelmatch fixture tests use the official Mapbox test set:
  - 8 fixture pairs with exact expected mismatch counts
  - expected diff image comparison against golden outputs
  - decoded RGBA byte equality checks to ensure pixel-perfect output matching
- Python integration tests cover:
  - high-level API contracts and error behavior
  - tall-page and subtle-change scenarios
  - cross-validation against a pure-Python reference implementation
    - pixel diff counts must match exactly
    - SSIM must stay within tolerance

Run the full correctness suite:

```bash
# Rust core only
cargo test -p pixelhog

# Full suite including Python integration tests
cargo test
uv run --python 3.12 --with maturin --with pytest --with pillow bash -lc \
  "maturin develop --release && pytest -q"
```

## Benchmarks

The repo includes both Criterion benches and pipeline breakdown tools.

- `cargo bench` runs Criterion API benchmarks (PNG-bytes entry points).
- Breakdown binaries in `examples/` measure where time goes:
  - `breakdown.rs`: decode vs core diff vs encode vs API call
  - `ssim_breakdown.rs`: decode/pad vs core SSIM vs API call
  - `combined_estimate.rs`: separate calls vs combined single-decode flow

Run:

```bash
cargo bench -p pixelhog
cargo run -p pixelhog --release --example breakdown
cargo run -p pixelhog --release --example ssim_breakdown
cargo run -p pixelhog --release --example combined_estimate
```

For screenshot-style workloads, the practical guidance is:
- `diff_count` is cheaper than `diff` when you do not need an artifact.
- `compare(..., return_diff=False)` avoids duplicate decode work when you need both diff-count and SSIM.

## Development

```bash
# Rust tests (includes canonical Mapbox fixture tests)
cargo test

# Python extension + tests
uv run --python 3.12 --with maturin --with pytest --with pillow bash -lc \
  "maturin develop --release && pytest -q"

# Lint/format/type-check
uv run --python 3.12 --with ruff ruff format --check .
uv run --python 3.12 --with ruff ruff check .
uv run --python 3.12 --with ty --with pytest --with pillow ty check . --python .venv
```

## License

This repository is MIT licensed. See [LICENSE](LICENSE).

Algorithm attribution for pixelmatch is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
