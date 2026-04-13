# pixelhog

Fast visual regression primitives for Python, implemented in Rust.

`pixelhog` compares screenshots in two complementary ways:
- `diff`: exact pixel-level differences (with anti-alias handling), optional diff image output
- `ssim`: perceptual similarity score in `[0.0, 1.0]`

## Install (local dev)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -U pip maturin
maturin develop --release
```

## Quickstart

```python
from pixelhog import diff, ssim, compare

# 1) Pixel-level diff + diff PNG
# returns: (diff_png_bytes, diff_count, width, height)
diff_png, diff_count, width, height = diff(baseline_png, current_png)

# 2) Perceptual similarity
score = ssim(baseline_png, current_png)

# 3) One-call gate (count + SSIM), no diff image encoding overhead
# returns: (diff_count, ssim, width, height, maybe_diff_png)
diff_count, score, width, height, _ = compare(
    baseline_png,
    current_png,
    return_diff=False,
)
```

## API at a glance

| Function | Input | Output | Use when |
|---|---|---|---|
| `diff` | PNG bytes | `(diff_png, diff_count, width, height)` | You need a visual diff artifact |
| `diff_count` | PNG bytes | `(diff_count, width, height)` | You only need the mismatch count |
| `ssim` | PNG bytes | `float` | You need perceptual similarity |
| `compare` | PNG bytes | `(diff_count, ssim, width, height, optional_diff_png)` | You want both metrics in one call |
| `diff_rgba` | RGBA bytes + sizes | `(diff_rgba, diff_count, width, height)` | You already decoded images in Python/Rust |
| `diff_count_rgba` | RGBA bytes + sizes | `(diff_count, width, height)` | Count-only on pre-decoded buffers |
| `ssim_rgba` | RGBA bytes + sizes | `float` | SSIM on pre-decoded buffers |
| `compare_rgba` | RGBA bytes + sizes | `(diff_count, ssim, width, height, optional_diff_rgba)` | Combined metrics on pre-decoded buffers |
| `diff_batch` | `list[(baseline_png, current_png)]` | `list[diff result]` | Run many diffs in one call |
| `diff_count_batch` | `list[(baseline_png, current_png)]` | `list[count result]` | Batch count-only checks |
| `ssim_batch` | `list[(baseline_png, current_png)]` | `list[float]` | Batch SSIM checks |
| `compare_batch` | `list[(baseline_png, current_png)]` | `list[compare result]` | Batch combined checks |

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
