# pixelhog

Rust-powered pixel-level diffing and SSIM scoring for PNG bytes, exposed to Python via PyO3.

## Python API

```python
from pixelhog import (
    compare,
    compare_batch,
    compare_rgba,
    diff,
    diff_batch,
    diff_count,
    diff_count_batch,
    diff_count_rgba,
    diff_rgba,
    ssim,
    ssim_batch,
    ssim_rgba,
)

# PNG bytes in -> PNG diff out
# Returns: (diff_png_bytes, diff_count, width, height)
diff_png, diff_count, width, height = diff(baseline_png, current_png)

# Count-only variant (no diff image encode)
# Returns: (diff_count, width, height)
diff_count, width, height = diff_count(baseline_png, current_png)

# SSIM score from 0.0 to 1.0
score = ssim(baseline_png, current_png)

# Combined call (single decode path). Optional diff image.
# Returns: (diff_count, ssim, width, height, diff_png_or_none)
diff_count, score, width, height, maybe_diff = compare(
    baseline_png,
    current_png,
    return_diff=True,
)

# Lower-level RGBA APIs (raw RGBA bytes + explicit sizes)
diff_rgba_bytes, diff_count, width, height = diff_rgba(
    baseline_rgba,
    baseline_width,
    baseline_height,
    current_rgba,
    current_width,
    current_height,
)
score = ssim_rgba(
    baseline_rgba,
    baseline_width,
    baseline_height,
    current_rgba,
    current_width,
    current_height,
)

# Batch APIs (input: list[(baseline_png, current_png)])
diffs = diff_batch([(baseline_png, current_png)])
scores = ssim_batch([(baseline_png, current_png)])
combined = compare_batch([(baseline_png, current_png)], return_diff=False)
```

## Behavior

- Accepts PNG bytes for high-level APIs.
- Pads smaller image to larger dimensions with transparent pixels.
- `diff` always returns a diff PNG image.
- `diff_count` skips diff-image generation for lower overhead.
- `compare` computes pixel diff count + SSIM in one call; `return_diff` controls whether to include a diff PNG.
- SSIM uses 11x11 uniform windows with reflect padding.
- For images smaller than 11x11, SSIM falls back to global SSIM.
- No SSIM visualization output is produced in this crate.

## Build and test

```bash
# Rust tests
cargo test

# Python extension + tests (Python >3.11)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -U pip maturin pytest pillow
maturin develop
pytest

# Formatting / lint / typing
uv run --python 3.12 --with ruff ruff check .
uv run --python 3.12 --with ruff ruff format --check .
uv run --python 3.12 --with ty ty check . --python .venv

# Benchmarks
cargo bench
```

## License

This repository is MIT licensed. See [LICENSE](LICENSE).

Algorithm attribution for pixelmatch is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
