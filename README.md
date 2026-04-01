# pixelhog

Rust-powered pixel-by-pixel diffing and SSIM scoring for PNG bytes, exposed to Python via PyO3.

## Python API

```python
from pixelhog import pixelmatch, compute_ssim

# Pixel-by-pixel diff
# Returns: (diff_png_bytes, diff_count, width, height)
diff_png, diff_count, width, height = pixelmatch(
    baseline_png,
    current_png,
    threshold=0.1,
    alpha=0.1,
    include_aa=False,
    diff_color=(255, 0, 0),
    aa_color=(255, 255, 0),
    diff_color_alt=None,  # optional alternative color for darker-vs-lighter diffs
)

# SSIM score from 0.0 to 1.0
score = compute_ssim(baseline_png, current_png)
```

## Behavior

- Accepts PNG bytes only.
- Pads smaller image to larger dimensions with transparent pixels.
- `pixelmatch` always returns a diff PNG image.
- For identical images, diff output is faded grayscale using `alpha`.
- SSIM uses 11x11 uniform windows with reflect padding.
- For images smaller than 11x11, SSIM falls back to global SSIM.

## Build and test

```bash
# Rust tests
cargo test

# Python extension + tests
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -U pip maturin pytest pillow
maturin develop
pytest

# Benchmarks
cargo bench
```

## License

This repository is MIT licensed. See [LICENSE](LICENSE).

Algorithm attribution for pixelmatch is documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
