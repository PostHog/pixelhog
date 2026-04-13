# pixelhog

Rust workspace with a Python binding, built with maturin/PyO3.

## Structure

- `crates/pixelhog/` — core Rust library (pixelmatch diff + SSIM)
- `crates/pixelhog-python/` — PyO3 binding crate (abi3, single wheel per platform)
- `python/tests/` — Python integration tests
- `pyproject.toml` — maturin build config, pytest config

## Build & test

```bash
# Rust
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test

# Python (builds extension, runs pytest)
uv run --python 3.12 --with maturin --with pytest --with pillow bash -lc \
  "maturin develop --release && pytest -q"
```

## Release

Tag `vX.Y.Z` on main triggers `.github/workflows/wheels.yml`:
builds cross-platform wheels → verifies tag matches pyproject.toml version →
attests artifacts → publishes to PyPI via OIDC trusted publishing →
creates GitHub Release with wheels attached.

The `release` GitHub environment has branch restrictions enabled.
Version must be updated in both `pyproject.toml` and `crates/*/Cargo.toml`.

## Conventions

- Place functions/methods before their call sites
- Prefer `pathlib.Path` over string paths in Python
- Prefer classes in Python unless they don't make sense
- Use `thiserror` for Rust error types
