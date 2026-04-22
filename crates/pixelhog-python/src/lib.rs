#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use ::pixelhog::{
    compare_png, compare_rgba, create_thumbnail, diff_count_png, diff_count_rgba, diff_png,
    diff_rgba, ssim_png, ssim_rgba, PixelmatchOptions, ThumbnailOptions,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

fn to_py_err(e: ::pixelhog::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn pixelmatch_options(
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
) -> PyResult<PixelmatchOptions> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(PyValueError::new_err(
            "threshold must be in the range [0.0, 1.0]",
        ));
    }

    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(
            "alpha must be in the range [0.0, 1.0]",
        ));
    }

    Ok(PixelmatchOptions {
        threshold,
        alpha,
        include_aa,
        diff_color: [diff_color.0, diff_color.1, diff_color.2],
        aa_color: [aa_color.0, aa_color.1, aa_color.2],
        diff_color_alt: diff_color_alt.map(|c| [c.0, c.1, c.2]),
    })
}

fn pixelmatch_count_options(threshold: f64, include_aa: bool) -> PyResult<PixelmatchOptions> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(PyValueError::new_err(
            "threshold must be in the range [0.0, 1.0]",
        ));
    }

    Ok(PixelmatchOptions {
        threshold,
        include_aa,
        ..PixelmatchOptions::default()
    })
}

fn thumbnail_options(
    thumbnail_width: Option<usize>,
    thumbnail_height: Option<usize>,
) -> Option<ThumbnailOptions> {
    thumbnail_width.map(|max_width| ThumbnailOptions {
        max_width,
        max_height: thumbnail_height,
    })
}

/// Create a lossless WebP thumbnail from a PNG image.
///
/// Scales down to `width`, preserving aspect ratio. If `height` is set,
/// crops from the top after resizing — useful for fixed-size grid cells.
/// Images already within bounds are re-encoded without resizing.
#[pyfunction]
#[pyo3(name = "thumbnail", signature = (png_bytes, width = 200, height = None))]
fn thumbnail_py(
    py: Python<'_>,
    png_bytes: &[u8],
    width: usize,
    height: Option<usize>,
) -> PyResult<Py<PyBytes>> {
    let result = py
        .allow_threads(|| create_thumbnail(png_bytes, width, height))
        .map_err(to_py_err)?;
    Ok(PyBytes::new(py, &result).into())
}

/// Compute a pixel-level diff between two PNG images.
///
/// The diff image highlights mismatched pixels in `diff_color` and
/// anti-aliased pixels in `aa_color`.
#[pyfunction]
#[pyo3(name = "diff", signature = (
    baseline_png,
    current_png,
    threshold = 0.1,
    alpha = 0.1,
    include_aa = false,
    diff_color = (255, 0, 0),
    aa_color = (255, 255, 0),
    diff_color_alt = None,
))]
fn diff_py(
    py: Python<'_>,
    baseline_png: &[u8],
    current_png: &[u8],
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
) -> PyResult<(Py<PyBytes>, usize, usize, usize)> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;

    let (diff_png, diff_count, width, height) = py
        .allow_threads(|| diff_png(baseline_png, current_png, &options))
        .map_err(to_py_err)?;

    let diff_bytes = PyBytes::new(py, &diff_png).into();
    Ok((diff_bytes, diff_count, width, height))
}

/// Count mismatched pixels between two PNG images without producing a diff image.
///
/// Faster than `diff()` when you only need the count.
#[pyfunction]
#[pyo3(name = "diff_count", signature = (
    baseline_png,
    current_png,
    threshold = 0.1,
    include_aa = false,
))]
fn diff_count_py(
    py: Python<'_>,
    baseline_png: &[u8],
    current_png: &[u8],
    threshold: f64,
    include_aa: bool,
) -> PyResult<(usize, usize, usize)> {
    let options = pixelmatch_count_options(threshold, include_aa)?;
    py.allow_threads(|| diff_count_png(baseline_png, current_png, &options))
        .map_err(to_py_err)
}

/// Compute SSIM (structural similarity) between two PNG images.
///
/// Score in [0.0, 1.0] where 1.0 means identical.
#[pyfunction]
#[pyo3(name = "ssim")]
fn ssim_py(py: Python<'_>, baseline_png: &[u8], current_png: &[u8]) -> PyResult<f64> {
    py.allow_threads(|| ssim_png(baseline_png, current_png))
        .map_err(to_py_err)
}

/// Compute pixel diff count and SSIM in a single call (one decode pass).
///
/// Set `return_diff=True` to include the diff image. Set `thumbnail_width`
/// to generate a WebP thumbnail of the current image from the already-decoded buffer.
#[pyfunction]
#[pyo3(name = "compare", signature = (
    baseline_png,
    current_png,
    threshold = 0.1,
    alpha = 0.1,
    include_aa = false,
    diff_color = (255, 0, 0),
    aa_color = (255, 255, 0),
    diff_color_alt = None,
    return_diff = false,
    thumbnail_width = None,
    thumbnail_height = None,
))]
fn compare_py(
    py: Python<'_>,
    baseline_png: &[u8],
    current_png: &[u8],
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
    return_diff: bool,
    thumbnail_width: Option<usize>,
    thumbnail_height: Option<usize>,
) -> PyResult<(
    usize,
    f64,
    usize,
    usize,
    Option<Py<PyBytes>>,
    Option<Py<PyBytes>>,
)> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;
    let thumb = thumbnail_options(thumbnail_width, thumbnail_height);

    let (diff_png, diff_count, ssim, width, height, thumb_webp) = py
        .allow_threads(|| {
            compare_png(
                baseline_png,
                current_png,
                &options,
                return_diff,
                thumb.as_ref(),
            )
        })
        .map_err(to_py_err)?;

    let diff_bytes = diff_png.map(|bytes| PyBytes::new(py, &bytes).into());
    let thumb_bytes = thumb_webp.map(|bytes| PyBytes::new(py, &bytes).into());
    Ok((diff_count, ssim, width, height, diff_bytes, thumb_bytes))
}

/// Compute a pixel-level diff from pre-decoded RGBA buffers.
///
/// Same as `diff()` but accepts raw RGBA bytes instead of PNG.
#[pyfunction]
#[pyo3(name = "diff_rgba", signature = (
    baseline_rgba,
    baseline_width,
    baseline_height,
    current_rgba,
    current_width,
    current_height,
    threshold = 0.1,
    alpha = 0.1,
    include_aa = false,
    diff_color = (255, 0, 0),
    aa_color = (255, 255, 0),
    diff_color_alt = None,
))]
fn diff_rgba_py(
    py: Python<'_>,
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
) -> PyResult<(Py<PyBytes>, usize, usize, usize)> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;

    let (diff_rgba, diff_count, width, height) = py
        .allow_threads(|| {
            diff_rgba(
                baseline_rgba,
                baseline_width,
                baseline_height,
                current_rgba,
                current_width,
                current_height,
                &options,
            )
        })
        .map_err(to_py_err)?;

    let diff_bytes = PyBytes::new(py, &diff_rgba).into();
    Ok((diff_bytes, diff_count, width, height))
}

/// Count mismatched pixels from pre-decoded RGBA buffers.
///
/// Same as `diff_count()` but accepts raw RGBA bytes instead of PNG.
#[pyfunction]
#[pyo3(name = "diff_count_rgba", signature = (
    baseline_rgba,
    baseline_width,
    baseline_height,
    current_rgba,
    current_width,
    current_height,
    threshold = 0.1,
    include_aa = false,
))]
fn diff_count_rgba_py(
    py: Python<'_>,
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
    threshold: f64,
    include_aa: bool,
) -> PyResult<(usize, usize, usize)> {
    let options = pixelmatch_count_options(threshold, include_aa)?;
    py.allow_threads(|| {
        diff_count_rgba(
            baseline_rgba,
            baseline_width,
            baseline_height,
            current_rgba,
            current_width,
            current_height,
            &options,
        )
    })
    .map_err(to_py_err)
}

/// Compute SSIM from pre-decoded RGBA buffers.
///
/// Same as `ssim()` but accepts raw RGBA bytes instead of PNG.
#[pyfunction]
#[pyo3(name = "ssim_rgba")]
fn ssim_rgba_py(
    py: Python<'_>,
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
) -> PyResult<f64> {
    py.allow_threads(|| {
        ssim_rgba(
            baseline_rgba,
            baseline_width,
            baseline_height,
            current_rgba,
            current_width,
            current_height,
        )
    })
    .map_err(to_py_err)
}

/// Compute pixel diff count and SSIM from pre-decoded RGBA buffers.
///
/// Same as `compare()` but accepts raw RGBA bytes instead of PNG.
#[pyfunction]
#[pyo3(name = "compare_rgba", signature = (
    baseline_rgba,
    baseline_width,
    baseline_height,
    current_rgba,
    current_width,
    current_height,
    threshold = 0.1,
    alpha = 0.1,
    include_aa = false,
    diff_color = (255, 0, 0),
    aa_color = (255, 255, 0),
    diff_color_alt = None,
    return_diff = false,
    thumbnail_width = None,
    thumbnail_height = None,
))]
fn compare_rgba_py(
    py: Python<'_>,
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
    return_diff: bool,
    thumbnail_width: Option<usize>,
    thumbnail_height: Option<usize>,
) -> PyResult<(
    usize,
    f64,
    usize,
    usize,
    Option<Py<PyBytes>>,
    Option<Py<PyBytes>>,
)> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;
    let thumb = thumbnail_options(thumbnail_width, thumbnail_height);

    let (diff_rgba, diff_count, ssim, width, height, thumb_webp) = py
        .allow_threads(|| {
            compare_rgba(
                baseline_rgba,
                baseline_width,
                baseline_height,
                current_rgba,
                current_width,
                current_height,
                &options,
                return_diff,
                thumb.as_ref(),
            )
        })
        .map_err(to_py_err)?;

    let diff_bytes = diff_rgba.map(|bytes| PyBytes::new(py, &bytes).into());
    let thumb_bytes = thumb_webp.map(|bytes| PyBytes::new(py, &bytes).into());
    Ok((diff_count, ssim, width, height, diff_bytes, thumb_bytes))
}

/// Diff multiple image pairs in parallel.
#[pyfunction]
#[pyo3(name = "diff_batch", signature = (
    pairs,
    threshold = 0.1,
    alpha = 0.1,
    include_aa = false,
    diff_color = (255, 0, 0),
    aa_color = (255, 255, 0),
    diff_color_alt = None,
))]
fn diff_batch_py(
    py: Python<'_>,
    pairs: Vec<(Vec<u8>, Vec<u8>)>,
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
) -> PyResult<Vec<(Py<PyBytes>, usize, usize, usize)>> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;

    let results = py
        .allow_threads(|| {
            let r: Result<Vec<_>, ::pixelhog::Error> = pairs
                .into_par_iter()
                .map(|(baseline_png, current_png)| diff_png(&baseline_png, &current_png, &options))
                .collect();
            r
        })
        .map_err(to_py_err)?;

    Ok(results
        .into_iter()
        .map(|(diff_png, diff_count, width, height)| {
            (
                PyBytes::new(py, &diff_png).into(),
                diff_count,
                width,
                height,
            )
        })
        .collect())
}

/// Count mismatched pixels for multiple image pairs in parallel.
#[pyfunction]
#[pyo3(name = "diff_count_batch", signature = (
    pairs,
    threshold = 0.1,
    include_aa = false,
))]
fn diff_count_batch_py(
    py: Python<'_>,
    pairs: Vec<(Vec<u8>, Vec<u8>)>,
    threshold: f64,
    include_aa: bool,
) -> PyResult<Vec<(usize, usize, usize)>> {
    let options = pixelmatch_count_options(threshold, include_aa)?;

    py.allow_threads(|| {
        let r: Result<Vec<_>, ::pixelhog::Error> = pairs
            .into_par_iter()
            .map(|(baseline_png, current_png)| {
                diff_count_png(&baseline_png, &current_png, &options)
            })
            .collect();
        r
    })
    .map_err(to_py_err)
}

/// Compute SSIM for multiple image pairs in parallel.
#[pyfunction]
#[pyo3(name = "ssim_batch")]
fn ssim_batch_py(py: Python<'_>, pairs: Vec<(Vec<u8>, Vec<u8>)>) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        let r: Result<Vec<_>, ::pixelhog::Error> = pairs
            .into_par_iter()
            .map(|(baseline_png, current_png)| ssim_png(&baseline_png, &current_png))
            .collect();
        r
    })
    .map_err(to_py_err)
}

/// Compare multiple image pairs in parallel (diff count + SSIM per pair).
///
/// Batch version of `compare()`. Set `thumbnail_width` to generate a
/// thumbnail per pair.
#[pyfunction]
#[pyo3(name = "compare_batch", signature = (
    pairs,
    threshold = 0.1,
    alpha = 0.1,
    include_aa = false,
    diff_color = (255, 0, 0),
    aa_color = (255, 255, 0),
    diff_color_alt = None,
    return_diff = false,
    thumbnail_width = None,
    thumbnail_height = None,
))]
fn compare_batch_py(
    py: Python<'_>,
    pairs: Vec<(Vec<u8>, Vec<u8>)>,
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
    return_diff: bool,
    thumbnail_width: Option<usize>,
    thumbnail_height: Option<usize>,
) -> PyResult<
    Vec<(
        usize,
        f64,
        usize,
        usize,
        Option<Py<PyBytes>>,
        Option<Py<PyBytes>>,
    )>,
> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;
    let thumb = thumbnail_options(thumbnail_width, thumbnail_height);

    let results = py
        .allow_threads(|| {
            let r: Result<Vec<_>, ::pixelhog::Error> = pairs
                .into_par_iter()
                .map(|(baseline_png, current_png)| {
                    compare_png(
                        &baseline_png,
                        &current_png,
                        &options,
                        return_diff,
                        thumb.as_ref(),
                    )
                })
                .collect();
            r
        })
        .map_err(to_py_err)?;

    Ok(results
        .into_iter()
        .map(|(diff_png, diff_count, ssim, width, height, thumb_webp)| {
            let diff_bytes = diff_png.map(|bytes| PyBytes::new(py, &bytes).into());
            let thumb_bytes = thumb_webp.map(|bytes| PyBytes::new(py, &bytes).into());
            (diff_count, ssim, width, height, diff_bytes, thumb_bytes)
        })
        .collect())
}

#[pymodule]
fn pixelhog(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(thumbnail_py, m)?)?;
    m.add_function(wrap_pyfunction!(diff_py, m)?)?;
    m.add_function(wrap_pyfunction!(diff_count_py, m)?)?;
    m.add_function(wrap_pyfunction!(ssim_py, m)?)?;
    m.add_function(wrap_pyfunction!(compare_py, m)?)?;

    m.add_function(wrap_pyfunction!(diff_rgba_py, m)?)?;
    m.add_function(wrap_pyfunction!(diff_count_rgba_py, m)?)?;
    m.add_function(wrap_pyfunction!(ssim_rgba_py, m)?)?;
    m.add_function(wrap_pyfunction!(compare_rgba_py, m)?)?;

    m.add_function(wrap_pyfunction!(diff_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(diff_count_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(ssim_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(compare_batch_py, m)?)?;

    Ok(())
}
