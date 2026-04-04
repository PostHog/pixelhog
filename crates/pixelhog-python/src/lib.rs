#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use ::pixelhog::{
    compare_png, compare_rgba, diff_count_png, diff_count_rgba, diff_png, diff_rgba, ssim_png,
    ssim_rgba, PixelmatchOptions,
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

    let (diff_png, diff_count, width, height) =
        diff_png(baseline_png, current_png, &options).map_err(to_py_err)?;

    let diff_bytes = PyBytes::new(py, &diff_png).into();
    Ok((diff_bytes, diff_count, width, height))
}

#[pyfunction]
#[pyo3(name = "diff_count", signature = (
    baseline_png,
    current_png,
    threshold = 0.1,
    include_aa = false,
))]
fn diff_count_py(
    baseline_png: &[u8],
    current_png: &[u8],
    threshold: f64,
    include_aa: bool,
) -> PyResult<(usize, usize, usize)> {
    let options = pixelmatch_count_options(threshold, include_aa)?;
    diff_count_png(baseline_png, current_png, &options).map_err(to_py_err)
}

#[pyfunction]
#[pyo3(name = "ssim")]
fn ssim_py(baseline_png: &[u8], current_png: &[u8]) -> PyResult<f64> {
    ssim_png(baseline_png, current_png).map_err(to_py_err)
}

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
) -> PyResult<(usize, f64, usize, usize, Option<Py<PyBytes>>)> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;

    let (diff_png, diff_count, ssim, width, height) =
        compare_png(baseline_png, current_png, &options, return_diff)
            .map_err(to_py_err)?;

    let diff_bytes = diff_png.map(|bytes| PyBytes::new(py, &bytes).into());
    Ok((diff_count, ssim, width, height, diff_bytes))
}

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

    let (diff_rgba, diff_count, width, height) = diff_rgba(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
        &options,
    )
    .map_err(to_py_err)?;

    let diff_bytes = PyBytes::new(py, &diff_rgba).into();
    Ok((diff_bytes, diff_count, width, height))
}

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
    diff_count_rgba(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
        &options,
    )
    .map_err(to_py_err)
}

#[pyfunction]
#[pyo3(name = "ssim_rgba")]
fn ssim_rgba_py(
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
) -> PyResult<f64> {
    ssim_rgba(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
    )
    .map_err(to_py_err)
}

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
) -> PyResult<(usize, f64, usize, usize, Option<Py<PyBytes>>)> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;

    let (diff_rgba, diff_count, ssim, width, height) = compare_rgba(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
        &options,
        return_diff,
    )
    .map_err(to_py_err)?;

    let diff_bytes = diff_rgba.map(|bytes| PyBytes::new(py, &bytes).into());
    Ok((diff_count, ssim, width, height, diff_bytes))
}

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

    let results: Result<Vec<_>, ::pixelhog::Error> = pairs
        .into_par_iter()
        .map(|(baseline_png, current_png)| diff_png(&baseline_png, &current_png, &options))
        .collect();

    let results = results.map_err(to_py_err)?;

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

#[pyfunction]
#[pyo3(name = "diff_count_batch", signature = (
    pairs,
    threshold = 0.1,
    include_aa = false,
))]
fn diff_count_batch_py(
    pairs: Vec<(Vec<u8>, Vec<u8>)>,
    threshold: f64,
    include_aa: bool,
) -> PyResult<Vec<(usize, usize, usize)>> {
    let options = pixelmatch_count_options(threshold, include_aa)?;

    let results: Result<Vec<_>, ::pixelhog::Error> = pairs
        .into_par_iter()
        .map(|(baseline_png, current_png)| diff_count_png(&baseline_png, &current_png, &options))
        .collect();

    results.map_err(to_py_err)
}

#[pyfunction]
#[pyo3(name = "ssim_batch")]
fn ssim_batch_py(pairs: Vec<(Vec<u8>, Vec<u8>)>) -> PyResult<Vec<f64>> {
    let results: Result<Vec<_>, ::pixelhog::Error> = pairs
        .into_par_iter()
        .map(|(baseline_png, current_png)| ssim_png(&baseline_png, &current_png))
        .collect();

    results.map_err(to_py_err)
}

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
) -> PyResult<Vec<(usize, f64, usize, usize, Option<Py<PyBytes>>)>> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;

    let results: Result<Vec<_>, ::pixelhog::Error> = pairs
        .into_par_iter()
        .map(|(baseline_png, current_png)| {
            compare_png(&baseline_png, &current_png, &options, return_diff)
        })
        .collect();

    let results = results.map_err(to_py_err)?;

    Ok(results
        .into_iter()
        .map(|(diff_png, diff_count, ssim, width, height)| {
            let diff_bytes = diff_png.map(|bytes| PyBytes::new(py, &bytes).into());
            (diff_count, ssim, width, height, diff_bytes)
        })
        .collect())
}

#[pymodule]
fn pixelhog(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

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
