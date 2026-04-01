mod image_utils;
mod pixelmatch;
mod ssim;

use image_utils::{decode_png_rgba, encode_png_rgba, pad_images_to_largest_owned};
use pixelmatch::pixelmatch_rgba;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::join;
use ssim::compute_ssim_rgba;

pub use pixelmatch::{PixelmatchOptions, PixelmatchOutput};

pub fn pixelmatch_png(
    baseline_png: &[u8],
    current_png: &[u8],
    options: &PixelmatchOptions,
) -> Result<(Vec<u8>, usize, usize, usize), String> {
    let (baseline_decoded, current_decoded) = join(
        || decode_png_rgba(baseline_png),
        || decode_png_rgba(current_png),
    );
    let (baseline_rgba, baseline_width, baseline_height) = baseline_decoded?;
    let (current_rgba, current_width, current_height) = current_decoded?;

    let (baseline_padded, current_padded, width, height) = pad_images_to_largest_owned(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
    )?;

    let diff = pixelmatch_rgba(&baseline_padded, &current_padded, width, height, options)?;
    let diff_png = encode_png_rgba(&diff.diff_rgba, width, height)?;

    Ok((diff_png, diff.diff_count, width, height))
}

pub fn compute_ssim_png(baseline_png: &[u8], current_png: &[u8]) -> Result<f64, String> {
    let (baseline_decoded, current_decoded) = join(
        || decode_png_rgba(baseline_png),
        || decode_png_rgba(current_png),
    );
    let (baseline_rgba, baseline_width, baseline_height) = baseline_decoded?;
    let (current_rgba, current_width, current_height) = current_decoded?;

    let (baseline_padded, current_padded, width, height) = pad_images_to_largest_owned(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
    )?;

    compute_ssim_rgba(&baseline_padded, &current_padded, width, height)
}

#[pyfunction]
#[pyo3(name = "pixelmatch", signature = (
    baseline_png,
    current_png,
    threshold = 0.1,
    alpha = 0.1,
    include_aa = false,
    diff_color = (255, 0, 0),
    aa_color = (255, 255, 0),
    diff_color_alt = None
))]
fn pixelmatch_py(
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

    let options = PixelmatchOptions {
        threshold,
        alpha,
        include_aa,
        diff_color: [diff_color.0, diff_color.1, diff_color.2],
        aa_color: [aa_color.0, aa_color.1, aa_color.2],
        diff_color_alt: diff_color_alt.map(|c| [c.0, c.1, c.2]),
    };

    let (diff_png, diff_count, width, height) =
        pixelmatch_png(baseline_png, current_png, &options).map_err(PyValueError::new_err)?;

    let diff_bytes = PyBytes::new(py, &diff_png).into();

    Ok((diff_bytes, diff_count, width, height))
}

#[pyfunction]
fn compute_ssim(baseline_png: &[u8], current_png: &[u8]) -> PyResult<f64> {
    compute_ssim_png(baseline_png, current_png).map_err(PyValueError::new_err)
}

#[pymodule]
fn pixelhog(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pixelmatch_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ssim, m)?)?;
    Ok(())
}
