mod image_utils;
mod pixelmatch;
mod ssim;

use image_utils::{decode_png_rgba, encode_png_rgba, pad_images_to_largest_cow};
use pixelmatch::{pixelmatch_count_rgba, pixelmatch_rgba};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::join;
use rayon::prelude::*;
use ssim::compute_ssim_rgba;

pub use pixelmatch::{PixelmatchCountOutput, PixelmatchOptions, PixelmatchOutput};

pub type DiffPngOutput = (Vec<u8>, usize, usize, usize);
pub type DiffRgbaOutput = (Vec<u8>, usize, usize, usize);
pub type DiffCountOutput = (usize, usize, usize);
pub type ComparePngOutput = (Option<Vec<u8>>, usize, f64, usize, usize);
pub type CompareRgbaOutput = (Option<Vec<u8>>, usize, f64, usize, usize);

type DecodedImage = (Vec<u8>, usize, usize);

fn decode_png_pair(
    baseline_png: &[u8],
    current_png: &[u8],
) -> Result<(DecodedImage, DecodedImage), String> {
    let (baseline_decoded, current_decoded) = join(
        || decode_png_rgba(baseline_png),
        || decode_png_rgba(current_png),
    );
    Ok((baseline_decoded?, current_decoded?))
}

fn compare_rgba_padded(
    baseline_rgba: &[u8],
    current_rgba: &[u8],
    width: usize,
    height: usize,
    options: &PixelmatchOptions,
    return_diff: bool,
) -> Result<CompareRgbaOutput, String> {
    let (pixel_result, ssim_result) = join(
        || {
            if return_diff {
                let diff = pixelmatch_rgba(baseline_rgba, current_rgba, width, height, options)?;
                Ok::<(Option<Vec<u8>>, usize), String>((Some(diff.diff_rgba), diff.diff_count))
            } else {
                let diff =
                    pixelmatch_count_rgba(baseline_rgba, current_rgba, width, height, options)?;
                Ok::<(Option<Vec<u8>>, usize), String>((None, diff.diff_count))
            }
        },
        || compute_ssim_rgba(baseline_rgba, current_rgba, width, height),
    );

    let (diff_rgba, diff_count) = pixel_result?;
    let ssim = ssim_result?;

    Ok((diff_rgba, diff_count, ssim, width, height))
}

pub fn diff_png(
    baseline_png: &[u8],
    current_png: &[u8],
    options: &PixelmatchOptions,
) -> Result<DiffPngOutput, String> {
    let (
        (baseline_rgba, baseline_width, baseline_height),
        (current_rgba, current_width, current_height),
    ) = decode_png_pair(baseline_png, current_png)?;

    let (diff_rgba, diff_count, width, height) = diff_rgba(
        &baseline_rgba,
        baseline_width,
        baseline_height,
        &current_rgba,
        current_width,
        current_height,
        options,
    )?;

    let diff_png = encode_png_rgba(&diff_rgba, width, height)?;
    Ok((diff_png, diff_count, width, height))
}

pub fn diff_count_png(
    baseline_png: &[u8],
    current_png: &[u8],
    options: &PixelmatchOptions,
) -> Result<DiffCountOutput, String> {
    let (
        (baseline_rgba, baseline_width, baseline_height),
        (current_rgba, current_width, current_height),
    ) = decode_png_pair(baseline_png, current_png)?;

    diff_count_rgba(
        &baseline_rgba,
        baseline_width,
        baseline_height,
        &current_rgba,
        current_width,
        current_height,
        options,
    )
}

pub fn ssim_png(baseline_png: &[u8], current_png: &[u8]) -> Result<f64, String> {
    let (
        (baseline_rgba, baseline_width, baseline_height),
        (current_rgba, current_width, current_height),
    ) = decode_png_pair(baseline_png, current_png)?;

    ssim_rgba(
        &baseline_rgba,
        baseline_width,
        baseline_height,
        &current_rgba,
        current_width,
        current_height,
    )
}

pub fn compare_png(
    baseline_png: &[u8],
    current_png: &[u8],
    options: &PixelmatchOptions,
    return_diff: bool,
) -> Result<ComparePngOutput, String> {
    let (
        (baseline_rgba, baseline_width, baseline_height),
        (current_rgba, current_width, current_height),
    ) = decode_png_pair(baseline_png, current_png)?;

    let (diff_rgba, diff_count, ssim, width, height) = compare_rgba(
        &baseline_rgba,
        baseline_width,
        baseline_height,
        &current_rgba,
        current_width,
        current_height,
        options,
        return_diff,
    )?;

    let diff_png = match diff_rgba {
        Some(diff_rgba) => Some(encode_png_rgba(&diff_rgba, width, height)?),
        None => None,
    };

    Ok((diff_png, diff_count, ssim, width, height))
}

pub fn diff_rgba(
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
    options: &PixelmatchOptions,
) -> Result<DiffRgbaOutput, String> {
    let (baseline_padded, current_padded, width, height) = pad_images_to_largest_cow(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
    )?;

    let diff = pixelmatch_rgba(
        baseline_padded.as_ref(),
        current_padded.as_ref(),
        width,
        height,
        options,
    )?;

    Ok((diff.diff_rgba, diff.diff_count, width, height))
}

pub fn diff_count_rgba(
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
    options: &PixelmatchOptions,
) -> Result<DiffCountOutput, String> {
    let (baseline_padded, current_padded, width, height) = pad_images_to_largest_cow(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
    )?;

    let diff = pixelmatch_count_rgba(
        baseline_padded.as_ref(),
        current_padded.as_ref(),
        width,
        height,
        options,
    )?;

    Ok((diff.diff_count, width, height))
}

pub fn ssim_rgba(
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
) -> Result<f64, String> {
    let (baseline_padded, current_padded, width, height) = pad_images_to_largest_cow(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
    )?;

    compute_ssim_rgba(
        baseline_padded.as_ref(),
        current_padded.as_ref(),
        width,
        height,
    )
}

pub fn compare_rgba(
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
    options: &PixelmatchOptions,
    return_diff: bool,
) -> Result<CompareRgbaOutput, String> {
    let (baseline_padded, current_padded, width, height) = pad_images_to_largest_cow(
        baseline_rgba,
        baseline_width,
        baseline_height,
        current_rgba,
        current_width,
        current_height,
    )?;

    compare_rgba_padded(
        baseline_padded.as_ref(),
        current_padded.as_ref(),
        width,
        height,
        options,
        return_diff,
    )
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
        diff_png(baseline_png, current_png, &options).map_err(PyValueError::new_err)?;

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
    diff_count_png(baseline_png, current_png, &options).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(name = "ssim")]
fn ssim_py(baseline_png: &[u8], current_png: &[u8]) -> PyResult<f64> {
    ssim_png(baseline_png, current_png).map_err(PyValueError::new_err)
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
            .map_err(PyValueError::new_err)?;

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
    .map_err(PyValueError::new_err)?;

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
    .map_err(PyValueError::new_err)
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
    .map_err(PyValueError::new_err)
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
    .map_err(PyValueError::new_err)?;

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

    let results: Result<Vec<_>, String> = pairs
        .into_par_iter()
        .map(|(baseline_png, current_png)| diff_png(&baseline_png, &current_png, &options))
        .collect();

    let results = results.map_err(PyValueError::new_err)?;

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

    let results: Result<Vec<_>, String> = pairs
        .into_par_iter()
        .map(|(baseline_png, current_png)| diff_count_png(&baseline_png, &current_png, &options))
        .collect();

    results.map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(name = "ssim_batch")]
fn ssim_batch_py(pairs: Vec<(Vec<u8>, Vec<u8>)>) -> PyResult<Vec<f64>> {
    let results: Result<Vec<_>, String> = pairs
        .into_par_iter()
        .map(|(baseline_png, current_png)| ssim_png(&baseline_png, &current_png))
        .collect();

    results.map_err(PyValueError::new_err)
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

    let results: Result<Vec<_>, String> = pairs
        .into_par_iter()
        .map(|(baseline_png, current_png)| {
            compare_png(&baseline_png, &current_png, &options, return_diff)
        })
        .collect();

    let results = results.map_err(PyValueError::new_err)?;

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
