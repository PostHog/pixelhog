//! Fast pixelmatch and SSIM image comparison.
//!
//! Compares images in two complementary ways:
//! - **Pixel diff** — exact pixel-level differences with anti-alias detection.
//! - **SSIM** — perceptual similarity score in `[0.0, 1.0]`.
//!
//! High-level functions accept PNG bytes and decode internally. When images
//! differ in size, the smaller one is padded with transparent pixels.

mod error;
pub mod image_utils;
pub mod pixelmatch;
pub mod ssim;

use image_utils::{decode_png_rgba, encode_png_rgba, pad_images_to_largest_cow};
use pixelmatch::{pixelmatch_count_rgba, pixelmatch_rgba};
use rayon::join;
use ssim::compute_ssim_rgba;

pub use error::Error;
pub use pixelmatch::{PixelmatchCountOutput, PixelmatchOptions, PixelmatchOutput};

/// `(diff_png_bytes, diff_count, width, height)`
pub type DiffPngOutput = (Vec<u8>, usize, usize, usize);
/// `(diff_rgba_bytes, diff_count, width, height)`
pub type DiffRgbaOutput = (Vec<u8>, usize, usize, usize);
/// `(diff_count, width, height)`
pub type DiffCountOutput = (usize, usize, usize);
/// `(optional_diff_png_bytes, diff_count, ssim, width, height)`
pub type ComparePngOutput = (Option<Vec<u8>>, usize, f64, usize, usize);
/// `(optional_diff_rgba_bytes, diff_count, ssim, width, height)`
pub type CompareRgbaOutput = (Option<Vec<u8>>, usize, f64, usize, usize);

type DecodedImage = (Vec<u8>, usize, usize);

fn decode_png_pair(
    baseline_png: &[u8],
    current_png: &[u8],
) -> Result<(DecodedImage, DecodedImage), Error> {
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
) -> Result<CompareRgbaOutput, Error> {
    let (pixel_result, ssim_result) = join(
        || {
            if return_diff {
                let diff = pixelmatch_rgba(baseline_rgba, current_rgba, width, height, options)?;
                Ok::<(Option<Vec<u8>>, usize), Error>((Some(diff.diff_rgba), diff.diff_count))
            } else {
                let diff =
                    pixelmatch_count_rgba(baseline_rgba, current_rgba, width, height, options)?;
                Ok::<(Option<Vec<u8>>, usize), Error>((None, diff.diff_count))
            }
        },
        || compute_ssim_rgba(baseline_rgba, current_rgba, width, height),
    );

    let (diff_rgba, diff_count) = pixel_result?;
    let ssim = ssim_result?;

    Ok((diff_rgba, diff_count, ssim, width, height))
}

/// Compute a pixel-level diff from two PNG images.
///
/// Decodes both PNGs, pads to matching dimensions, and produces a diff
/// image highlighting mismatched pixels. Returns the diff as PNG bytes.
pub fn diff_png(
    baseline_png: &[u8],
    current_png: &[u8],
    options: &PixelmatchOptions,
) -> Result<DiffPngOutput, Error> {
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

/// Count mismatched pixels between two PNG images without producing a diff image.
pub fn diff_count_png(
    baseline_png: &[u8],
    current_png: &[u8],
    options: &PixelmatchOptions,
) -> Result<DiffCountOutput, Error> {
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

/// Compute the SSIM (structural similarity) score between two PNG images.
///
/// Returns a value in `[0.0, 1.0]` where 1.0 means identical.
/// Uses 11×11 uniform windows with reflect padding; falls back to
/// global SSIM for images smaller than the window size.
pub fn ssim_png(baseline_png: &[u8], current_png: &[u8]) -> Result<f64, Error> {
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

/// Compute both pixel diff count and SSIM from two PNG images in a single call.
///
/// When `return_diff` is true, the diff image is also produced (as PNG bytes).
/// When false, diff image generation is skipped to save encoding time.
pub fn compare_png(
    baseline_png: &[u8],
    current_png: &[u8],
    options: &PixelmatchOptions,
    return_diff: bool,
) -> Result<ComparePngOutput, Error> {
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

/// Compute a pixel-level diff from pre-decoded RGBA buffers.
///
/// Images may differ in size — the smaller one is padded with transparent pixels.
/// Returns raw RGBA bytes of the diff image.
pub fn diff_rgba(
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
    options: &PixelmatchOptions,
) -> Result<DiffRgbaOutput, Error> {
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

/// Count mismatched pixels from pre-decoded RGBA buffers without producing a diff image.
pub fn diff_count_rgba(
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
    options: &PixelmatchOptions,
) -> Result<DiffCountOutput, Error> {
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

/// Compute SSIM from pre-decoded RGBA buffers.
pub fn ssim_rgba(
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
) -> Result<f64, Error> {
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

/// Compute both pixel diff count and SSIM from pre-decoded RGBA buffers.
#[allow(clippy::too_many_arguments)]
pub fn compare_rgba(
    baseline_rgba: &[u8],
    baseline_width: usize,
    baseline_height: usize,
    current_rgba: &[u8],
    current_width: usize,
    current_height: usize,
    options: &PixelmatchOptions,
    return_diff: bool,
) -> Result<CompareRgbaOutput, Error> {
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
