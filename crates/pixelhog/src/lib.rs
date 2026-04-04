pub mod image_utils;
pub mod pixelmatch;
pub mod ssim;

use image_utils::{decode_png_rgba, encode_png_rgba, pad_images_to_largest_cow};
use pixelmatch::{pixelmatch_count_rgba, pixelmatch_rgba};
use rayon::join;
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
