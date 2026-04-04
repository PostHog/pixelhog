use crate::image_utils::rgba_to_grayscale_f64;

const C1: f64 = 6.5025;
const C2: f64 = 58.5225;
const WIN_SIZE: usize = 11;

/// Compute SSIM on equal-sized RGBA buffers.
///
/// Converts to grayscale internally. Falls back to global SSIM
/// when either dimension is smaller than the 11×11 window.
pub fn compute_ssim_rgba(
    baseline: &[u8],
    current: &[u8],
    width: usize,
    height: usize,
) -> Result<f64, String> {
    let expected_len = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| "image dimensions overflowed".to_string())?;

    if baseline.len() != expected_len || current.len() != expected_len {
        return Err("invalid RGBA inputs for SSIM".to_string());
    }

    let gray1 = rgba_to_grayscale_f64(baseline);
    let gray2 = rgba_to_grayscale_f64(current);

    compute_ssim_grayscale(&gray1, &gray2, width, height)
}

/// Compute SSIM on equal-sized grayscale buffers (one `f64` per pixel).
pub fn compute_ssim_grayscale(
    baseline: &[f64],
    current: &[f64],
    width: usize,
    height: usize,
) -> Result<f64, String> {
    let expected_len = width
        .checked_mul(height)
        .ok_or_else(|| "image dimensions overflowed".to_string())?;

    if baseline.len() != expected_len || current.len() != expected_len {
        return Err("invalid grayscale inputs for SSIM".to_string());
    }

    if expected_len == 0 {
        return Ok(1.0);
    }

    if width < WIN_SIZE || height < WIN_SIZE {
        return Ok(clamp_unit(global_ssim(baseline, current)));
    }

    let ((mu1, mu2), (sigma1_src, sigma2_src, sigma12_src)) = rayon::join(
        || {
            rayon::join(
                || box_filter_reflect(baseline, width, height, WIN_SIZE),
                || box_filter_reflect(current, width, height, WIN_SIZE),
            )
        },
        || {
            let mut baseline_sq = vec![0.0; expected_len];
            let mut current_sq = vec![0.0; expected_len];
            let mut baseline_current = vec![0.0; expected_len];

            for i in 0..expected_len {
                let a = baseline[i];
                let b = current[i];
                baseline_sq[i] = a * a;
                current_sq[i] = b * b;
                baseline_current[i] = a * b;
            }

            let (sigma1_src, (sigma2_src, sigma12_src)) = rayon::join(
                || box_filter_reflect(&baseline_sq, width, height, WIN_SIZE),
                || {
                    rayon::join(
                        || box_filter_reflect(&current_sq, width, height, WIN_SIZE),
                        || box_filter_reflect(&baseline_current, width, height, WIN_SIZE),
                    )
                },
            );

            (sigma1_src, sigma2_src, sigma12_src)
        },
    );

    let mut sum = 0.0;

    for i in 0..expected_len {
        let mu1_sq = mu1[i] * mu1[i];
        let mu2_sq = mu2[i] * mu2[i];
        let mu1_mu2 = mu1[i] * mu2[i];

        let sigma1_sq = sigma1_src[i] - mu1_sq;
        let sigma2_sq = sigma2_src[i] - mu2_sq;
        let sigma12 = sigma12_src[i] - mu1_mu2;

        let numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2);
        let denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2);

        let local = if denominator == 0.0 {
            1.0
        } else {
            numerator / denominator
        };

        sum += local;
    }

    Ok(clamp_unit(sum / expected_len as f64))
}

fn global_ssim(baseline: &[f64], current: &[f64]) -> f64 {
    let n = baseline.len() as f64;

    let mu1 = baseline.iter().sum::<f64>() / n;
    let mu2 = current.iter().sum::<f64>() / n;

    let mut sigma1_sq = 0.0;
    let mut sigma2_sq = 0.0;
    let mut sigma12 = 0.0;

    for (&a, &b) in baseline.iter().zip(current.iter()) {
        let da = a - mu1;
        let db = b - mu2;
        sigma1_sq += da * da;
        sigma2_sq += db * db;
        sigma12 += da * db;
    }

    sigma1_sq /= n;
    sigma2_sq /= n;
    sigma12 /= n;

    let numerator = (2.0 * mu1 * mu2 + C1) * (2.0 * sigma12 + C2);
    let denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2);

    if denominator == 0.0 {
        1.0
    } else {
        numerator / denominator
    }
}

fn box_filter_reflect(src: &[f64], width: usize, height: usize, size: usize) -> Vec<f64> {
    let pad = size / 2;
    let padded_width = width + 2 * pad;
    let padded_height = height + 2 * pad;

    let mut padded = vec![0.0; padded_width * padded_height];

    for py in 0..padded_height {
        let sy = reflect_index(py as isize - pad as isize, height);
        for px in 0..padded_width {
            let sx = reflect_index(px as isize - pad as isize, width);
            padded[py * padded_width + px] = src[sy * width + sx];
        }
    }

    let int_width = padded_width + 1;
    let int_height = padded_height + 1;
    let mut integral = vec![0.0; int_width * int_height];

    for y in 0..padded_height {
        let mut row_sum = 0.0;
        for x in 0..padded_width {
            row_sum += padded[y * padded_width + x];
            let idx = (y + 1) * int_width + (x + 1);
            integral[idx] = integral[y * int_width + (x + 1)] + row_sum;
        }
    }

    let area = (size * size) as f64;
    let mut out = vec![0.0; width * height];

    for y in 0..height {
        let top = y;
        let bottom = y + size;
        for x in 0..width {
            let left = x;
            let right = x + size;

            let sum = integral[bottom * int_width + right]
                - integral[top * int_width + right]
                - integral[bottom * int_width + left]
                + integral[top * int_width + left];

            out[y * width + x] = sum / area;
        }
    }

    out
}

fn reflect_index(mut i: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }

    let n = len as isize;
    while i < 0 || i >= n {
        if i < 0 {
            i = -i;
        } else {
            i = 2 * n - i - 2;
        }
    }

    i as usize
}

fn clamp_unit(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}
