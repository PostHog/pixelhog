use crate::image_utils::validate_rgba_len;
use crate::Error;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;
const GOLDEN_RATIO_PLUS_ONE: f64 = 2.618_033_988_749_895;
const MAX_YIQ_DELTA: f64 = 35215.0;
const PARALLEL_MIN_PIXELS: usize = 262_144;
const PARALLEL_ROW_BLOCK: usize = 64;

/// Configuration for pixel-level image comparison.
#[derive(Debug, Clone)]
pub struct PixelmatchOptions {
    /// Color distance threshold in `[0.0, 1.0]`. Lower values catch subtler differences.
    pub threshold: f64,
    /// Blending factor for unchanged pixels in the diff image (`[0.0, 1.0]`).
    pub alpha: f64,
    /// When false (default), anti-aliased pixels are excluded from the diff count.
    pub include_aa: bool,
    /// RGB color used to mark differing pixels (default: red).
    pub diff_color: [u8; 3],
    /// RGB color used to mark anti-aliased pixels (default: yellow).
    pub aa_color: [u8; 3],
    /// Optional alternate diff color for pixels where image2 is darker than image1.
    pub diff_color_alt: Option<[u8; 3]>,
}

impl Default for PixelmatchOptions {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            alpha: 0.1,
            include_aa: false,
            diff_color: [255, 0, 0],
            aa_color: [255, 255, 0],
            diff_color_alt: None,
        }
    }
}

/// Result of a pixel diff that includes the diff image.
#[derive(Debug, Clone)]
pub struct PixelmatchOutput {
    /// Raw RGBA bytes of the diff visualization.
    pub diff_rgba: Vec<u8>,
    /// Number of pixels that differ beyond the threshold.
    pub diff_count: usize,
    pub width: usize,
    pub height: usize,
}

/// Result of a count-only pixel diff (no diff image produced).
#[derive(Debug, Clone, Copy)]
pub struct PixelmatchCountOutput {
    /// Number of pixels that differ beyond the threshold.
    pub diff_count: usize,
    pub width: usize,
    pub height: usize,
}

/// Core pixel diff on equal-sized RGBA buffers, producing a diff image.
///
/// Both buffers must have length `width * height * 4`. Use the higher-level
/// [`diff_rgba`](crate::diff_rgba) or [`diff_png`](crate::diff_png) functions
/// for automatic padding and PNG decoding.
pub fn pixelmatch_rgba(
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    options: &PixelmatchOptions,
) -> Result<PixelmatchOutput, Error> {
    validate_options(options)?;
    validate_rgba_len(img1.len(), width, height)?;
    validate_rgba_len(img2.len(), width, height)?;

    let len = width.checked_mul(height).ok_or(Error::Overflow)?;

    let mut output = vec![0u8; img1.len()];
    let alpha = options.alpha;
    let include_aa = options.include_aa;
    let diff_color = options.diff_color;
    let aa_color = options.aa_color;
    let diff_color_alt = options.diff_color_alt;

    // Fast path: identical images.
    if img1 == img2 {
        for i in 0..len {
            draw_gray_pixel(img1, i * 4, alpha, &mut output);
        }

        return Ok(PixelmatchOutput {
            diff_rgba: output,
            diff_count: 0,
            width,
            height,
        });
    }

    let max_delta = MAX_YIQ_DELTA * options.threshold * options.threshold;
    let a32_buf = rgba_as_u32_slice(img1);
    let b32_buf = rgba_as_u32_slice(img2);
    let a32 = a32_buf.as_slice();
    let b32 = b32_buf.as_slice();

    let row_stride = width * 4;
    let diff_count = if len >= PARALLEL_MIN_PIXELS {
        let block_stride = row_stride * PARALLEL_ROW_BLOCK;
        output
            .par_chunks_mut(block_stride)
            .enumerate()
            .map(|(block_idx, block_out)| {
                let mut block_diff = 0usize;
                let start_row = block_idx * PARALLEL_ROW_BLOCK;
                let rows = block_out.len() / row_stride;

                for row_offset in 0..rows {
                    let y = start_row + row_offset;
                    let row_start = row_offset * row_stride;
                    let row_end = row_start + row_stride;

                    block_diff += process_row(
                        y,
                        &mut block_out[row_start..row_end],
                        img1,
                        img2,
                        width,
                        height,
                        a32,
                        b32,
                        max_delta,
                        alpha,
                        include_aa,
                        diff_color,
                        aa_color,
                        diff_color_alt,
                    );
                }

                block_diff
            })
            .sum()
    } else {
        let mut diff = 0usize;
        for y in 0..height {
            let row_start = y * row_stride;
            let row_end = row_start + row_stride;
            diff += process_row(
                y,
                &mut output[row_start..row_end],
                img1,
                img2,
                width,
                height,
                a32,
                b32,
                max_delta,
                alpha,
                include_aa,
                diff_color,
                aa_color,
                diff_color_alt,
            );
        }
        diff
    };

    Ok(PixelmatchOutput {
        diff_rgba: output,
        diff_count,
        width,
        height,
    })
}

/// Core count-only pixel diff on equal-sized RGBA buffers (no diff image).
pub fn pixelmatch_count_rgba(
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    options: &PixelmatchOptions,
) -> Result<PixelmatchCountOutput, Error> {
    pixelmatch_count_rgba_inner(img1, img2, width, height, options, None)
}

/// Count-only pixel diff with early exit once `max_diffs` is reached.
///
/// Returns as soon as the diff count meets or exceeds `max_diffs`.
/// The returned count may slightly exceed `max_diffs` due to parallel
/// processing granularity.
pub fn pixelmatch_count_rgba_capped(
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    options: &PixelmatchOptions,
    max_diffs: usize,
) -> Result<PixelmatchCountOutput, Error> {
    if max_diffs == 0 {
        return Ok(PixelmatchCountOutput {
            diff_count: 0,
            width,
            height,
        });
    }
    pixelmatch_count_rgba_inner(img1, img2, width, height, options, Some(max_diffs))
}

fn pixelmatch_count_rgba_inner(
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    options: &PixelmatchOptions,
    max_diffs: Option<usize>,
) -> Result<PixelmatchCountOutput, Error> {
    validate_options(options)?;
    validate_rgba_len(img1.len(), width, height)?;
    validate_rgba_len(img2.len(), width, height)?;

    let len = width.checked_mul(height).ok_or(Error::Overflow)?;

    // Fast path: identical images.
    if img1 == img2 {
        return Ok(PixelmatchCountOutput {
            diff_count: 0,
            width,
            height,
        });
    }

    let max_delta = MAX_YIQ_DELTA * options.threshold * options.threshold;
    let a32_buf = rgba_as_u32_slice(img1);
    let b32_buf = rgba_as_u32_slice(img2);
    let a32 = a32_buf.as_slice();
    let b32 = b32_buf.as_slice();
    let include_aa = options.include_aa;

    // Threshold=0 + no-AA fast path: any pixel with different u32 value is a diff.
    // Skips color_delta entirely — just count mismatches.
    if max_delta == 0.0 && !include_aa {
        let diff_count = count_u32_mismatches(a32, b32, len, max_diffs);
        return Ok(PixelmatchCountOutput {
            diff_count,
            width,
            height,
        });
    }

    let diff_count = if len >= PARALLEL_MIN_PIXELS {
        match max_diffs {
            Some(cap) => {
                let counter = AtomicUsize::new(0);
                (0..height)
                    .into_par_iter()
                    .map(|y| {
                        if counter.load(Ordering::Relaxed) >= cap {
                            return 0;
                        }
                        let row_diff = process_row_count(
                            y, img1, img2, width, height, a32, b32, max_delta, include_aa,
                        );
                        counter.fetch_add(row_diff, Ordering::Relaxed);
                        row_diff
                    })
                    .sum()
            }
            None => (0..height)
                .into_par_iter()
                .map(|y| {
                    process_row_count(
                        y, img1, img2, width, height, a32, b32, max_delta, include_aa,
                    )
                })
                .sum(),
        }
    } else {
        let mut diff = 0usize;
        for y in 0..height {
            diff += process_row_count(
                y, img1, img2, width, height, a32, b32, max_delta, include_aa,
            );
            if let Some(cap) = max_diffs {
                if diff >= cap {
                    break;
                }
            }
        }
        diff
    };

    Ok(PixelmatchCountOutput {
        diff_count,
        width,
        height,
    })
}

/// Fast u32 mismatch count — processes in chunks for cache friendliness.
fn count_u32_mismatches(a: &[u32], b: &[u32], len: usize, max_diffs: Option<usize>) -> usize {
    if len >= PARALLEL_MIN_PIXELS {
        match max_diffs {
            Some(cap) => {
                let counter = AtomicUsize::new(0);
                a.par_chunks(PARALLEL_ROW_BLOCK * 64)
                    .zip(b.par_chunks(PARALLEL_ROW_BLOCK * 64))
                    .map(|(chunk_a, chunk_b)| {
                        if counter.load(Ordering::Relaxed) >= cap {
                            return 0;
                        }
                        let chunk_diff = chunk_a
                            .iter()
                            .zip(chunk_b.iter())
                            .filter(|(a, b)| a != b)
                            .count();
                        counter.fetch_add(chunk_diff, Ordering::Relaxed);
                        chunk_diff
                    })
                    .sum()
            }
            None => a
                .par_chunks(PARALLEL_ROW_BLOCK * 64)
                .zip(b.par_chunks(PARALLEL_ROW_BLOCK * 64))
                .map(|(chunk_a, chunk_b)| {
                    chunk_a
                        .iter()
                        .zip(chunk_b.iter())
                        .filter(|(a, b)| a != b)
                        .count()
                })
                .sum(),
        }
    } else {
        let mut count = 0usize;
        for (a, b) in a.iter().zip(b.iter()) {
            if a != b {
                count += 1;
                if let Some(cap) = max_diffs {
                    if count >= cap {
                        break;
                    }
                }
            }
        }
        count
    }
}

/// Fast mask building — threshold=0 + no-AA means any u32 mismatch is a diff.
fn build_mask_u32_fast(a: &[u32], b: &[u32], len: usize) -> (Vec<bool>, usize) {
    let mut mask = vec![false; len];
    let diff_count = if len >= PARALLEL_MIN_PIXELS {
        let chunk_size = PARALLEL_ROW_BLOCK * 64;
        mask.par_chunks_mut(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk_mask)| {
                let offset = chunk_idx * chunk_size;
                let mut count = 0;
                for (i, m) in chunk_mask.iter_mut().enumerate() {
                    if a[offset + i] != b[offset + i] {
                        *m = true;
                        count += 1;
                    }
                }
                count
            })
            .sum()
    } else {
        let mut count = 0;
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            if av != bv {
                mask[i] = true;
                count += 1;
            }
        }
        count
    };
    (mask, diff_count)
}

/// Result of a mask-producing pixel diff.
#[derive(Debug, Clone)]
pub struct PixelmatchMaskOutput {
    /// Boolean mask where `true` = pixel differs beyond threshold.
    pub diff_mask: Vec<bool>,
    /// Number of pixels that differ beyond the threshold.
    pub diff_count: usize,
    pub width: usize,
    pub height: usize,
}

/// Pixel diff that produces a boolean mask instead of a diff image.
///
/// The mask can be fed into [`crate::clusters::compute_clusters`] for
/// connected-component analysis.
pub fn pixelmatch_mask_rgba(
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    options: &PixelmatchOptions,
) -> Result<PixelmatchMaskOutput, Error> {
    validate_options(options)?;
    validate_rgba_len(img1.len(), width, height)?;
    validate_rgba_len(img2.len(), width, height)?;

    let len = width.checked_mul(height).ok_or(Error::Overflow)?;

    if img1 == img2 {
        return Ok(PixelmatchMaskOutput {
            diff_mask: vec![false; len],
            diff_count: 0,
            width,
            height,
        });
    }

    let max_delta = MAX_YIQ_DELTA * options.threshold * options.threshold;
    let a32_buf = rgba_as_u32_slice(img1);
    let b32_buf = rgba_as_u32_slice(img2);
    let a32 = a32_buf.as_slice();
    let b32 = b32_buf.as_slice();
    let include_aa = options.include_aa;

    // Threshold=0 + no-AA fast path for mask building.
    if max_delta == 0.0 && !include_aa {
        let (mask, diff_count) = build_mask_u32_fast(a32, b32, len);
        return Ok(PixelmatchMaskOutput {
            diff_mask: mask,
            diff_count,
            width,
            height,
        });
    }

    let mut mask = vec![false; len];

    let diff_count = if len >= PARALLEL_MIN_PIXELS {
        let row_stride = width;
        mask.par_chunks_mut(row_stride)
            .enumerate()
            .map(|(y, row_mask)| {
                process_row_mask(
                    y, row_mask, img1, img2, width, height, a32, b32, max_delta, include_aa,
                )
            })
            .sum()
    } else {
        let mut diff = 0usize;
        for y in 0..height {
            let row_start = y * width;
            let row_end = row_start + width;
            diff += process_row_mask(
                y,
                &mut mask[row_start..row_end],
                img1,
                img2,
                width,
                height,
                a32,
                b32,
                max_delta,
                include_aa,
            );
        }
        diff
    };

    Ok(PixelmatchMaskOutput {
        diff_mask: mask,
        diff_count,
        width,
        height,
    })
}

#[inline]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn process_row_mask(
    y: usize,
    row_mask: &mut [bool],
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    a32: &[u32],
    b32: &[u32],
    max_delta: f64,
    include_aa: bool,
) -> usize {
    let mut diff_count = 0usize;
    let row_offset_pixels = y * width;

    for x in 0..width {
        let pixel_index = row_offset_pixels + x;
        if a32[pixel_index] == b32[pixel_index] {
            continue;
        }

        let pixel_pos = pixel_index * 4;
        let delta = color_delta(img1, img2, pixel_pos, pixel_pos, false);
        if delta.abs() <= max_delta {
            continue;
        }

        let excluded_aa = !include_aa
            && (antialiased(img1, x, y, width, height, a32, b32)
                || antialiased(img2, x, y, width, height, b32, a32));

        if !excluded_aa {
            row_mask[x] = true;
            diff_count += 1;
        }
    }

    diff_count
}

#[allow(clippy::too_many_arguments)]
fn process_row(
    y: usize,
    row_out: &mut [u8],
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    a32: &[u32],
    b32: &[u32],
    max_delta: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: [u8; 3],
    aa_color: [u8; 3],
    diff_color_alt: Option<[u8; 3]>,
) -> usize {
    let mut diff_count = 0usize;
    let row_offset_pixels = y * width;

    for x in 0..width {
        let pixel_index = row_offset_pixels + x;
        let pixel_pos = pixel_index * 4;
        let out_pos = x * 4;

        let delta = if a32[pixel_index] == b32[pixel_index] {
            0.0
        } else {
            color_delta(img1, img2, pixel_pos, pixel_pos, false)
        };

        if delta.abs() > max_delta {
            let excluded_aa = !include_aa
                && (antialiased(img1, x, y, width, height, a32, b32)
                    || antialiased(img2, x, y, width, height, b32, a32));

            if excluded_aa {
                draw_pixel(row_out, out_pos, aa_color[0], aa_color[1], aa_color[2]);
            } else {
                let color = if delta < 0.0 {
                    diff_color_alt.unwrap_or(diff_color)
                } else {
                    diff_color
                };
                draw_pixel(row_out, out_pos, color[0], color[1], color[2]);
                diff_count += 1;
            }
        } else {
            let gray = gray_pixel_value(img1, pixel_pos, alpha);
            draw_pixel(row_out, out_pos, gray, gray, gray);
        }
    }

    diff_count
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_row_count(
    y: usize,
    img1: &[u8],
    img2: &[u8],
    width: usize,
    height: usize,
    a32: &[u32],
    b32: &[u32],
    max_delta: f64,
    include_aa: bool,
) -> usize {
    let mut diff_count = 0usize;
    let row_offset_pixels = y * width;

    for x in 0..width {
        let pixel_index = row_offset_pixels + x;
        if a32[pixel_index] == b32[pixel_index] {
            continue;
        }

        let pixel_pos = pixel_index * 4;
        let delta = color_delta(img1, img2, pixel_pos, pixel_pos, false);
        if delta.abs() <= max_delta {
            continue;
        }

        let excluded_aa = !include_aa
            && (antialiased(img1, x, y, width, height, a32, b32)
                || antialiased(img2, x, y, width, height, b32, a32));

        if !excluded_aa {
            diff_count += 1;
        }
    }

    diff_count
}

enum U32Pixels<'a> {
    Borrowed(&'a [u32]),
    Owned(Vec<u32>),
}

impl U32Pixels<'_> {
    fn as_slice(&self) -> &[u32] {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Owned(vec) => vec,
        }
    }
}

fn validate_options(options: &PixelmatchOptions) -> Result<(), Error> {
    if !(0.0..=1.0).contains(&options.threshold) {
        return Err(Error::InvalidOption(
            "threshold must be in the range [0.0, 1.0]",
        ));
    }

    if !(0.0..=1.0).contains(&options.alpha) {
        return Err(Error::InvalidOption(
            "alpha must be in the range [0.0, 1.0]",
        ));
    }

    Ok(())
}

fn rgba_as_u32_slice(img: &[u8]) -> U32Pixels<'_> {
    // SAFETY: We only accept the aligned view when both prefix and suffix are empty, so
    // the returned slice covers the full byte buffer with valid alignment and length.
    let (prefix, words, suffix) = unsafe { img.align_to::<u32>() };
    if prefix.is_empty() && suffix.is_empty() {
        U32Pixels::Borrowed(words)
    } else {
        U32Pixels::Owned(rgba_to_u32_owned(img))
    }
}

fn rgba_to_u32_owned(img: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(img.len() / 4);
    for px in img.chunks_exact(4) {
        out.push(u32::from_ne_bytes([px[0], px[1], px[2], px[3]]));
    }
    out
}

#[inline]
fn antialiased(
    img: &[u8],
    x1: usize,
    y1: usize,
    width: usize,
    height: usize,
    a32: &[u32],
    b32: &[u32],
) -> bool {
    let x0 = x1.saturating_sub(1);
    let y0 = y1.saturating_sub(1);
    let x2 = (x1 + 1).min(width - 1);
    let y2 = (y1 + 1).min(height - 1);

    let pos = y1 * width + x1;

    let mut zeroes = if x1 == x0 || x1 == x2 || y1 == y0 || y1 == y2 {
        1
    } else {
        0
    };

    let mut min = 0.0;
    let mut max = 0.0;
    let mut min_x = 0usize;
    let mut min_y = 0usize;
    let mut max_x = 0usize;
    let mut max_y = 0usize;

    for x in x0..=x2 {
        for y in y0..=y2 {
            if x == x1 && y == y1 {
                continue;
            }

            let delta = color_delta(img, img, pos * 4, (y * width + x) * 4, true);

            if delta == 0.0 {
                zeroes += 1;
                if zeroes > 2 {
                    return false;
                }
            } else if delta < min {
                min = delta;
                min_x = x;
                min_y = y;
            } else if delta > max {
                max = delta;
                max_x = x;
                max_y = y;
            }
        }
    }

    if min == 0.0 || max == 0.0 {
        return false;
    }

    (has_many_siblings(a32, min_x, min_y, width, height)
        && has_many_siblings(b32, min_x, min_y, width, height))
        || (has_many_siblings(a32, max_x, max_y, width, height)
            && has_many_siblings(b32, max_x, max_y, width, height))
}

#[inline]
fn has_many_siblings(img: &[u32], x1: usize, y1: usize, width: usize, height: usize) -> bool {
    let x0 = x1.saturating_sub(1);
    let y0 = y1.saturating_sub(1);
    let x2 = (x1 + 1).min(width - 1);
    let y2 = (y1 + 1).min(height - 1);
    let val = img[y1 * width + x1];

    let mut zeroes = if x1 == x0 || x1 == x2 || y1 == y0 || y1 == y2 {
        1
    } else {
        0
    };

    for x in x0..=x2 {
        for y in y0..=y2 {
            if x == x1 && y == y1 {
                continue;
            }
            if val == img[y * width + x] {
                zeroes += 1;
                if zeroes > 2 {
                    return true;
                }
            }
        }
    }

    false
}

#[inline]
fn color_delta(img1: &[u8], img2: &[u8], k: usize, m: usize, y_only: bool) -> f64 {
    let r1 = img1[k] as f64;
    let g1 = img1[k + 1] as f64;
    let b1 = img1[k + 2] as f64;
    let a1 = img1[k + 3] as f64;

    let r2 = img2[m] as f64;
    let g2 = img2[m + 1] as f64;
    let b2 = img2[m + 2] as f64;
    let a2 = img2[m + 3] as f64;

    let mut dr = r1 - r2;
    let mut dg = g1 - g2;
    let mut db = b1 - b2;
    let da = a1 - a2;

    if dr == 0.0 && dg == 0.0 && db == 0.0 && da == 0.0 {
        return 0.0;
    }

    if a1 < 255.0 || a2 < 255.0 {
        let rb = 48.0 + 159.0 * ((k % 2) as f64);
        let gb = 48.0 + 159.0 * parity_from_ratio(k, GOLDEN_RATIO);
        let bb = 48.0 + 159.0 * parity_from_ratio(k, GOLDEN_RATIO_PLUS_ONE);

        dr = (r1 * a1 - r2 * a2 - rb * da) / 255.0;
        dg = (g1 * a1 - g2 * a2 - gb * da) / 255.0;
        db = (b1 * a1 - b2 * a2 - bb * da) / 255.0;
    }

    let y = dr * 0.298_895_31 + dg * 0.586_622_47 + db * 0.114_482_23;
    if y_only {
        return y;
    }

    let i = dr * 0.595_977_99 - dg * 0.274_176_10 - db * 0.321_801_89;
    let q = dr * 0.211_470_17 - dg * 0.522_617_11 + db * 0.311_146_94;

    let delta = 0.5053 * y * y + 0.299 * i * i + 0.1957 * q * q;

    if y > 0.0 {
        -delta
    } else {
        delta
    }
}

#[inline]
fn parity_from_ratio(index: usize, divisor: f64) -> f64 {
    ((index as f64 / divisor).floor() as i64).rem_euclid(2) as f64
}

#[inline]
fn draw_pixel(output: &mut [u8], pos: usize, r: u8, g: u8, b: u8) {
    output[pos] = r;
    output[pos + 1] = g;
    output[pos + 2] = b;
    output[pos + 3] = 255;
}

#[inline]
fn gray_pixel_value(img: &[u8], pos: usize, alpha: f64) -> u8 {
    let val = 255.0
        + (img[pos] as f64 * 0.298_895_31
            + img[pos + 1] as f64 * 0.586_622_47
            + img[pos + 2] as f64 * 0.114_482_23
            - 255.0)
            * alpha
            * (img[pos + 3] as f64 / 255.0);

    val.clamp(0.0, 255.0) as u8
}

fn draw_gray_pixel(img: &[u8], pos: usize, alpha: f64, output: &mut [u8]) {
    let v = gray_pixel_value(img, pos, alpha);
    draw_pixel(output, pos, v, v, v);
}

#[cfg(test)]
mod tests {
    use super::{pixelmatch_rgba, PixelmatchOptions};

    fn rgba_from_luma(values: &[[u8; 3]]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|row| row.iter())
            .flat_map(|v| [*v, *v, *v, 255])
            .collect()
    }

    #[test]
    fn aa_pixels_are_excluded_when_include_aa_false() {
        // 3x3 synthetic slope where center is a likely anti-aliased transition pixel.
        let img1 = rgba_from_luma(&[[100, 100, 200], [100, 128, 200], [100, 200, 200]]);
        let img2 = rgba_from_luma(&[[100, 100, 200], [100, 160, 200], [100, 200, 200]]);

        let include_aa = PixelmatchOptions {
            include_aa: true,
            ..PixelmatchOptions::default()
        };

        let exclude_aa = PixelmatchOptions {
            include_aa: false,
            ..PixelmatchOptions::default()
        };

        let with_aa = pixelmatch_rgba(&img1, &img2, 3, 3, &include_aa).unwrap();
        let without_aa = pixelmatch_rgba(&img1, &img2, 3, 3, &exclude_aa).unwrap();

        assert!(with_aa.diff_count >= without_aa.diff_count);
    }
}
