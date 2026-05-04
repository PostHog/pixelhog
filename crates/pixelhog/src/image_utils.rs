use crate::Error;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::imageops::FilterType as ResizeFilter;
use image::{ColorType, ImageEncoder, ImageFormat, RgbaImage};
use std::borrow::Cow;

/// `(rgba_bytes, width, height)`
pub type DecodedPng = (Vec<u8>, usize, usize);

/// Decode a PNG image into raw RGBA bytes.
pub fn decode_png_rgba(bytes: &[u8]) -> Result<DecodedPng, Error> {
    let dynamic = image::load_from_memory_with_format(bytes, ImageFormat::Png)?;
    let rgba = dynamic.to_rgba8();
    let (width, height) = rgba.dimensions();
    let width =
        usize::try_from(width).map_err(|_| Error::DimensionTooLarge { dimension: "width" })?;
    let height = usize::try_from(height).map_err(|_| Error::DimensionTooLarge {
        dimension: "height",
    })?;
    let raw = rgba.into_raw();

    validate_rgba_len(raw.len(), width, height)?;

    Ok((raw, width, height))
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    rgb
}

fn to_u32_dims(width: usize, height: usize) -> Result<(u32, u32), Error> {
    let w = u32::try_from(width).map_err(|_| Error::DimensionTooLarge { dimension: "width" })?;
    let h = u32::try_from(height).map_err(|_| Error::DimensionTooLarge {
        dimension: "height",
    })?;
    Ok((w, h))
}

/// Encode raw RGBA bytes into an RGB PNG image (alpha channel is stripped).
pub fn encode_png(rgba: &[u8], width: usize, height: usize) -> Result<Vec<u8>, Error> {
    validate_rgba_len(rgba.len(), width, height)?;
    let (width_u32, height_u32) = to_u32_dims(width, height)?;
    let rgb = rgba_to_rgb(rgba);

    let mut out = Vec::new();
    let encoder =
        PngEncoder::new_with_quality(&mut out, CompressionType::Default, FilterType::Adaptive);
    encoder
        .write_image(&rgb, width_u32, height_u32, ColorType::Rgb8.into())
        .map_err(Error::Encode)?;

    Ok(out)
}

fn resize_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    new_width: usize,
    new_height: usize,
) -> Result<Vec<u8>, Error> {
    let (w, h) = to_u32_dims(width, height)?;
    let (nw, nh) = to_u32_dims(new_width, new_height)?;
    let img = RgbaImage::from_raw(w, h, rgba.to_vec())
        .ok_or_else(|| Error::Resize("failed to create image buffer".into()))?;
    let resized = image::imageops::resize(&img, nw, nh, ResizeFilter::Lanczos3);
    Ok(resized.into_raw())
}

fn encode_webp_lossless(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, Error> {
    use image::codecs::webp::WebPEncoder;

    let mut out = Vec::new();
    let encoder = WebPEncoder::new_lossless(&mut out);
    encoder
        .write_image(rgb, width, height, ColorType::Rgb8.into())
        .map_err(Error::EncodeWebp)?;

    Ok(out)
}

/// Resize RGBA bytes to a lossless WebP thumbnail.
///
/// Uses bounding-box contain with minimum dimension floors:
/// 1. Scale to fit within `max_width × max_height` (contain)
/// 2. If either dimension would fall below `min_width` / `min_height`,
///    skip resize and top-left crop the original instead
/// 3. Images already within bounds are re-encoded as-is
///
/// Top-left crop preserves the most informative region of screenshots.
pub fn thumbnail_webp(
    rgba: &[u8],
    width: usize,
    height: usize,
    max_width: usize,
    max_height: Option<usize>,
) -> Result<Vec<u8>, Error> {
    thumbnail_webp_full(rgba, width, height, max_width, max_height, None, None)
}

/// Full thumbnail generation with minimum dimension floors.
///
/// Algorithm: scale to `max_width`, crop to `max_height` from top-left.
/// If the result would be smaller than `min_width` or `min_height`,
/// skip the resize and top-left crop the original instead.
/// No upscaling ever happens.
#[allow(clippy::too_many_arguments)]
pub fn thumbnail_webp_full(
    rgba: &[u8],
    width: usize,
    height: usize,
    max_width: usize,
    max_height: Option<usize>,
    min_width: Option<usize>,
    min_height: Option<usize>,
) -> Result<Vec<u8>, Error> {
    validate_rgba_len(rgba.len(), width, height)?;

    if max_width == 0 {
        return Err(Error::InvalidOption("thumbnail max_width must be > 0"));
    }
    if max_height == Some(0) {
        return Err(Error::InvalidOption("thumbnail max_height must be > 0"));
    }

    if width == 0 || height == 0 {
        let (w, h) = to_u32_dims(width, height)?;
        let rgb = rgba_to_rgb(rgba);
        return encode_webp_lossless(&rgb, w, h);
    }

    let effective_max_h = max_height.unwrap_or(usize::MAX);
    let effective_min_w = min_width.unwrap_or(0);
    let effective_min_h = min_height.unwrap_or(0);

    // Scale to max_width (preserve aspect ratio, never upscale).
    let (scaled_w, scaled_h) = if width <= max_width {
        (width, height)
    } else {
        let new_h = ((height as f64) * (max_width as f64) / (width as f64)).round() as usize;
        (max_width, new_h.max(1))
    };

    // Check if scaling would violate minimum dimension floors.
    if scaled_w < effective_min_w || scaled_h < effective_min_h {
        // Skip resize, top-left crop the original.
        let crop_w = width.min(max_width);
        let crop_h = height.min(effective_max_h);
        let cropped = top_left_crop(rgba, width, crop_w, crop_h);
        let (w32, h32) = to_u32_dims(crop_w, crop_h)?;
        let rgb = rgba_to_rgb(&cropped);
        return encode_webp_lossless(&rgb, w32, h32);
    }

    // Resize.
    let resized = if scaled_w == width && scaled_h == height {
        rgba.to_vec()
    } else {
        resize_rgba(rgba, width, height, scaled_w, scaled_h)?
    };

    // Crop to max_height from top-left if needed.
    let (final_rgba, final_w, final_h) = if scaled_h > effective_max_h {
        let cropped = top_left_crop(&resized, scaled_w, scaled_w, effective_max_h);
        (cropped, scaled_w, effective_max_h)
    } else {
        (resized, scaled_w, scaled_h)
    };

    let (w32, h32) = to_u32_dims(final_w, final_h)?;
    let rgb = rgba_to_rgb(&final_rgba);
    encode_webp_lossless(&rgb, w32, h32)
}

fn top_left_crop(rgba: &[u8], src_width: usize, crop_w: usize, crop_h: usize) -> Vec<u8> {
    let src_stride = src_width * 4;
    let dst_stride = crop_w * 4;
    let mut out = Vec::with_capacity(crop_w * crop_h * 4);
    for row in 0..crop_h {
        let src_start = row * src_stride;
        out.extend_from_slice(&rgba[src_start..src_start + dst_stride]);
    }
    out
}

/// Pad two RGBA images to matching dimensions using transparent pixels.
///
/// Returns borrowed slices when no padding is needed (zero-copy).
#[allow(clippy::type_complexity)]
pub fn pad_images_to_largest_cow<'a>(
    img1: &'a [u8],
    width1: usize,
    height1: usize,
    img2: &'a [u8],
    width2: usize,
    height2: usize,
) -> Result<(Cow<'a, [u8]>, Cow<'a, [u8]>, usize, usize), Error> {
    validate_rgba_len(img1.len(), width1, height1)?;
    validate_rgba_len(img2.len(), width2, height2)?;

    let width = width1.max(width2);
    let height = height1.max(height2);

    if width == width1 && height == height1 && width == width2 && height == height2 {
        return Ok((Cow::Borrowed(img1), Cow::Borrowed(img2), width, height));
    }

    let padded1 = if width == width1 && height == height1 {
        Cow::Borrowed(img1)
    } else {
        Cow::Owned(pad_rgba_to_size(img1, width1, height1, width, height)?)
    };

    let padded2 = if width == width2 && height == height2 {
        Cow::Borrowed(img2)
    } else {
        Cow::Owned(pad_rgba_to_size(img2, width2, height2, width, height)?)
    };

    Ok((padded1, padded2, width, height))
}

/// Convert RGBA bytes to grayscale using BT.601 luminance weights.
pub fn rgba_to_grayscale_f64(rgba: &[u8]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rgba.len() / 4);
    for px in rgba.chunks_exact(4) {
        let r = px[0] as f64;
        let g = px[1] as f64;
        let b = px[2] as f64;
        out.push(r * 0.299 + g * 0.587 + b * 0.114);
    }
    out
}

fn pad_rgba_to_size(
    rgba: &[u8],
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Result<Vec<u8>, Error> {
    if width > target_width || height > target_height {
        return Err(Error::PadOverflow);
    }

    let target_len = checked_len(target_width, target_height)?;
    let mut padded = vec![0u8; target_len];

    if width == 0 || height == 0 {
        return Ok(padded);
    }

    let src_stride = width * 4;
    let dst_stride = target_width * 4;

    for row in 0..height {
        let src_start = row * src_stride;
        let src_end = src_start + src_stride;
        let dst_start = row * dst_stride;
        let dst_end = dst_start + src_stride;
        padded[dst_start..dst_end].copy_from_slice(&rgba[src_start..src_end]);
    }

    Ok(padded)
}

/// Validate that an RGBA buffer has the expected length for the given dimensions.
pub fn validate_rgba_len(len: usize, width: usize, height: usize) -> Result<(), Error> {
    let expected = checked_len(width, height)?;
    if len != expected {
        return Err(Error::BufferLength {
            expected,
            actual: len,
            width,
            height,
        });
    }
    Ok(())
}

fn checked_len(width: usize, height: usize) -> Result<usize, Error> {
    width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(4))
        .ok_or(Error::Overflow)
}
