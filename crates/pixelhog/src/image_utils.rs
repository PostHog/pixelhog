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
/// Constrains to `max_width`. If `max_height` is set, crops from the top
/// after resizing. Images already within bounds are re-encoded as-is.
pub fn thumbnail_webp(
    rgba: &[u8],
    width: usize,
    height: usize,
    max_width: usize,
    max_height: Option<usize>,
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

    let (scaled_w, scaled_h) = if width <= max_width {
        (width, height)
    } else {
        let new_h = ((height as f64) * (max_width as f64) / (width as f64)).round() as usize;
        (max_width, new_h.max(1))
    };

    let resized = if scaled_w == width && scaled_h == height {
        rgba.to_vec()
    } else {
        resize_rgba(rgba, width, height, scaled_w, scaled_h)?
    };

    let (final_rgba, final_w, final_h) = match max_height {
        Some(mh) if scaled_h > mh => {
            let cropped_len = scaled_w * mh * 4;
            (resized[..cropped_len].to_vec(), scaled_w, mh)
        }
        _ => (resized, scaled_w, scaled_h),
    };

    let (w32, h32) = to_u32_dims(final_w, final_h)?;
    let rgb = rgba_to_rgb(&final_rgba);
    encode_webp_lossless(&rgb, w32, h32)
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
