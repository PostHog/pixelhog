use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ColorType, ImageEncoder, ImageFormat};
use std::borrow::Cow;

pub type DecodedPng = (Vec<u8>, usize, usize);

pub fn decode_png_rgba(bytes: &[u8]) -> Result<DecodedPng, String> {
    let dynamic = image::load_from_memory_with_format(bytes, ImageFormat::Png)
        .map_err(|err| format!("failed to decode PNG: {err}"))?;
    let rgba = dynamic.to_rgba8();
    let (width, height) = rgba.dimensions();
    let width = usize::try_from(width).map_err(|_| "image width is too large".to_string())?;
    let height = usize::try_from(height).map_err(|_| "image height is too large".to_string())?;
    let raw = rgba.into_raw();

    validate_rgba_len(raw.len(), width, height)?;

    Ok((raw, width, height))
}

pub fn encode_png_rgba(rgba: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    validate_rgba_len(rgba.len(), width, height)?;

    let width_u32 = u32::try_from(width).map_err(|_| "image width is too large".to_string())?;
    let height_u32 = u32::try_from(height).map_err(|_| "image height is too large".to_string())?;

    let mut out = Vec::new();
    let encoder =
        PngEncoder::new_with_quality(&mut out, CompressionType::Fast, FilterType::NoFilter);
    encoder
        .write_image(rgba, width_u32, height_u32, ColorType::Rgba8.into())
        .map_err(|err| format!("failed to encode PNG: {err}"))?;

    Ok(out)
}

#[allow(dead_code)]
pub fn pad_images_to_largest_owned(
    img1: Vec<u8>,
    width1: usize,
    height1: usize,
    img2: Vec<u8>,
    width2: usize,
    height2: usize,
) -> Result<(Vec<u8>, Vec<u8>, usize, usize), String> {
    validate_rgba_len(img1.len(), width1, height1)?;
    validate_rgba_len(img2.len(), width2, height2)?;

    let width = width1.max(width2);
    let height = height1.max(height2);

    if width == width1 && height == height1 && width == width2 && height == height2 {
        return Ok((img1, img2, width, height));
    }

    let padded1 = if width == width1 && height == height1 {
        img1
    } else {
        pad_rgba_to_size(&img1, width1, height1, width, height)?
    };

    let padded2 = if width == width2 && height == height2 {
        img2
    } else {
        pad_rgba_to_size(&img2, width2, height2, width, height)?
    };

    Ok((padded1, padded2, width, height))
}

pub fn pad_images_to_largest_cow<'a>(
    img1: &'a [u8],
    width1: usize,
    height1: usize,
    img2: &'a [u8],
    width2: usize,
    height2: usize,
) -> Result<(Cow<'a, [u8]>, Cow<'a, [u8]>, usize, usize), String> {
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
) -> Result<Vec<u8>, String> {
    if width > target_width || height > target_height {
        return Err("source image is larger than target size".to_string());
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

pub fn validate_rgba_len(len: usize, width: usize, height: usize) -> Result<(), String> {
    let expected = checked_len(width, height)?;
    if len != expected {
        return Err(format!(
            "invalid RGBA buffer length: expected {expected} bytes for {width}x{height}, got {len}"
        ));
    }
    Ok(())
}

fn checked_len(width: usize, height: usize) -> Result<usize, String> {
    width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| "image dimensions overflowed buffer size".to_string())
}
