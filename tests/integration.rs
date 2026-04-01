use approx::assert_relative_eq;
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder, ImageFormat};
use pixelhog::{compute_ssim_png, pixelmatch_png, PixelmatchOptions};

fn encode_png(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let encoder = PngEncoder::new(&mut out);
    encoder
        .write_image(rgba, width as u32, height as u32, ColorType::Rgba8.into())
        .expect("failed to encode PNG");
    out
}

fn solid_png(width: usize, height: usize, color: [u8; 4]) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];
    for px in rgba.chunks_exact_mut(4) {
        px.copy_from_slice(&color);
    }
    encode_png(&rgba, width, height)
}

fn decode_png_dimensions(bytes: &[u8]) -> (u32, u32) {
    let img = image::load_from_memory_with_format(bytes, ImageFormat::Png)
        .expect("diff output should decode")
        .to_rgba8();
    img.dimensions()
}

#[test]
fn test_identical_images_zero_diff() {
    let baseline = solid_png(16, 12, [240, 10, 20, 255]);
    let current = baseline.clone();

    let options = PixelmatchOptions::default();
    let (diff_png, diff_count, width, height) =
        pixelmatch_png(&baseline, &current, &options).expect("pixelmatch should succeed");

    assert_eq!(diff_count, 0);
    assert_eq!((width, height), (16, 12));

    let (dw, dh) = decode_png_dimensions(&diff_png);
    assert_eq!((dw, dh), (16, 12));
}

#[test]
fn test_completely_different_images_full_diff() {
    let baseline = solid_png(10, 8, [0, 0, 0, 255]);
    let current = solid_png(10, 8, [255, 255, 255, 255]);

    let options = PixelmatchOptions::default();
    let (_, diff_count, width, height) =
        pixelmatch_png(&baseline, &current, &options).expect("pixelmatch should succeed");

    assert_eq!((width, height), (10, 8));
    assert_eq!(diff_count, 80);
}

#[test]
fn test_partial_diff() {
    let width = 20;
    let height = 10;

    let mut baseline_rgba = vec![0u8; width * height * 4];
    let mut current_rgba = vec![0u8; width * height * 4];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            baseline_rgba[idx..idx + 4].copy_from_slice(&[255, 0, 0, 255]);
            if x < width / 2 {
                current_rgba[idx..idx + 4].copy_from_slice(&[255, 0, 0, 255]);
            } else {
                current_rgba[idx..idx + 4].copy_from_slice(&[0, 0, 255, 255]);
            }
        }
    }

    let baseline = encode_png(&baseline_rgba, width, height);
    let current = encode_png(&current_rgba, width, height);

    let options = PixelmatchOptions::default();
    let (_, diff_count, _, _) =
        pixelmatch_png(&baseline, &current, &options).expect("pixelmatch should succeed");

    assert_eq!(diff_count, width * height / 2);
}

#[test]
fn test_different_sizes_pads_to_larger() {
    let baseline = solid_png(10, 8, [200, 0, 0, 255]);
    let current = solid_png(12, 10, [200, 0, 0, 255]);

    let options = PixelmatchOptions::default();
    let (_, diff_count, width, height) =
        pixelmatch_png(&baseline, &current, &options).expect("pixelmatch should succeed");

    assert_eq!((width, height), (12, 10));
    assert_eq!(diff_count, 40);
}

#[test]
fn test_threshold_controls_sensitivity() {
    let baseline = solid_png(8, 8, [120, 120, 120, 255]);
    let current = solid_png(8, 8, [132, 132, 132, 255]);

    let low = PixelmatchOptions {
        threshold: 0.01,
        ..PixelmatchOptions::default()
    };
    let high = PixelmatchOptions {
        threshold: 0.3,
        ..PixelmatchOptions::default()
    };

    let (_, low_count, _, _) = pixelmatch_png(&baseline, &current, &low).expect("low threshold");
    let (_, high_count, _, _) = pixelmatch_png(&baseline, &current, &high).expect("high threshold");

    assert!(low_count > high_count);
    assert_eq!(high_count, 0);
}

#[test]
fn test_diff_image_is_valid_png() {
    let baseline = solid_png(6, 6, [0, 255, 0, 255]);
    let current = solid_png(6, 6, [0, 0, 255, 255]);

    let options = PixelmatchOptions::default();
    let (diff_png, _, _, _) = pixelmatch_png(&baseline, &current, &options).expect("pixelmatch");

    let decoded = image::load_from_memory_with_format(&diff_png, ImageFormat::Png)
        .expect("diff output should be a valid PNG")
        .to_rgba8();

    assert_eq!(decoded.dimensions(), (6, 6));
}

#[test]
fn test_ssim_identical_images_score_one() {
    let baseline = solid_png(64, 64, [20, 40, 80, 255]);
    let current = baseline.clone();

    let score = compute_ssim_png(&baseline, &current).expect("ssim should succeed");
    assert_relative_eq!(score, 1.0, epsilon = 1e-12);
}

#[test]
fn test_ssim_completely_different_images_low_score() {
    let baseline = solid_png(64, 64, [0, 0, 0, 255]);
    let current = solid_png(64, 64, [255, 255, 255, 255]);

    let score = compute_ssim_png(&baseline, &current).expect("ssim should succeed");
    assert!(score < 0.1);
}

#[test]
fn test_ssim_slight_difference_high_score() {
    let width = 120;
    let height = 100;
    let mut baseline_rgba = vec![0u8; width * height * 4];
    let mut current_rgba = vec![0u8; width * height * 4];

    for px in baseline_rgba.chunks_exact_mut(4) {
        px.copy_from_slice(&[180, 180, 180, 255]);
    }
    current_rgba.copy_from_slice(&baseline_rgba);

    let idx = (height / 2 * width + width / 2) * 4;
    current_rgba[idx..idx + 4].copy_from_slice(&[170, 170, 170, 255]);

    let baseline = encode_png(&baseline_rgba, width, height);
    let current = encode_png(&current_rgba, width, height);

    let score = compute_ssim_png(&baseline, &current).expect("ssim should succeed");
    assert!(score > 0.98);
}

#[test]
fn test_ssim_small_images_below_window_size() {
    let baseline = solid_png(5, 5, [100, 100, 100, 255]);
    let current = solid_png(5, 5, [130, 130, 130, 255]);

    let score = compute_ssim_png(&baseline, &current).expect("ssim should succeed");
    assert!((0.0..=1.0).contains(&score));
    assert!(score < 1.0);
}

#[test]
fn test_ssim_different_sizes_pads_to_larger() {
    let baseline = solid_png(9, 9, [255, 255, 255, 255]);
    let current = solid_png(14, 14, [255, 255, 255, 255]);

    let score = compute_ssim_png(&baseline, &current).expect("ssim should succeed");

    assert!((0.0..=1.0).contains(&score));
    assert!(score < 1.0);
}
