use std::time::Instant;

use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use pixelhog::image_utils::{decode_png_rgba, encode_png_rgba, pad_images_to_largest_cow};
use pixelhog::pixelmatch::PixelmatchOptions as CorePixelmatchOptions;
use pixelhog::ssim::compute_ssim_rgba;
use pixelhog::{diff_png, ssim_png, PixelmatchOptions as ApiPixelmatchOptions};
use rayon::join;

fn encode_png(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let encoder = PngEncoder::new(&mut out);
    encoder
        .write_image(rgba, width as u32, height as u32, ColorType::Rgba8.into())
        .expect("failed to encode benchmark image");
    out
}

fn make_screenshot_pair(width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    let mut baseline = vec![0u8; width * height * 4];
    let mut current = vec![0u8; width * height * 4];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;

            let r = ((x * 13 + y * 3) % 256) as u8;
            let g = ((x * 7 + y * 11) % 256) as u8;
            let b = ((x * 5 + y * 17) % 256) as u8;
            let a = if (x + y) % 8 == 0 { 220 } else { 255 };

            baseline[idx] = r;
            baseline[idx + 1] = g;
            baseline[idx + 2] = b;
            baseline[idx + 3] = a;

            let mut r2 = r;
            let mut g2 = g;
            let mut b2 = b;
            let mut a2 = a;

            if x > width / 3 && x < width / 3 * 2 && y > height / 4 && y < height / 4 * 3 {
                r2 = r2.wrapping_add(12);
                g2 = g2.wrapping_add(20);
                b2 = b2.wrapping_sub(5);
            }

            if (x + 2 * y) % 97 == 0 {
                a2 = a2.saturating_sub(80);
            }

            current[idx] = r2;
            current[idx + 1] = g2;
            current[idx + 2] = b2;
            current[idx + 3] = a2;
        }
    }

    (
        encode_png(&baseline, width, height),
        encode_png(&current, width, height),
    )
}

fn avg_ms(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}

fn main() {
    let width = 1059usize;
    let height = 3642usize;
    let runs = 20usize;

    let (baseline_png, current_png) = make_screenshot_pair(width, height);
    let options = CorePixelmatchOptions::default();
    let api_options = ApiPixelmatchOptions::default();

    // Warmup
    for _ in 0..3 {
        let _ = diff_png(&baseline_png, &current_png, &api_options).unwrap();
        let _ = ssim_png(&baseline_png, &current_png).unwrap();
    }

    let mut separate_ms = Vec::with_capacity(runs);
    let mut combined_ms = Vec::with_capacity(runs);

    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = diff_png(&baseline_png, &current_png, &api_options).expect("diff_png");
        let _ = ssim_png(&baseline_png, &current_png).expect("ssim_png");
        let t1 = Instant::now();
        separate_ms.push((t1 - t0).as_secs_f64() * 1000.0);

        let t2 = Instant::now();
        let (left, right) = join(
            || decode_png_rgba(&baseline_png),
            || decode_png_rgba(&current_png),
        );
        let (baseline_rgba, bw, bh) = left.expect("decode left");
        let (current_rgba, cw, ch) = right.expect("decode right");
        let (baseline_padded, current_padded, w, h) =
            pad_images_to_largest_cow(&baseline_rgba, bw, bh, &current_rgba, cw, ch).expect("pad");

        let diff = pixelhog::pixelmatch::pixelmatch_rgba(baseline_padded.as_ref(), current_padded.as_ref(), w, h, &options)
            .expect("core pixelmatch");
        let _score = compute_ssim_rgba(baseline_padded.as_ref(), current_padded.as_ref(), w, h).expect("core ssim");
        let _diff_png = encode_png_rgba(&diff.diff_rgba, w, h).expect("encode diff");
        let t3 = Instant::now();

        combined_ms.push((t3 - t2).as_secs_f64() * 1000.0);
    }

    let sep = avg_ms(&separate_ms);
    let comb = avg_ms(&combined_ms);
    let saved = sep - comb;
    let saved_pct = if sep > 0.0 { saved / sep * 100.0 } else { 0.0 };

    println!("separate_calls_avg_ms={sep:.2}");
    println!("combined_single_decode_avg_ms={comb:.2}");
    println!("saved_ms={saved:.2}");
    println!("saved_percent={saved_pct:.2}");
}
