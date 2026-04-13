use std::time::Instant;

use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use pixelhog::image_utils::{decode_png_rgba, encode_png_rgba, pad_images_to_largest_cow};
use pixelhog::pixelmatch::{pixelmatch_rgba, PixelmatchOptions};
use pixelhog::{diff_png, PixelmatchOptions as ApiPixelmatchOptions};

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
    let options = PixelmatchOptions::default();
    let api_options = ApiPixelmatchOptions::default();

    let mut decode_ms = Vec::with_capacity(runs);
    let mut core_ms = Vec::with_capacity(runs);
    let mut encode_ms = Vec::with_capacity(runs);
    let mut full_ms = Vec::with_capacity(runs);
    let mut api_ms = Vec::with_capacity(runs);

    for _ in 0..runs {
        let t0 = Instant::now();
        let (baseline_rgba, bw, bh) = decode_png_rgba(&baseline_png).expect("decode baseline");
        let (current_rgba, cw, ch) = decode_png_rgba(&current_png).expect("decode current");
        let (baseline_padded, current_padded, w, h) =
            pad_images_to_largest_cow(&baseline_rgba, bw, bh, &current_rgba, cw, ch)
                .expect("pad images");
        let t1 = Instant::now();

        let out = pixelmatch_rgba(
            baseline_padded.as_ref(),
            current_padded.as_ref(),
            w,
            h,
            &options,
        )
        .expect("pixelmatch core");
        let t2 = Instant::now();

        let _diff_png = encode_png_rgba(&out.diff_rgba, w, h).expect("encode diff png");
        let t3 = Instant::now();

        let _api = diff_png(&baseline_png, &current_png, &api_options).expect("api pixelmatch");
        let t4 = Instant::now();

        decode_ms.push((t1 - t0).as_secs_f64() * 1000.0);
        core_ms.push((t2 - t1).as_secs_f64() * 1000.0);
        encode_ms.push((t3 - t2).as_secs_f64() * 1000.0);
        full_ms.push((t3 - t0).as_secs_f64() * 1000.0);
        api_ms.push((t4 - t3).as_secs_f64() * 1000.0);
    }

    println!("decode_avg_ms={:.2}", avg_ms(&decode_ms));
    println!("core_pixelmatch_avg_ms={:.2}", avg_ms(&core_ms));
    println!("encode_avg_ms={:.2}", avg_ms(&encode_ms));
    println!("full_pipeline_avg_ms={:.2}", avg_ms(&full_ms));
    println!("diff_png_api_avg_ms={:.2}", avg_ms(&api_ms));
}
