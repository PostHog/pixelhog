use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use pixelhog::{compute_ssim_png, pixelmatch_png, PixelmatchOptions};

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

fn bench_api(c: &mut Criterion) {
    let width = std::env::var("BENCH_WIDTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1059);
    let height = std::env::var("BENCH_HEIGHT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3642);

    let (baseline, current) = make_screenshot_pair(width, height);
    let options = PixelmatchOptions::default();

    let mut group = c.benchmark_group("api_png_bytes");

    group.bench_function("pixelmatch", |b| {
        b.iter(|| {
            let (diff_png, diff_count, out_w, out_h) = pixelmatch_png(
                black_box(&baseline),
                black_box(&current),
                black_box(&options),
            )
            .expect("pixelmatch benchmark should succeed");
            black_box((diff_png.len(), diff_count, out_w, out_h));
        })
    });

    group.bench_function("compute_ssim", |b| {
        b.iter(|| {
            let score = compute_ssim_png(black_box(&baseline), black_box(&current))
                .expect("ssim benchmark");
            black_box(score);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_api);
criterion_main!(benches);
