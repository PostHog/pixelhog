use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use pixelhog::{
    compare_png, diff_clusters_png, diff_count_png, diff_png, pixelmatch_count_rgba_capped,
    ssim_png, ClusterOptions, ComparePngOutput, DiffCountOutput, DiffPngOutput, PixelmatchOptions,
};

fn encode_png(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let encoder = PngEncoder::new(&mut out);
    encoder
        .write_image(rgba, width as u32, height as u32, ColorType::Rgba8.into())
        .expect("failed to encode benchmark image");
    out
}

struct TestImage {
    name: &'static str,
    width: usize,
    height: usize,
}

const IMAGES: &[TestImage] = &[
    TestImage {
        name: "2.1M_fullhd",
        width: 1920,
        height: 1080,
    },
    TestImage {
        name: "2.5M_dashboard",
        width: 826,
        height: 3070,
    },
    TestImage {
        name: "18M_scrollable",
        width: 750,
        height: 24162,
    },
];

/// Generate a pair of images with a realistic diff pattern.
///
/// ~30% of pixels differ in a rectangular region (simulating a content change),
/// plus sparse scattered single-pixel noise.
fn make_screenshot_pair(width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    let len = width * height * 4;
    let mut baseline = vec![0u8; len];
    let mut current = vec![0u8; len];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;

            let r = ((x * 13 + y * 3) % 256) as u8;
            let g = ((x * 7 + y * 11) % 256) as u8;
            let b = ((x * 5 + y * 17) % 256) as u8;

            baseline[idx] = r;
            baseline[idx + 1] = g;
            baseline[idx + 2] = b;
            baseline[idx + 3] = 255;

            current[idx] = r;
            current[idx + 1] = g;
            current[idx + 2] = b;
            current[idx + 3] = 255;

            // ~30% block diff in center region
            if x > width / 3 && x < width * 2 / 3 && y > height / 4 && y < height * 3 / 4 {
                current[idx] = r.wrapping_add(12);
                current[idx + 1] = g.wrapping_add(20);
                current[idx + 2] = b.wrapping_sub(5);
            }

            // Sparse noise (~1% of pixels)
            if (x * 31 + y * 37) % 97 == 0 {
                current[idx] = r.wrapping_add(3);
            }
        }
    }

    (
        encode_png(&baseline, width, height),
        encode_png(&current, width, height),
    )
}

/// Generate raw RGBA pair (no PNG encoding) for functions that take raw buffers.
fn make_screenshot_pair_rgba(width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    let len = width * height * 4;
    let mut baseline = vec![0u8; len];
    let mut current = vec![0u8; len];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;

            let r = ((x * 13 + y * 3) % 256) as u8;
            let g = ((x * 7 + y * 11) % 256) as u8;
            let b = ((x * 5 + y * 17) % 256) as u8;

            baseline[idx] = r;
            baseline[idx + 1] = g;
            baseline[idx + 2] = b;
            baseline[idx + 3] = 255;

            current[idx] = r;
            current[idx + 1] = g;
            current[idx + 2] = b;
            current[idx + 3] = 255;

            if x > width / 3 && x < width * 2 / 3 && y > height / 4 && y < height * 3 / 4 {
                current[idx] = r.wrapping_add(12);
                current[idx + 1] = g.wrapping_add(20);
                current[idx + 2] = b.wrapping_sub(5);
            }

            if (x * 31 + y * 37) % 97 == 0 {
                current[idx] = r.wrapping_add(3);
            }
        }
    }

    (baseline, current)
}

/// Generate a pair with only ~2% of pixels different (small header change).
/// Simulates the common VR case: minor UI change at top of a tall page.
fn make_small_diff_pair(width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    let len = width * height * 4;
    let mut baseline = vec![0u8; len];
    let mut current = vec![0u8; len];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;

            let r = ((x * 13 + y * 3) % 256) as u8;
            let g = ((x * 7 + y * 11) % 256) as u8;
            let b = ((x * 5 + y * 17) % 256) as u8;

            baseline[idx] = r;
            baseline[idx + 1] = g;
            baseline[idx + 2] = b;
            baseline[idx + 3] = 255;

            current[idx] = r;
            current[idx + 1] = g;
            current[idx + 2] = b;
            current[idx + 3] = 255;

            // Small header diff: top 5% of the image, middle 60% width
            if y < height / 20 && x > width / 5 && x < width * 4 / 5 {
                current[idx] = r.wrapping_add(30);
                current[idx + 1] = g.wrapping_add(15);
                current[idx + 2] = b.wrapping_sub(10);
            }
        }
    }

    (
        encode_png(&baseline, width, height),
        encode_png(&current, width, height),
    )
}

// -- Benchmark groups --------------------------------------------------------

/// Honeydiff-comparable: count-only, threshold=0, no AA (their exact benchmark conditions)
fn bench_count_threshold_zero(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_t0_noaa");
    let options = PixelmatchOptions {
        threshold: 0.0,
        include_aa: false,
        ..PixelmatchOptions::default()
    };

    for img in IMAGES {
        let (baseline, current) = make_screenshot_pair(img.width, img.height);
        group.bench_with_input(
            BenchmarkId::new("diff_count", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r: DiffCountOutput = diff_count_png(
                        black_box(baseline),
                        black_box(current),
                        black_box(&options),
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );
    }

    group.finish();
}

/// Production scenario: count-only, threshold=0.1, no AA
fn bench_count_default(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_default");
    let options = PixelmatchOptions::default();

    for img in IMAGES {
        let (baseline, current) = make_screenshot_pair(img.width, img.height);
        group.bench_with_input(
            BenchmarkId::new("diff_count", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r: DiffCountOutput = diff_count_png(
                        black_box(baseline),
                        black_box(current),
                        black_box(&options),
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );
    }

    group.finish();
}

/// Full diff image generation (the expensive path)
fn bench_diff_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("diff_image");
    let options = PixelmatchOptions::default();

    for img in IMAGES {
        let (baseline, current) = make_screenshot_pair(img.width, img.height);
        group.bench_with_input(
            BenchmarkId::new("diff_png", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r: DiffPngOutput =
                        diff_png(black_box(baseline), black_box(current), black_box(&options))
                            .unwrap();
                    black_box(r);
                })
            },
        );
    }

    group.finish();
}

/// SSIM only
fn bench_ssim(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssim");

    for img in IMAGES {
        let (baseline, current) = make_screenshot_pair(img.width, img.height);
        group.bench_with_input(
            BenchmarkId::new("ssim_png", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r: f64 = ssim_png(black_box(baseline), black_box(current)).unwrap();
                    black_box(r);
                })
            },
        );
    }

    group.finish();
}

/// Combined compare (diff count + SSIM, no diff image) — the PostHog VR hot path
fn bench_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare");
    let options = PixelmatchOptions::default();

    for img in IMAGES {
        let (baseline, current) = make_screenshot_pair(img.width, img.height);
        group.bench_with_input(
            BenchmarkId::new("compare_no_diff", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r: ComparePngOutput = compare_png(
                        black_box(baseline),
                        black_box(current),
                        black_box(&options),
                        false,
                        None,
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );
    }

    group.finish();
}

/// Small diff scenario (~2% change) — realistic VR "did the header break?" case
fn bench_small_diff(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_diff");
    let options = PixelmatchOptions::default();

    for img in IMAGES {
        let (baseline, current) = make_small_diff_pair(img.width, img.height);

        group.bench_with_input(
            BenchmarkId::new("count", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r: DiffCountOutput = diff_count_png(
                        black_box(baseline),
                        black_box(current),
                        black_box(&options),
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compare", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r: ComparePngOutput = compare_png(
                        black_box(baseline),
                        black_box(current),
                        black_box(&options),
                        false,
                        None,
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );
    }

    group.finish();
}

/// Identical images — fast-path performance
fn bench_identical(c: &mut Criterion) {
    let mut group = c.benchmark_group("identical");
    let options = PixelmatchOptions::default();

    for img in IMAGES {
        let (baseline, _) = make_screenshot_pair(img.width, img.height);
        let current = baseline.clone();

        group.bench_with_input(
            BenchmarkId::new("diff_count", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r: DiffCountOutput = diff_count_png(
                        black_box(baseline),
                        black_box(current),
                        black_box(&options),
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );
    }

    group.finish();
}

/// Cluster computation (mask + dilation + CCL)
fn bench_clusters(c: &mut Criterion) {
    let mut group = c.benchmark_group("clusters");
    let options = PixelmatchOptions::default();
    let cluster_opts = ClusterOptions::default();

    for img in IMAGES {
        let (baseline, current) = make_screenshot_pair(img.width, img.height);
        group.bench_with_input(
            BenchmarkId::new("diff_clusters", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r = diff_clusters_png(
                        black_box(baseline),
                        black_box(current),
                        black_box(&options),
                        black_box(&cluster_opts),
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );
    }

    // Small diff scenario — more realistic for VR
    for img in IMAGES {
        let (baseline, current) = make_small_diff_pair(img.width, img.height);
        group.bench_with_input(
            BenchmarkId::new("clusters_small_diff", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r = diff_clusters_png(
                        black_box(baseline),
                        black_box(current),
                        black_box(&options),
                        black_box(&cluster_opts),
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );
    }

    group.finish();
}

/// Early exit (max_diffs) — stops once threshold is reached
fn bench_early_exit(c: &mut Criterion) {
    let mut group = c.benchmark_group("early_exit");
    let options = PixelmatchOptions::default();

    for img in IMAGES {
        let (baseline, current) = make_screenshot_pair_rgba(img.width, img.height);

        // Bail after just 100 diffs (CI quick-fail scenario)
        group.bench_with_input(
            BenchmarkId::new("cap_100", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r = pixelmatch_count_rgba_capped(
                        black_box(baseline),
                        black_box(current),
                        img.width,
                        img.height,
                        black_box(&options),
                        100,
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );

        // Bail after 1000 diffs
        group.bench_with_input(
            BenchmarkId::new("cap_1000", img.name),
            &(&baseline, &current),
            |b, (baseline, current)| {
                b.iter(|| {
                    let r = pixelmatch_count_rgba_capped(
                        black_box(baseline),
                        black_box(current),
                        img.width,
                        img.height,
                        black_box(&options),
                        1000,
                    )
                    .unwrap();
                    black_box(r);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_count_threshold_zero,
    bench_count_default,
    bench_diff_image,
    bench_ssim,
    bench_compare,
    bench_small_diff,
    bench_identical,
    bench_clusters,
    bench_early_exit,
);
criterion_main!(benches);
