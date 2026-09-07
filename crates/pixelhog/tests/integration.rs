use approx::assert_relative_eq;
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder, ImageFormat};
use pixelhog::{
    compare_png, diff_clusters_png, diff_count_png, diff_png, diff_rgba, ssim_png, ssim_rgba,
    ClusterOptions, Comparison, Error, PixelmatchOptions, RowAlignmentOptions, ShiftBand,
    ShiftBandKind,
};

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

fn decode_png_rgba(bytes: &[u8]) -> (Vec<u8>, usize, usize) {
    let rgba = image::load_from_memory_with_format(bytes, ImageFormat::Png)
        .expect("png decode should succeed")
        .to_rgba8();
    let (w, h) = rgba.dimensions();
    let width = usize::try_from(w).expect("width should fit in usize");
    let height = usize::try_from(h).expect("height should fit in usize");
    (rgba.into_raw(), width, height)
}

#[test]
fn test_identical_images_zero_diff() {
    let baseline = solid_png(16, 12, [240, 10, 20, 255]);
    let current = baseline.clone();

    let options = PixelmatchOptions::default();
    let (diff_png, diff_count, width, height) =
        diff_png(&baseline, &current, &options).expect("pixelmatch should succeed");

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
        diff_png(&baseline, &current, &options).expect("pixelmatch should succeed");

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
        diff_png(&baseline, &current, &options).expect("pixelmatch should succeed");

    assert_eq!(diff_count, width * height / 2);
}

#[test]
fn test_different_sizes_pads_to_larger() {
    let baseline = solid_png(10, 8, [200, 0, 0, 255]);
    let current = solid_png(12, 10, [200, 0, 0, 255]);

    let options = PixelmatchOptions::default();
    let (_, diff_count, width, height) =
        diff_png(&baseline, &current, &options).expect("pixelmatch should succeed");

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

    let (_, low_count, _, _) = diff_png(&baseline, &current, &low).expect("low threshold");
    let (_, high_count, _, _) = diff_png(&baseline, &current, &high).expect("high threshold");

    assert!(low_count > high_count);
    assert_eq!(high_count, 0);
}

#[test]
fn test_diff_image_is_valid_png() {
    let baseline = solid_png(6, 6, [0, 255, 0, 255]);
    let current = solid_png(6, 6, [0, 0, 255, 255]);

    let options = PixelmatchOptions::default();
    let (diff_png, _, _, _) = diff_png(&baseline, &current, &options).expect("pixelmatch");

    let decoded = image::load_from_memory_with_format(&diff_png, ImageFormat::Png)
        .expect("diff output should be a valid PNG")
        .to_rgba8();

    assert_eq!(decoded.dimensions(), (6, 6));
}

#[test]
fn test_ssim_identical_images_score_one() {
    let baseline = solid_png(64, 64, [20, 40, 80, 255]);
    let current = baseline.clone();

    let score = ssim_png(&baseline, &current).expect("ssim should succeed");
    assert_relative_eq!(score, 1.0, epsilon = 1e-12);
}

#[test]
fn test_ssim_completely_different_images_low_score() {
    let baseline = solid_png(64, 64, [0, 0, 0, 255]);
    let current = solid_png(64, 64, [255, 255, 255, 255]);

    let score = ssim_png(&baseline, &current).expect("ssim should succeed");
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

    let score = ssim_png(&baseline, &current).expect("ssim should succeed");
    assert!(score > 0.98);
}

#[test]
fn test_ssim_small_images_below_window_size() {
    let baseline = solid_png(5, 5, [100, 100, 100, 255]);
    let current = solid_png(5, 5, [130, 130, 130, 255]);

    let score = ssim_png(&baseline, &current).expect("ssim should succeed");
    assert!((0.0..=1.0).contains(&score));
    assert!(score < 1.0);
}

#[test]
fn test_ssim_different_sizes_pads_to_larger() {
    let baseline = solid_png(9, 9, [255, 255, 255, 255]);
    let current = solid_png(14, 14, [255, 255, 255, 255]);

    let score = ssim_png(&baseline, &current).expect("ssim should succeed");

    assert!((0.0..=1.0).contains(&score));
    assert!(score < 1.0);
}

#[test]
fn test_diff_count_png_matches_diff_png() {
    let baseline = solid_png(14, 10, [20, 30, 40, 255]);
    let current = solid_png(14, 10, [40, 30, 40, 255]);

    let options = PixelmatchOptions {
        threshold: 0.05,
        ..PixelmatchOptions::default()
    };

    let (_, diff_count, width, height) = diff_png(&baseline, &current, &options).expect("diff");
    let (count_only, count_w, count_h) =
        diff_count_png(&baseline, &current, &options).expect("diff_count");

    assert_eq!((width, height), (count_w, count_h));
    assert_eq!(diff_count, count_only);
}

#[test]
fn test_compare_png_return_diff_toggle() {
    let baseline = solid_png(9, 9, [100, 100, 100, 255]);
    let current = solid_png(9, 9, [255, 20, 20, 255]);
    let options = PixelmatchOptions::default();

    let (maybe_diff, diff_count, ssim, width, height, thumb) =
        compare_png(&baseline, &current, &options, false, None).expect("compare without diff");
    assert!(maybe_diff.is_none());
    assert!(thumb.is_none());
    assert_eq!((width, height), (9, 9));
    assert!(diff_count > 0);
    assert!((0.0..=1.0).contains(&ssim));

    let (maybe_diff, diff_count_with_img, ssim_with_img, width_with_img, height_with_img, _) =
        compare_png(&baseline, &current, &options, true, None).expect("compare with diff");

    assert_eq!((width_with_img, height_with_img), (9, 9));
    assert_eq!(diff_count_with_img, diff_count);
    assert_relative_eq!(ssim_with_img, ssim, epsilon = 1e-12);
    assert!(maybe_diff.is_some());
}

#[test]
fn test_rgba_and_png_paths_match() {
    let baseline = solid_png(13, 7, [120, 20, 200, 255]);
    let current = solid_png(13, 7, [100, 20, 200, 255]);
    let options = PixelmatchOptions::default();

    let (png_diff, png_diff_count, png_w, png_h) =
        diff_png(&baseline, &current, &options).expect("png diff");
    let png_ssim = ssim_png(&baseline, &current).expect("png ssim");

    let (baseline_raw, bw, bh) = decode_png_rgba(&baseline);
    let (current_raw, cw, ch) = decode_png_rgba(&current);
    let (rgba_diff, rgba_diff_count, rgba_w, rgba_h) =
        diff_rgba(&baseline_raw, bw, bh, &current_raw, cw, ch, &options).expect("rgba diff");
    let rgba_ssim = ssim_rgba(&baseline_raw, bw, bh, &current_raw, cw, ch).expect("rgba ssim");

    assert_eq!((rgba_w, rgba_h), (png_w, png_h));
    assert_eq!(rgba_diff_count, png_diff_count);
    assert_relative_eq!(rgba_ssim, png_ssim, epsilon = 1e-12);

    let (png_raw, _, _) = decode_png_rgba(&png_diff);
    assert_eq!(png_raw, rgba_diff);
}

#[test]
fn test_clusters_two_separate_regions() {
    // 100x100 white image with two distinct colored blocks
    let width = 100;
    let height = 100;
    let baseline = solid_png(width, height, [255, 255, 255, 255]);

    // Create current with two separate 10x10 blocks of red
    let mut current_rgba = vec![255u8; width * height * 4];
    // Block 1: top-left corner (5..15, 5..15)
    for y in 5..15 {
        for x in 5..15 {
            let idx = (y * width + x) * 4;
            current_rgba[idx] = 200; // R
            current_rgba[idx + 1] = 0; // G
            current_rgba[idx + 2] = 0; // B
        }
    }
    // Block 2: bottom-right (80..90, 80..90)
    for y in 80..90 {
        for x in 80..90 {
            let idx = (y * width + x) * 4;
            current_rgba[idx] = 0;
            current_rgba[idx + 1] = 0;
            current_rgba[idx + 2] = 200;
        }
    }
    let current = encode_png(&current_rgba, width, height);

    let options = PixelmatchOptions::default();
    let raw_opts = ClusterOptions {
        min_pixels: 1,
        min_side: 0,
        dilation: 0,
        max_clusters: None,
        ..Default::default()
    };
    let (diff_count, cluster_output, w, h) =
        diff_clusters_png(&baseline, &current, &options, &raw_opts).expect("clusters");

    assert_eq!((w, h), (width, height));
    assert_eq!(diff_count, 200);
    assert_eq!(cluster_output.clusters.len(), 2);

    let mut bboxes: Vec<_> = cluster_output
        .clusters
        .iter()
        .map(|c| (c.bbox.x, c.bbox.y))
        .collect();
    bboxes.sort();
    assert_eq!(bboxes[0], (5, 5));
    assert_eq!(bboxes[1], (80, 80));
}

#[test]
fn test_clusters_count_matches_diff_count() {
    let baseline = solid_png(50, 50, [100, 100, 100, 255]);
    let current = solid_png(50, 50, [200, 100, 100, 255]);

    let options = PixelmatchOptions::default();
    let raw_opts = ClusterOptions {
        min_pixels: 1,
        min_side: 0,
        dilation: 0,
        max_clusters: None,
        ..Default::default()
    };
    let (count, _, _) = diff_count_png(&baseline, &current, &options).expect("count");
    let (cluster_count, cluster_output, _, _) =
        diff_clusters_png(&baseline, &current, &options, &raw_opts).expect("clusters");

    assert_eq!(count, cluster_count);
    assert_eq!(cluster_output.clusters.len(), 1);
    assert_eq!(cluster_output.clusters[0].pixel_count, 50 * 50);
}

#[test]
fn test_clusters_identical_images_empty() {
    let img = solid_png(20, 20, [128, 128, 128, 255]);
    let options = PixelmatchOptions::default();
    let raw_opts = ClusterOptions {
        min_pixels: 1,
        min_side: 0,
        dilation: 0,
        max_clusters: None,
        ..Default::default()
    };

    let (diff_count, cluster_output, _, _) =
        diff_clusters_png(&img, &img, &options, &raw_opts).expect("clusters");

    assert_eq!(diff_count, 0);
    assert!(cluster_output.clusters.is_empty());
}

// -- Row alignment -----------------------------------------------------------

fn rgba_from_luma(values: &[[u8; 3]]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|row| row.iter())
        .flat_map(|v| [*v, *v, *v, 255])
        .collect()
}

/// A page with a border and per-row content, so every row hashes distinctly.
fn page_rgba(width: usize, height: usize) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            let border = x < 2 || x + 2 >= width || y < 2 || y + 2 >= height;
            let pixel = if border {
                [30, 30, 30, 255]
            } else {
                [
                    ((x * 3 + y * 97) % 256) as u8,
                    ((y * 71 + 40) % 256) as u8,
                    ((x * 11 + y * 53) % 256) as u8,
                    255,
                ]
            };
            rgba[idx..idx + 4].copy_from_slice(&pixel);
        }
    }
    rgba
}

fn row_of(rgba: &[u8], width: usize, y: usize) -> &[u8] {
    &rgba[y * width * 4..(y + 1) * width * 4]
}

/// Copy `rgba` with one extra row of `color` pushed in at row `at`.
fn with_inserted_row(
    rgba: &[u8],
    width: usize,
    height: usize,
    at: usize,
    color: [u8; 4],
) -> Vec<u8> {
    let stride = width * 4;
    let mut out = Vec::with_capacity(stride * (height + 1));
    out.extend_from_slice(&rgba[..at * stride]);
    for _ in 0..width {
        out.extend_from_slice(&color);
    }
    out.extend_from_slice(&rgba[at * stride..]);
    out
}

fn paint_block(rgba: &mut [u8], width: usize, x0: usize, y0: usize, size: usize, color: [u8; 4]) {
    for y in y0..y0 + size {
        for x in x0..x0 + size {
            let idx = (y * width + x) * 4;
            rgba[idx..idx + 4].copy_from_slice(&color);
        }
    }
}

#[test]
fn test_row_alignment_single_inserted_row() {
    let (width, height, at) = (60, 80, 30);
    let baseline = page_rgba(width, height);
    let current = with_inserted_row(&baseline, width, height, at, [255, 0, 255, 255]);

    let cmp = Comparison::from_rgba(&baseline, width, height, &current, width, height + 1)
        .expect("comparison");
    let options = PixelmatchOptions::default();
    let alignment = cmp
        .row_alignment(&options, &RowAlignmentOptions::default())
        .expect("alignment");

    assert!(alignment.aligned);
    assert_eq!(alignment.inserted_rows, 1);
    assert_eq!(alignment.deleted_rows, 0);
    assert_eq!(alignment.changed_rows, 0);
    assert_eq!(alignment.residual_count, 0);
    assert_eq!(
        alignment.bands,
        vec![ShiftBand {
            y: at,
            rows: 1,
            kind: ShiftBandKind::Inserted,
        }]
    );

    // Top-aligned diffing flags everything below the inserted row.
    let naive = cmp.diff_count(&options).expect("diff count");
    assert!(naive > width * (height - at) / 2, "naive diff was {naive}");

    let diff = cmp
        .aligned_diff_image_rgba(&alignment, &options)
        .expect("aligned diff image");
    assert_eq!((diff.width, diff.height), (width, height + 1));

    for y in 0..diff.height {
        let is_band = row_of(&diff.diff_rgba, width, y)
            .chunks_exact(4)
            .all(|px| px[..3] == options.diff_color);
        assert_eq!(is_band, y == at, "row {y} band state");
    }
}

#[test]
fn test_row_alignment_single_deleted_row() {
    let (width, height, at) = (60, 80, 25);
    let current = page_rgba(width, height);
    let baseline = with_inserted_row(&current, width, height, at, [0, 255, 255, 255]);

    let cmp = Comparison::from_rgba(&baseline, width, height + 1, &current, width, height)
        .expect("comparison");
    let alignment = cmp
        .row_alignment(
            &PixelmatchOptions::default(),
            &RowAlignmentOptions::default(),
        )
        .expect("alignment");

    assert!(alignment.aligned);
    assert_eq!(alignment.deleted_rows, 1);
    assert_eq!(alignment.inserted_rows, 0);
    assert_eq!(alignment.residual_count, 0);
    assert_eq!(
        alignment.bands,
        vec![ShiftBand {
            y: at,
            rows: 1,
            kind: ShiftBandKind::Deleted,
        }]
    );
}

#[test]
fn test_row_alignment_side_by_side_drift_is_a_content_change() {
    let (width, height) = (60, 200);
    let baseline = page_rgba(width, height);
    let mut current = baseline.clone();

    // Only the right half of a 40-row block moves down one pixel.
    let (block_start, block_rows) = (40, 40);
    let stride = width * 4;
    let half = (width / 2) * 4;
    for y in (block_start + 1..=block_start + block_rows).rev() {
        let (above, row) = current.split_at_mut(y * stride);
        row[half..stride]
            .copy_from_slice(&above[(y - 1) * stride + half..(y - 1) * stride + stride]);
    }

    let cmp = Comparison::from_rgba(&baseline, width, height, &current, width, height)
        .expect("comparison");
    let options = PixelmatchOptions::default();
    let alignment = cmp
        .row_alignment(&options, &RowAlignmentOptions::default())
        .expect("alignment");

    assert!(alignment.aligned);
    assert_eq!(alignment.inserted_rows, 0);
    assert_eq!(alignment.deleted_rows, 0);
    assert_eq!(alignment.changed_rows, block_rows);
    assert!(alignment.residual_count > 0);
    assert!(alignment.bands.is_empty());
}

#[test]
fn test_row_alignment_bails_out_when_every_row_differs() {
    let (width, height) = (60, 80);
    let baseline = page_rgba(width, height);
    let mut current = vec![0u8; baseline.len()];

    // Shift the whole page one column to the right: no row hash survives.
    let stride = width * 4;
    for y in 0..height {
        let row = &baseline[y * stride..(y + 1) * stride];
        current[y * stride + 4..(y + 1) * stride].copy_from_slice(&row[..stride - 4]);
    }

    let cmp = Comparison::from_rgba(&baseline, width, height, &current, width, height)
        .expect("comparison");
    let alignment = cmp
        .row_alignment(
            &PixelmatchOptions::default(),
            &RowAlignmentOptions::default(),
        )
        .expect("alignment");

    assert!(!alignment.aligned);
    assert!(alignment.segments.is_empty());
    assert!(alignment.bands.is_empty());
    assert_eq!(alignment.residual_count, 0);
}

#[test]
fn test_aligned_clusters_ignore_the_shift_but_keep_real_changes() {
    let (width, height, at) = (60, 120, 20);
    let baseline = page_rgba(width, height);
    let shifted = with_inserted_row(&baseline, width, height, at, [255, 0, 255, 255]);

    let options = PixelmatchOptions::default();
    let cluster_options = ClusterOptions {
        min_pixels: 1,
        min_side: 0,
        dilation: 0,
        max_clusters: None,
        ..Default::default()
    };

    let cmp = Comparison::from_rgba(&baseline, width, height, &shifted, width, height + 1)
        .expect("comparison");
    let alignment = cmp
        .row_alignment(&options, &RowAlignmentOptions::default())
        .expect("alignment");
    let clusters = cmp
        .aligned_clusters(&alignment, &options, &cluster_options)
        .expect("aligned clusters");
    assert!(clusters.clusters.is_empty());

    // Same shift, plus a block that really changed.
    let mut changed = shifted.clone();
    paint_block(&mut changed, width, 20, 70, 20, [255, 0, 0, 255]);

    let cmp = Comparison::from_rgba(&baseline, width, height, &changed, width, height + 1)
        .expect("comparison");
    let alignment = cmp
        .row_alignment(&options, &RowAlignmentOptions::default())
        .expect("alignment");
    let clusters = cmp
        .aligned_clusters(&alignment, &options, &cluster_options)
        .expect("aligned clusters");

    assert_eq!(alignment.inserted_rows, 1);
    assert_eq!(clusters.clusters.len(), 1);
    let bbox = clusters.clusters[0].bbox;
    assert_eq!((bbox.x, bbox.y), (20, 70));
    assert_eq!((bbox.width, bbox.height), (20, 20));
}

#[test]
fn test_aligned_ssim_ignores_the_shift() {
    let (width, height, at) = (60, 120, 20);
    let baseline = page_rgba(width, height);
    let shifted = with_inserted_row(&baseline, width, height, at, [255, 0, 255, 255]);

    let cmp = Comparison::from_rgba(&baseline, width, height, &shifted, width, height + 1)
        .expect("comparison");
    let alignment = cmp
        .row_alignment(
            &PixelmatchOptions::default(),
            &RowAlignmentOptions::default(),
        )
        .expect("alignment");

    let plain = cmp.ssim().expect("ssim");
    let aligned = cmp.aligned_ssim(&alignment).expect("aligned ssim");

    assert!(aligned > 0.99, "aligned ssim was {aligned}");
    assert!(aligned > plain, "plain ssim was {plain}");
}

#[test]
fn test_row_alignment_ignores_anti_aliasing_at_a_replaced_row() {
    // A 3x3 slope whose middle row differs only by an anti-aliased fringe pixel.
    // Diffed on its own that row has no neighbours to judge by, so the segment
    // has to be widened with context before pixelmatch sees it.
    let baseline_rgba = rgba_from_luma(&[[100, 100, 200], [100, 128, 200], [100, 200, 200]]);
    let current_rgba = rgba_from_luma(&[[100, 100, 200], [100, 160, 200], [100, 200, 200]]);

    let cmp = Comparison::from_rgba(&baseline_rgba, 3, 3, &current_rgba, 3, 3).expect("comparison");
    let options = PixelmatchOptions::default();
    let alignment = cmp
        .row_alignment(&options, &RowAlignmentOptions::default())
        .expect("alignment");

    assert_eq!(cmp.diff_count(&options).expect("diff count"), 0);
    assert!(alignment.aligned);
    assert_eq!(alignment.changed_rows, 1);
    assert_eq!(alignment.residual_count, 0);
}

#[test]
fn test_row_alignment_rejects_a_width_change() {
    let baseline = page_rgba(60, 40);
    let current = page_rgba(50, 40);

    let cmp = Comparison::from_rgba(&baseline, 60, 40, &current, 50, 40).expect("comparison");
    let alignment = cmp
        .row_alignment(
            &PixelmatchOptions::default(),
            &RowAlignmentOptions::default(),
        )
        .expect("alignment");

    assert!(!alignment.aligned);
    assert!(alignment.segments.is_empty());
}

#[test]
fn test_aligned_calls_reject_an_alignment_from_another_pair() {
    let options = PixelmatchOptions::default();
    let small = page_rgba(60, 40);
    let large = page_rgba(60, 80);

    let small_cmp = Comparison::from_rgba(&small, 60, 40, &small, 60, 40).expect("comparison");
    let large_cmp = Comparison::from_rgba(&large, 60, 80, &large, 60, 80).expect("comparison");
    let foreign = small_cmp
        .row_alignment(&options, &RowAlignmentOptions::default())
        .expect("alignment");

    assert!(matches!(
        large_cmp.aligned_diff_image_rgba(&foreign, &options),
        Err(Error::AlignmentMismatch)
    ));
    assert!(matches!(
        large_cmp.aligned_ssim(&foreign),
        Err(Error::AlignmentMismatch)
    ));
    assert!(matches!(
        large_cmp.aligned_clusters(&foreign, &options, &ClusterOptions::default()),
        Err(Error::AlignmentMismatch)
    ));
}

#[test]
fn test_aligned_calls_reject_an_unaligned_result() {
    let options = PixelmatchOptions::default();
    let baseline = page_rgba(60, 40);
    let current = page_rgba(50, 40);

    let cmp = Comparison::from_rgba(&baseline, 60, 40, &current, 50, 40).expect("comparison");
    let alignment = cmp
        .row_alignment(&options, &RowAlignmentOptions::default())
        .expect("alignment");

    assert!(matches!(
        cmp.aligned_clusters(&alignment, &options, &ClusterOptions::default()),
        Err(Error::NotAligned)
    ));
    assert!(matches!(
        cmp.aligned_diff_image_rgba(&alignment, &options),
        Err(Error::NotAligned)
    ));
    assert!(matches!(
        cmp.aligned_ssim(&alignment),
        Err(Error::NotAligned)
    ));
}

#[test]
fn test_row_alignment_rejects_an_out_of_range_edit_ratio() {
    let page = page_rgba(20, 20);
    let cmp = Comparison::from_rgba(&page, 20, 20, &page, 20, 20).expect("comparison");

    let options = RowAlignmentOptions {
        max_edit_ratio: 1.5,
        ..RowAlignmentOptions::default()
    };

    assert!(matches!(
        cmp.row_alignment(&PixelmatchOptions::default(), &options),
        Err(Error::InvalidOption(_))
    ));
}

#[test]
fn test_rows_deleted_from_the_bottom_leave_the_seam_past_the_last_row() {
    let (width, height) = (12, 40);
    let baseline = page_rgba(width, height);
    let current = baseline[..(height - 3) * width * 4].to_vec();

    let options = PixelmatchOptions::default();
    let cmp = Comparison::from_rgba(&baseline, width, height, &current, width, height - 3)
        .expect("comparison");
    let alignment = cmp
        .row_alignment(&options, &RowAlignmentOptions::default())
        .expect("alignment");

    assert!(alignment.aligned);
    assert_eq!(
        alignment.bands,
        vec![ShiftBand {
            y: height - 3,
            rows: 3,
            kind: ShiftBandKind::Deleted,
        }]
    );

    // The seam has no row of its own, so nothing in the image is painted for it.
    let diff = cmp
        .aligned_diff_image_rgba(&alignment, &options)
        .expect("aligned diff image");
    assert_eq!((diff.width, diff.height), (width, height - 3));
    let last_row = row_of(&diff.diff_rgba, width, height - 4);
    assert!(last_row
        .chunks_exact(4)
        .all(|px| px[..3] != options.diff_color));
}

fn solid_rgba(width: usize, height: usize, color: [u8; 4]) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];
    for px in rgba.chunks_exact_mut(4) {
        px.copy_from_slice(&color);
    }
    rgba
}

#[test]
fn test_aligned_calls_reject_a_same_size_alignment_from_another_pair() {
    let options = PixelmatchOptions::default();
    let (width, height) = (20, 20);

    let page = page_rgba(width, height);
    let identical =
        Comparison::from_rgba(&page, width, height, &page, width, height).expect("comparison");
    let foreign = identical
        .row_alignment(&options, &RowAlignmentOptions::default())
        .expect("alignment");

    let black = solid_rgba(width, height, [0, 0, 0, 255]);
    let white = solid_rgba(width, height, [255, 255, 255, 255]);
    let other =
        Comparison::from_rgba(&black, width, height, &white, width, height).expect("comparison");

    assert!(matches!(
        other.aligned_diff_image_rgba(&foreign, &options),
        Err(Error::AlignmentMismatch)
    ));
    assert!(matches!(
        other.aligned_ssim(&foreign),
        Err(Error::AlignmentMismatch)
    ));
    assert!(matches!(
        other.aligned_clusters(&foreign, &options, &ClusterOptions::default()),
        Err(Error::AlignmentMismatch)
    ));
}

#[test]
fn test_aligned_clusters_rejects_an_invalid_threshold() {
    let (width, height, at) = (60, 40, 20);
    let baseline = page_rgba(width, height);
    let current = with_inserted_row(&baseline, width, height, at, [255, 0, 255, 255]);

    let cmp = Comparison::from_rgba(&baseline, width, height, &current, width, height + 1)
        .expect("comparison");
    let alignment = cmp
        .row_alignment(
            &PixelmatchOptions::default(),
            &RowAlignmentOptions::default(),
        )
        .expect("alignment");

    // A pure shift has no Replace segments, so nothing else would look at the options.
    assert_eq!(alignment.changed_rows, 0);

    let invalid = PixelmatchOptions {
        threshold: 2.0,
        ..PixelmatchOptions::default()
    };
    assert!(matches!(
        cmp.aligned_clusters(&alignment, &invalid, &ClusterOptions::default()),
        Err(Error::InvalidOption(_))
    ));
}
