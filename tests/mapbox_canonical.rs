use image::ImageFormat;
use pixelhog::{pixelmatch_png, PixelmatchOptions};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct CanonicalCase {
    left: &'static str,
    right: &'static str,
    expected_diff: &'static str,
    expected_mismatch: usize,
    options: PixelmatchOptions,
}

#[test]
fn mapbox_canonical_fixture_suite() {
    let fixtures = fixtures_dir();

    let cases = vec![
        CanonicalCase {
            left: "1a",
            right: "1b",
            expected_diff: "1diff",
            expected_mismatch: 143,
            options: PixelmatchOptions {
                threshold: 0.05,
                ..PixelmatchOptions::default()
            },
        },
        CanonicalCase {
            left: "1a",
            right: "1b",
            expected_diff: "1diffdefaultthreshold",
            expected_mismatch: 106,
            options: PixelmatchOptions::default(),
        },
        CanonicalCase {
            left: "2a",
            right: "2b",
            expected_diff: "2diff",
            expected_mismatch: 12_437,
            options: PixelmatchOptions {
                threshold: 0.05,
                alpha: 0.5,
                diff_color: [255, 0, 255],
                aa_color: [0, 192, 0],
                ..PixelmatchOptions::default()
            },
        },
        CanonicalCase {
            left: "3a",
            right: "3b",
            expected_diff: "3diff",
            expected_mismatch: 212,
            options: PixelmatchOptions {
                threshold: 0.05,
                ..PixelmatchOptions::default()
            },
        },
        CanonicalCase {
            left: "4a",
            right: "4b",
            expected_diff: "4diff",
            expected_mismatch: 36_049,
            options: PixelmatchOptions {
                threshold: 0.05,
                ..PixelmatchOptions::default()
            },
        },
        CanonicalCase {
            left: "5a",
            right: "5b",
            expected_diff: "5diff",
            expected_mismatch: 6,
            options: PixelmatchOptions {
                threshold: 0.05,
                ..PixelmatchOptions::default()
            },
        },
        CanonicalCase {
            left: "6a",
            right: "6b",
            expected_diff: "6diff",
            expected_mismatch: 51,
            options: PixelmatchOptions {
                threshold: 0.05,
                ..PixelmatchOptions::default()
            },
        },
        CanonicalCase {
            left: "7a",
            right: "7b",
            expected_diff: "7diff",
            expected_mismatch: 2_448,
            options: PixelmatchOptions {
                diff_color_alt: Some([0, 255, 0]),
                ..PixelmatchOptions::default()
            },
        },
        CanonicalCase {
            left: "8a",
            right: "5b",
            expected_diff: "8diff",
            expected_mismatch: 32_896,
            options: PixelmatchOptions {
                threshold: 0.05,
                ..PixelmatchOptions::default()
            },
        },
    ];

    for case in cases {
        let left_png = read_fixture(&fixtures, case.left);
        let right_png = read_fixture(&fixtures, case.right);
        let expected_diff_png = read_fixture(&fixtures, case.expected_diff);

        let (actual_diff_png, mismatch, width, height) =
            pixelmatch_png(&left_png, &right_png, &case.options)
                .unwrap_or_else(|err| panic!("case {:?} failed to run: {err}", case));

        assert_eq!(
            mismatch, case.expected_mismatch,
            "unexpected mismatch count for case {:?}",
            case
        );

        let (expected_raw, expected_width, expected_height) = decode_png_rgba(&expected_diff_png);
        let (actual_raw, actual_width, actual_height) = decode_png_rgba(&actual_diff_png);

        assert_eq!(
            (width, height),
            (expected_width, expected_height),
            "returned dimensions did not match expected fixture dimensions for case {:?}",
            case
        );
        assert_eq!(
            (actual_width, actual_height),
            (expected_width, expected_height),
            "encoded PNG dimensions did not match expected fixture dimensions for case {:?}",
            case
        );

        // Canonical check: decoded RGBA bytes must exactly match Mapbox golden fixtures.
        assert_eq!(
            actual_raw, expected_raw,
            "diff image bytes differed from Mapbox golden fixture for case {:?}",
            case
        );
    }
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mapbox")
}

fn read_fixture(fixtures: &Path, stem: &str) -> Vec<u8> {
    let path = fixtures.join(format!("{stem}.png"));
    fs::read(&path).unwrap_or_else(|err| panic!("failed to read fixture {}: {err}", path.display()))
}

fn decode_png_rgba(png: &[u8]) -> (Vec<u8>, usize, usize) {
    let rgba = image::load_from_memory_with_format(png, ImageFormat::Png)
        .expect("fixture should decode")
        .to_rgba8();
    let (width, height) = rgba.dimensions();
    let width = usize::try_from(width).expect("width should fit in usize");
    let height = usize::try_from(height).expect("height should fit in usize");
    (rgba.into_raw(), width, height)
}
