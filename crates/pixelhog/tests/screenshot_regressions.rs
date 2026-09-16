//! Real screenshot pairs that row alignment once showed wrong.
//!
//! The synthetic tests pin the mechanism. These pin what a reviewer saw on a
//! real page. Each pair is a crop of a Storybook screenshot from PostHog's
//! visual review, large enough that the edits fit the alignment budget.

use pixelhog::{
    ClusterOptions, Comparison, PixelmatchOptions, RowAlignmentOptions, ShiftBand, ShiftBandKind,
};
use std::fs;
use std::ops::Range;
use std::path::Path;

struct ScreenshotCase {
    /// Stem of the `-before.png` and `-after.png` pair under `tests/fixtures/screenshots/`.
    name: &'static str,
    /// Inserted rows, deleted rows and residual pixels. Callers threshold on
    /// these, so a change to what the diff image shows must not move them.
    counts: (usize, usize, usize),
    bands: Vec<ShiftBand>,
    /// Rows of the current image where the diff image and a cluster have to
    /// show the change, or `None` when the pair only moved and must have no
    /// cluster at all.
    changed_rows: Option<Range<usize>>,
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("screenshots")
        .join(format!("{name}.png"));
    fs::read(&path).unwrap_or_else(|err| panic!("failed to read fixture {}: {err}", path.display()))
}

#[test]
fn screenshot_regression_suite() {
    let cases = vec![
        // Story `scenes-app-engineering-analytics-author--author--light`: the
        // lead time box plots turned from vertical to horizontal and the card
        // lost 80 rows. Myers matched the blank card rows inside the chart, so
        // the diff image showed three inserted bands and four seams, and no
        // change inside the chart.
        ScreenshotCase {
            name: "lead-time-card",
            counts: (65, 145, 0),
            bands: vec![ShiftBand {
                y: 350,
                rows: 80,
                kind: ShiftBandKind::Deleted,
            }],
            changed_rows: Some(238..350),
        },
        // Story `mcp-apps-actions--list--light`: only the pagination "1" moved
        // inside its button. Its rows were split around one blank row into a
        // ten-row shift.
        ScreenshotCase {
            name: "pagination-page-number",
            counts: (10, 10, 0),
            bands: vec![],
            changed_rows: Some(215..229),
        },
        // Story `scenes-app-settings-environment--settings-environment-heatmaps--dark`:
        // a sidebar label changed in place ("Task agents" to "Model
        // preferences") and read as a thirteen-row shift.
        ScreenshotCase {
            name: "settings-nav-label",
            counts: (13, 13, 326),
            bands: vec![],
            changed_rows: Some(213..227),
        },
        // Story `errortracking-exceptioncard--exception-card-header-widths-with-action--dark`:
        // headings that re-rendered a pixel lower drew one-row bands across
        // the card where nothing moved. The heading change itself is drawn
        // either way; the bands are what this case guards.
        ScreenshotCase {
            name: "card-heading-drift",
            counts: (3, 3, 1965),
            bands: vec![],
            changed_rows: Some(56..68),
        },
        // Story `scenes-app-engineering-analytics-workflows--workflow-list--light`:
        // a panel grew by one row and nothing else changed. This is the shift
        // row alignment is for, and it has to stay one inserted row with no
        // cluster.
        ScreenshotCase {
            name: "panel-grew-one-row",
            counts: (1, 0, 0),
            bands: vec![ShiftBand {
                y: 108,
                rows: 1,
                kind: ShiftBandKind::Inserted,
            }],
            changed_rows: None,
        },
    ];

    let seam_color = [0, 0, 255];
    let options = PixelmatchOptions {
        diff_color_alt: Some(seam_color),
        ..Default::default()
    };

    for case in cases {
        let before = read_fixture(&format!("{}-before", case.name));
        let after = read_fixture(&format!("{}-after", case.name));
        let cmp = Comparison::from_png(&before, &after).expect("comparison");
        let alignment = cmp
            .row_alignment(&options, &RowAlignmentOptions::default())
            .expect("alignment");

        assert!(alignment.aligned, "{}: not aligned", case.name);
        assert_eq!(
            (
                alignment.inserted_rows,
                alignment.deleted_rows,
                alignment.residual_count
            ),
            case.counts,
            "{}: counts",
            case.name
        );
        assert_eq!(alignment.bands, case.bands, "{}: bands", case.name);

        // Only the bands are drawn as solid rows: a filled row per inserted
        // row, and one seam row per deleted band.
        let diff = cmp
            .aligned_diff_image_rgba(&alignment, &options)
            .expect("aligned diff image");
        let filled_with = |color: [u8; 3]| {
            diff.diff_rgba
                .chunks_exact(diff.width * 4)
                .filter(|row| row.chunks_exact(4).all(|px| px[..3] == color))
                .count()
        };
        let inserted: usize = case
            .bands
            .iter()
            .filter(|band| band.kind == ShiftBandKind::Inserted)
            .map(|band| band.rows)
            .sum();
        let seams = case
            .bands
            .iter()
            .filter(|band| band.kind == ShiftBandKind::Deleted && band.y < diff.height)
            .count();
        assert_eq!(
            filled_with(options.diff_color),
            inserted,
            "{}: filled rows",
            case.name
        );
        assert_eq!(filled_with(seam_color), seams, "{}: seam rows", case.name);

        let clusters = cmp
            .aligned_clusters(&alignment, &options, &ClusterOptions::default())
            .expect("aligned clusters");
        match &case.changed_rows {
            Some(rows) => {
                let stride = diff.width * 4;
                assert!(
                    diff.diff_rgba[rows.start * stride..rows.end * stride]
                        .chunks_exact(4)
                        .any(|px| px[..3] == options.diff_color || px[..3] == seam_color),
                    "{}: no changed pixels drawn in rows {rows:?}",
                    case.name
                );
                assert!(
                    clusters.clusters.iter().any(|cluster| {
                        let top = cluster.bbox.y;
                        top < rows.end && top + cluster.bbox.height > rows.start
                    }),
                    "{}: no cluster over rows {rows:?}",
                    case.name
                );
            }
            None => assert!(
                clusters.clusters.is_empty(),
                "{}: clusters where only rows moved",
                case.name
            ),
        }
    }
}
