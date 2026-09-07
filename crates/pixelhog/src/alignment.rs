//! Row alignment for image pairs that differ by a vertical shift.
//!
//! A one-pixel panel growth or an inserted banner moves everything below it
//! down. A top-aligned pixel diff then flags most of the page and SSIM reports
//! large dissimilarity, even though nothing else changed. This module hashes
//! each pixel row and runs a Myers diff over the hashes, which separates rows
//! that only moved from rows whose content really changed.
//!
//! Two guards shape the result. The edit budget bounds the O(ND) walk: a pair
//! too different to align gives up instead of running to completion. Replace
//! pairing turns an adjacent delete and insert back into a content change, so
//! anti-aliasing jitter on a text row is not reported as a shift.
//!
//! Rows repeated across the page (blank background, rules, spacers) need no
//! special handling. Myers minimizes edits, so it matches them by position:
//! pairing the nth blank row with the nth blank row costs nothing, while any
//! other pairing costs a delete and an insert.
//!
//! Alignment is vertical only. Two images of different widths have no row in
//! common by construction, so a width change disqualifies the pair.

use crate::clusters::{compute_clusters, ClusterOptions, ClustersOutput};
use crate::image_utils::encode_png;
use crate::pixelmatch::{
    draw_pixel, gray_pixel_value, pixelmatch_mask_rgba, pixelmatch_rgba, validate_options,
    PixelmatchOptions, PixelmatchOutput,
};
use crate::ssim::compute_ssim_rgba;
use crate::Error;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

/// Options for row alignment.
pub struct RowAlignmentOptions {
    /// Edit budget as a fraction of the combined row count.
    pub max_edit_ratio: f64,
    /// Hard cap on the edit budget, whatever the image height.
    pub max_edit_rows: usize,
}

impl Default for RowAlignmentOptions {
    fn default() -> Self {
        Self {
            max_edit_ratio: 0.25,
            max_edit_rows: 2048,
        }
    }
}

/// How a run of rows relates the two images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSegmentKind {
    /// Rows matched by hash — same content, possibly at a different y.
    Equal,
    /// Rows present in both images but with different content.
    Replace,
    /// Rows present only in the current image.
    Insert,
    /// Rows present only in the baseline image.
    Delete,
}

/// A run of consecutive rows sharing one [`RowSegmentKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowSegment {
    pub kind: RowSegmentKind,
    pub baseline_start: usize,
    pub current_start: usize,
    pub len: usize,
}

/// Whether a band of rows was added or removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftBandKind {
    Inserted,
    Deleted,
}

/// A band of rows that shifted the content below it, in current-image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShiftBand {
    /// Top row of an inserted band, or the seam row for a deleted one. Rows
    /// removed from the bottom of the page leave their seam past the last row,
    /// so this can equal the current image height.
    pub y: usize,
    pub rows: usize,
    pub kind: ShiftBandKind,
}

/// Result of aligning the rows of two images.
#[derive(Debug, Clone)]
pub struct RowAlignment {
    /// False when the pair could not be aligned; every other field is then zero or empty.
    pub aligned: bool,
    /// Myers edit distance over the row hashes.
    pub edit_distance: usize,
    pub inserted_rows: usize,
    pub deleted_rows: usize,
    /// Rows in `Replace` segments — content that changed rather than moved.
    pub changed_rows: usize,
    /// Differing pixels inside `Replace` segments, per the pixelmatch options.
    pub residual_count: usize,
    pub segments: Vec<RowSegment>,
    pub bands: Vec<ShiftBand>,
    // Row indices only mean something for the pair they were computed from, so
    // the aligned_* calls refuse an alignment from a differently sized pair.
    width: usize,
    baseline_height: usize,
    current_height: usize,
}

impl RowAlignment {
    fn unaligned(images: &RowImages) -> Self {
        Self {
            aligned: false,
            edit_distance: 0,
            inserted_rows: 0,
            deleted_rows: 0,
            changed_rows: 0,
            residual_count: 0,
            segments: Vec::new(),
            bands: Vec::new(),
            width: images.width,
            baseline_height: images.baseline_height,
            current_height: images.current_height,
        }
    }
}

/// The image pair as rows over a shared stride.
///
/// `width` is the padded width both buffers were stored at, so a row is always
/// `width * 4` bytes. The unpadded widths decide whether the pair can be
/// aligned at all; once it can, all three are the same number.
pub struct RowImages<'a> {
    pub baseline_rgba: &'a [u8],
    pub current_rgba: &'a [u8],
    pub width: usize,
    pub baseline_width: usize,
    pub baseline_height: usize,
    pub current_width: usize,
    pub current_height: usize,
}

/// Hash a row by the pixels it renders as.
///
/// A fully transparent pixel reads as zero whatever its RGB, which is how
/// pixelmatch sees it: two of them never differ, so two rows that differ only
/// under full transparency are the same row.
fn hash_row(row: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Screenshots are opaque almost everywhere, and hashing a row in one go is
    // several times faster than feeding it a pixel at a time. Whether a row has
    // a transparent pixel is a property of that row, so both images take the
    // same path for the same content.
    if row.chunks_exact(4).any(|pixel| pixel[3] == 0) {
        for pixel in row.chunks_exact(4) {
            let rendered = if pixel[3] == 0 {
                0
            } else {
                u32::from_ne_bytes([pixel[0], pixel[1], pixel[2], pixel[3]])
            };
            hasher.write_u32(rendered);
        }
    } else {
        hasher.write(row);
    }

    hasher.finish()
}

/// Hash the first `rows` rows of a padded buffer. Padding below them is skipped.
fn hash_rows(rgba: &[u8], stride: usize, rows: usize) -> Vec<u64> {
    if stride == 0 {
        return Vec::new();
    }

    rgba[..rows * stride]
        .par_chunks_exact(stride)
        .map(hash_row)
        .collect()
}

/// One row-level edit, before consecutive edits are merged into runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditOp {
    Equal,
    Insert,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EditRun {
    op: EditOp,
    baseline_start: usize,
    current_start: usize,
    len: usize,
}

fn coalesce_ops(ops: &[(EditOp, usize, usize)]) -> Vec<EditRun> {
    let mut runs: Vec<EditRun> = Vec::new();
    for (op, baseline_index, current_index) in ops {
        match runs.last_mut() {
            Some(run) if run.op == *op => run.len += 1,
            _ => runs.push(EditRun {
                op: *op,
                baseline_start: *baseline_index,
                current_start: *current_index,
                len: 1,
            }),
        }
    }
    runs
}

/// Walk the recorded V arrays back from `(n, m)` to the origin.
fn backtrack(trace: &[Vec<isize>], n: isize, m: isize) -> Vec<EditRun> {
    let mut ops: Vec<(EditOp, usize, usize)> = Vec::new();
    let mut x = n;
    let mut y = m;

    for (d, v) in trace.iter().enumerate().rev() {
        let d = d as isize;
        let k = x - y;
        // Trace entries hold V restricted to `-d..=d`, so k maps to index k + d.
        let at = |k: isize| v[(k + d) as usize];

        let (prev_x, prev_y) = if d == 0 {
            (0, 0)
        } else {
            let prev_k = if k == -d || (k != d && at(k - 1) < at(k + 1)) {
                k + 1
            } else {
                k - 1
            };
            let prev_x = at(prev_k);
            (prev_x, prev_x - prev_k)
        };

        while x > prev_x && y > prev_y {
            x -= 1;
            y -= 1;
            ops.push((EditOp::Equal, x as usize, y as usize));
        }

        if d == 0 {
            break;
        }

        if x > prev_x {
            x -= 1;
            ops.push((EditOp::Delete, x as usize, y as usize));
        } else if y > prev_y {
            y -= 1;
            ops.push((EditOp::Insert, x as usize, y as usize));
        }
    }

    ops.reverse();
    coalesce_ops(&ops)
}

/// Myers O(ND) diff over row tokens, returning `None` once D would exceed `max_d`.
///
/// One V array is kept per d so memory tracks the edit distance actually found,
/// not the budget.
fn myers_diff(baseline: &[u64], current: &[u64], max_d: usize) -> Option<(usize, Vec<EditRun>)> {
    let n = baseline.len() as isize;
    let m = current.len() as isize;
    let offset = max_d as isize + 1;
    let mut v = vec![0isize; 2 * (max_d + 1) + 1];
    let mut trace: Vec<Vec<isize>> = Vec::new();

    for d in 0..=max_d as isize {
        trace.push(v[(offset - d) as usize..=(offset + d) as usize].to_vec());

        let mut k = -d;
        while k <= d {
            let down =
                k == -d || (k != d && v[(offset + k - 1) as usize] < v[(offset + k + 1) as usize]);
            let mut x = if down {
                v[(offset + k + 1) as usize]
            } else {
                v[(offset + k - 1) as usize] + 1
            };
            let mut y = x - k;

            while x < n && y < m && baseline[x as usize] == current[y as usize] {
                x += 1;
                y += 1;
            }

            v[(offset + k) as usize] = x;
            if x >= n && y >= m {
                return Some((d as usize, backtrack(&trace, n, m)));
            }

            k += 2;
        }
    }

    None
}

/// The hash every row in the range shares, or `None` when they differ.
fn uniform_hash(hashes: &[u64], start: usize, len: usize) -> Option<u64> {
    let first = *hashes.get(start)?;
    hashes[start..start + len]
        .iter()
        .all(|hash| *hash == first)
        .then_some(first)
}

/// True when both ranges are one repeated row and it is the same row.
fn interchangeable(
    hashes: &[u64],
    a_start: usize,
    a_len: usize,
    b_start: usize,
    b_len: usize,
) -> bool {
    match (
        uniform_hash(hashes, a_start, a_len),
        uniform_hash(hashes, b_start, b_len),
    ) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// Slide an edit across a run of rows it cannot be told apart from.
///
/// Inside a block of identical rows Myers may put the delete at one end and the
/// insert at the other, at the same cost. A replaced line in a uniform block
/// then reads as a two-row shift with no residual, hiding the real change.
/// Moving the second edit to the near side of the block puts the two back
/// together, and `pair_replacements` turns them into the `Replace` they are.
fn slide_edits_together(runs: &mut [EditRun], baseline_hashes: &[u64], current_hashes: &[u64]) {
    for index in 0..runs.len().saturating_sub(2) {
        let (first, middle, last) = (runs[index], runs[index + 1], runs[index + 2]);
        if middle.op != EditOp::Equal {
            continue;
        }

        let slidable = match (first.op, last.op) {
            (EditOp::Insert, EditOp::Delete) => interchangeable(
                baseline_hashes,
                middle.baseline_start,
                middle.len,
                last.baseline_start,
                last.len,
            ),
            (EditOp::Delete, EditOp::Insert) => interchangeable(
                current_hashes,
                middle.current_start,
                middle.len,
                last.current_start,
                last.len,
            ),
            _ => false,
        };

        if !slidable {
            continue;
        }

        // The edit takes over where the equal run began, and the equal run
        // moves down by however many rows the edit consumes.
        runs[index + 1] = EditRun {
            baseline_start: middle.baseline_start,
            current_start: middle.current_start,
            ..last
        };
        runs[index + 2] = EditRun {
            baseline_start: middle.baseline_start
                + usize::from(last.op == EditOp::Delete) * last.len,
            current_start: middle.current_start + usize::from(last.op == EditOp::Insert) * last.len,
            ..middle
        };
    }
}

fn segment(
    kind: RowSegmentKind,
    baseline_start: usize,
    current_start: usize,
    len: usize,
) -> RowSegment {
    RowSegment {
        kind,
        baseline_start,
        current_start,
        len,
    }
}

/// Pair an adjacent delete/insert couple into a `Replace`, leaving the surplus.
///
/// Text that only jitters by an anti-aliased pixel deletes one row and inserts
/// one row at the same place. Pairing them keeps that a content change instead
/// of reporting a shift that never happened.
fn pair_replacements(runs: &[EditRun]) -> Vec<RowSegment> {
    let mut segments = Vec::with_capacity(runs.len());
    let mut index = 0;

    while index < runs.len() {
        let run = runs[index];
        let next = runs.get(index + 1).copied();

        let pair = match (run.op, next) {
            (EditOp::Delete, Some(next)) if next.op == EditOp::Insert => {
                Some((run.len, next.len, next))
            }
            (EditOp::Insert, Some(next)) if next.op == EditOp::Delete => {
                Some((next.len, run.len, next))
            }
            _ => None,
        };

        let Some((delete_len, insert_len, next)) = pair else {
            segments.push(segment(
                match run.op {
                    EditOp::Equal => RowSegmentKind::Equal,
                    EditOp::Insert => RowSegmentKind::Insert,
                    EditOp::Delete => RowSegmentKind::Delete,
                },
                run.baseline_start,
                run.current_start,
                run.len,
            ));
            index += 1;
            continue;
        };

        let baseline_start = run.baseline_start.min(next.baseline_start);
        let current_start = run.current_start.min(next.current_start);
        let paired = delete_len.min(insert_len);

        segments.push(segment(
            RowSegmentKind::Replace,
            baseline_start,
            current_start,
            paired,
        ));

        if delete_len > paired {
            segments.push(segment(
                RowSegmentKind::Delete,
                baseline_start + paired,
                current_start + paired,
                delete_len - paired,
            ));
        } else if insert_len > paired {
            segments.push(segment(
                RowSegmentKind::Insert,
                baseline_start + paired,
                current_start + paired,
                insert_len - paired,
            ));
        }

        index += 2;
    }

    segments
}

/// Bands sit in current-image coordinates so they overlay the current screenshot.
fn shift_bands(segments: &[RowSegment]) -> Vec<ShiftBand> {
    segments
        .iter()
        .filter_map(|seg| match seg.kind {
            RowSegmentKind::Insert => Some(ShiftBand {
                y: seg.current_start,
                rows: seg.len,
                kind: ShiftBandKind::Inserted,
            }),
            // Deleted rows have no current-image extent — mark the seam they left behind.
            RowSegmentKind::Delete => Some(ShiftBand {
                y: seg.current_start,
                rows: seg.len,
                kind: ShiftBandKind::Deleted,
            }),
            _ => None,
        })
        .collect()
}

fn row_range(rgba: &[u8], stride: usize, start: usize, len: usize) -> &[u8] {
    &rgba[start * stride..(start + len) * stride]
}

/// A `Replace` segment widened by one row of context above and below.
///
/// pixelmatch decides whether a pixel is anti-aliased from its neighbours, so a
/// segment diffed on its own has none beyond its edges and reports the fringe of
/// a text row as a real change. Context rows are only there to be looked at —
/// callers take the segment's own rows back out at `lead`.
struct SegmentWindow<'a> {
    baseline: &'a [u8],
    current: &'a [u8],
    rows: usize,
    lead: usize,
}

fn segment_window<'a>(images: &RowImages<'a>, seg: &RowSegment) -> SegmentWindow<'a> {
    let stride = images.width * 4;
    let lead = usize::from(seg.baseline_start > 0 && seg.current_start > 0);
    let trail = usize::from(
        seg.baseline_start + seg.len < images.baseline_height
            && seg.current_start + seg.len < images.current_height,
    );
    let rows = lead + seg.len + trail;

    SegmentWindow {
        baseline: row_range(
            images.baseline_rgba,
            stride,
            seg.baseline_start - lead,
            rows,
        ),
        current: row_range(images.current_rgba, stride, seg.current_start - lead, rows),
        rows,
        lead,
    }
}

/// The diff mask for a segment's own rows, judged with its context.
fn segment_mask(
    images: &RowImages,
    seg: &RowSegment,
    pixel_options: &PixelmatchOptions,
) -> Result<Vec<bool>, Error> {
    let window = segment_window(images, seg);
    let output = pixelmatch_mask_rgba(
        window.baseline,
        window.current,
        images.width,
        window.rows,
        pixel_options,
    )?;

    let start = window.lead * images.width;
    Ok(output.diff_mask[start..start + seg.len * images.width].to_vec())
}

fn mask_count(mask: &[bool]) -> usize {
    mask.iter().filter(|differs| **differs).count()
}

fn replace_segments(segments: &[RowSegment]) -> impl Iterator<Item = &RowSegment> {
    segments
        .iter()
        .filter(|seg| seg.kind == RowSegmentKind::Replace)
}

/// Count differing pixels inside the `Replace` segments only.
fn residual_count(
    images: &RowImages,
    segments: &[RowSegment],
    pixel_options: &PixelmatchOptions,
) -> Result<usize, Error> {
    let mut total = 0;
    for seg in replace_segments(segments) {
        total += mask_count(&segment_mask(images, seg, pixel_options)?);
    }
    Ok(total)
}

fn validate_alignment_options(options: &RowAlignmentOptions) -> Result<(), Error> {
    if !options.max_edit_ratio.is_finite() || !(0.0..=1.0).contains(&options.max_edit_ratio) {
        return Err(Error::InvalidOption(
            "max_edit_ratio must be in the range [0.0, 1.0]",
        ));
    }

    Ok(())
}

/// Align the rows of the two images.
///
/// Alignment is vertical only: a pair whose unpadded widths differ shares no
/// row and comes back unaligned without being hashed.
pub fn compute_row_alignment(
    images: &RowImages,
    pixel_options: &PixelmatchOptions,
    options: &RowAlignmentOptions,
) -> Result<RowAlignment, Error> {
    validate_options(pixel_options)?;
    validate_alignment_options(options)?;

    if images.baseline_width != images.current_width {
        return Ok(RowAlignment::unaligned(images));
    }

    let stride = images.width * 4;
    let baseline_hashes = hash_rows(images.baseline_rgba, stride, images.baseline_height);
    let current_hashes = hash_rows(images.current_rgba, stride, images.current_height);

    let total_rows = baseline_hashes.len() + current_hashes.len();
    let budget = (options.max_edit_ratio * total_rows as f64).ceil() as usize;
    // D can never exceed the combined row count, and the V array is sized from
    // max_d, so this also keeps a large max_edit_rows from allocating for edits
    // that cannot happen.
    let max_d = budget.min(options.max_edit_rows).min(total_rows);

    let Some((edit_distance, mut runs)) = myers_diff(&baseline_hashes, &current_hashes, max_d)
    else {
        return Ok(RowAlignment::unaligned(images));
    };

    slide_edits_together(&mut runs, &baseline_hashes, &current_hashes);
    let segments = pair_replacements(&runs);
    let bands = shift_bands(&segments);

    let rows_of = |kind: RowSegmentKind| -> usize {
        segments
            .iter()
            .filter(|s| s.kind == kind)
            .map(|s| s.len)
            .sum()
    };

    Ok(RowAlignment {
        aligned: true,
        edit_distance,
        inserted_rows: rows_of(RowSegmentKind::Insert),
        deleted_rows: rows_of(RowSegmentKind::Delete),
        changed_rows: rows_of(RowSegmentKind::Replace),
        residual_count: residual_count(images, &segments, pixel_options)?,
        segments,
        bands,
        width: images.width,
        baseline_height: images.baseline_height,
        current_height: images.current_height,
    })
}

/// An alignment may only be used with the pair it was computed from.
fn check_alignment(images: &RowImages, alignment: &RowAlignment) -> Result<(), Error> {
    if !alignment.aligned {
        return Err(Error::NotAligned);
    }

    if alignment.width != images.width
        || alignment.baseline_height != images.baseline_height
        || alignment.current_height != images.current_height
    {
        return Err(Error::AlignmentMismatch);
    }

    Ok(())
}

/// Cluster the residual — the pixels that differ inside `Replace` segments.
///
/// The mask is in current-image coordinates. Shift bands are not in it: the
/// bands are the shift signal and the mask is the residual, and a caller decides
/// separately whether to absorb a shift or flag it, reading
/// [`RowAlignment::bands`] for that.
pub fn aligned_clusters(
    images: &RowImages,
    alignment: &RowAlignment,
    pixel_options: &PixelmatchOptions,
    cluster_options: &ClusterOptions,
) -> Result<ClustersOutput, Error> {
    check_alignment(images, alignment)?;

    let width = images.width;
    let mut mask = vec![false; width * images.current_height];

    for seg in replace_segments(&alignment.segments) {
        let segment_mask = segment_mask(images, seg, pixel_options)?;
        let start = seg.current_start * width;
        mask[start..start + seg.len * width].copy_from_slice(&segment_mask);
    }

    Ok(compute_clusters(
        &mask,
        width,
        images.current_height,
        cluster_options,
    ))
}

fn draw_band(diff: &mut [u8], width: usize, height: usize, band: &ShiftBand, color: [u8; 3]) {
    let rows = match band.kind {
        // A deleted band has no current-image height — one seam row marks it.
        ShiftBandKind::Deleted => 1,
        ShiftBandKind::Inserted => band.rows,
    };
    let stride = width * 4;

    // Rows removed from the bottom leave their seam past the last row.
    for y in band.y..(band.y + rows).min(height) {
        for x in 0..width {
            draw_pixel(diff, y * stride + x * 4, color[0], color[1], color[2]);
        }
    }
}

/// Build the diff image in current-image coordinates.
///
/// Equal rows are grayed like pixelmatch does, `Replace` rows carry the usual
/// pixelmatch coloring, inserted bands are filled with `diff_color`, and a
/// deleted band is one seam row in `diff_color_alt` when set.
pub fn aligned_diff_image_rgba(
    images: &RowImages,
    alignment: &RowAlignment,
    pixel_options: &PixelmatchOptions,
) -> Result<PixelmatchOutput, Error> {
    check_alignment(images, alignment)?;
    validate_options(pixel_options)?;

    let width = images.width;
    let height = images.current_height;
    let stride = width * 4;
    let mut diff = vec![0u8; stride * height];

    // Start from the grayed current image; changed rows and bands paint over it.
    for y in 0..height {
        for x in 0..width {
            let value =
                gray_pixel_value(images.current_rgba, y * stride + x * 4, pixel_options.alpha);
            draw_pixel(&mut diff, y * stride + x * 4, value, value, value);
        }
    }

    let mut diff_count = 0;
    for seg in replace_segments(&alignment.segments) {
        let window = segment_window(images, seg);
        let output = pixelmatch_rgba(
            window.baseline,
            window.current,
            width,
            window.rows,
            pixel_options,
        )?;

        let own_rows = window.lead * stride..(window.lead + seg.len) * stride;
        let at = seg.current_start * stride;
        diff[at..at + seg.len * stride].copy_from_slice(&output.diff_rgba[own_rows]);

        // The window's own count includes the context rows, which are not ours.
        diff_count += mask_count(&segment_mask(images, seg, pixel_options)?);
    }

    let deleted_color = pixel_options
        .diff_color_alt
        .unwrap_or(pixel_options.diff_color);
    for band in &alignment.bands {
        let color = match band.kind {
            ShiftBandKind::Inserted => pixel_options.diff_color,
            ShiftBandKind::Deleted => deleted_color,
        };
        draw_band(&mut diff, width, height, band, color);
    }

    Ok(PixelmatchOutput {
        diff_rgba: diff,
        diff_count,
        width,
        height,
    })
}

/// PNG-encoded [`aligned_diff_image_rgba`].
pub fn aligned_diff_image_png(
    images: &RowImages,
    alignment: &RowAlignment,
    pixel_options: &PixelmatchOptions,
) -> Result<Vec<u8>, Error> {
    let output = aligned_diff_image_rgba(images, alignment, pixel_options)?;
    encode_png(&output.diff_rgba, output.width, output.height)
}

/// Copy the rows the two images have in common into one contiguous buffer.
fn stitch_matched_rows(
    rgba: &[u8],
    stride: usize,
    segments: &[RowSegment],
    start_of: impl Fn(&RowSegment) -> usize,
) -> Vec<u8> {
    let mut out = Vec::new();
    for seg in segments
        .iter()
        .filter(|s| s.kind == RowSegmentKind::Equal || s.kind == RowSegmentKind::Replace)
    {
        out.extend_from_slice(row_range(rgba, stride, start_of(seg), seg.len));
    }
    out
}

/// SSIM over the matched rows only, so a shift does not depress the score.
///
/// Inserted and deleted rows are dropped from both buffers before scoring, which
/// leaves the two stitched images the same height. Rows that were far apart in
/// the originals become neighbors at the seams, so the 11×11 window picks up
/// small artifacts there — acceptable against reporting the whole shift.
pub fn aligned_ssim(images: &RowImages, alignment: &RowAlignment) -> Result<f64, Error> {
    check_alignment(images, alignment)?;

    let stride = images.width * 4;
    if stride == 0 {
        return Ok(1.0);
    }

    let baseline = stitch_matched_rows(images.baseline_rgba, stride, &alignment.segments, |s| {
        s.baseline_start
    });
    let current = stitch_matched_rows(images.current_rgba, stride, &alignment.segments, |s| {
        s.current_start
    });

    // Nothing in common to score, the same way SSIM treats an empty image.
    if baseline.is_empty() {
        return Ok(1.0);
    }

    compute_ssim_rgba(&baseline, &current, images.width, baseline.len() / stride)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments_of(baseline: &[u64], current: &[u64], max_d: usize) -> (usize, Vec<RowSegment>) {
        let (edit_distance, mut runs) =
            myers_diff(baseline, current, max_d).expect("diff should fit the budget");
        slide_edits_together(&mut runs, baseline, current);
        (edit_distance, pair_replacements(&runs))
    }

    #[test]
    fn identical_sequences_have_no_edits() {
        let rows = [1u64, 2, 3, 4];
        let (edit_distance, segments) = segments_of(&rows, &rows, 8);

        assert_eq!(edit_distance, 0);
        assert_eq!(segments, vec![segment(RowSegmentKind::Equal, 0, 0, 4)]);
    }

    #[test]
    fn single_inserted_row_is_one_insert_segment() {
        let baseline = [1u64, 2, 3];
        let current = [1u64, 9, 2, 3];
        let (edit_distance, segments) = segments_of(&baseline, &current, 8);

        assert_eq!(edit_distance, 1);
        assert_eq!(
            segments,
            vec![
                segment(RowSegmentKind::Equal, 0, 0, 1),
                segment(RowSegmentKind::Insert, 1, 1, 1),
                segment(RowSegmentKind::Equal, 1, 2, 2),
            ]
        );
    }

    #[test]
    fn single_deleted_row_is_one_delete_segment() {
        let baseline = [1u64, 9, 2, 3];
        let current = [1u64, 2, 3];
        let (edit_distance, segments) = segments_of(&baseline, &current, 8);

        assert_eq!(edit_distance, 1);
        assert_eq!(
            segments,
            vec![
                segment(RowSegmentKind::Equal, 0, 0, 1),
                segment(RowSegmentKind::Delete, 1, 1, 1),
                segment(RowSegmentKind::Equal, 2, 1, 2),
            ]
        );
    }

    #[test]
    fn insert_far_from_delete_does_not_pair() {
        let baseline = [1u64, 7, 2, 3, 4, 5];
        let current = [1u64, 2, 3, 8, 4, 5];
        let (_, segments) = segments_of(&baseline, &current, 12);

        let kinds: Vec<_> = segments.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                RowSegmentKind::Equal,
                RowSegmentKind::Delete,
                RowSegmentKind::Equal,
                RowSegmentKind::Insert,
                RowSegmentKind::Equal,
            ]
        );
    }

    #[test]
    fn adjacent_delete_and_insert_pair_and_leave_the_surplus() {
        let baseline = [1u64, 7, 8, 9, 4];
        let current = [1u64, 5, 6, 4];
        let (_, segments) = segments_of(&baseline, &current, 12);

        assert_eq!(
            segments,
            vec![
                segment(RowSegmentKind::Equal, 0, 0, 1),
                segment(RowSegmentKind::Replace, 1, 1, 2),
                segment(RowSegmentKind::Delete, 3, 3, 1),
                segment(RowSegmentKind::Equal, 4, 3, 1),
            ]
        );
    }

    #[test]
    fn budget_exceeded_returns_no_diff() {
        let baseline: Vec<u64> = (0..40).collect();
        let current: Vec<u64> = (100..140).collect();

        assert!(myers_diff(&baseline, &current, 8).is_none());
    }

    #[test]
    fn fully_transparent_pixels_hash_the_same_whatever_their_color() {
        let row = [10u8, 20, 30, 255, 40, 50, 60, 0];
        let recolored = [10u8, 20, 30, 255, 200, 210, 220, 0];

        assert_eq!(hash_row(&row), hash_row(&recolored));
    }

    #[test]
    fn a_replaced_row_inside_a_uniform_block_is_one_replace() {
        // 100 identical rows with one of them replaced. Myers is free to delete
        // at one end of the block and insert at the other, which would read as a
        // shift; the edits have to come back together.
        let baseline = vec![7u64; 100];
        let mut current = baseline.clone();
        current[50] = 9;

        let (_, segments) = segments_of(&baseline, &current, 64);

        assert_eq!(
            segments,
            vec![
                segment(RowSegmentKind::Equal, 0, 0, 50),
                segment(RowSegmentKind::Replace, 50, 50, 1),
                segment(RowSegmentKind::Equal, 51, 51, 49),
            ]
        );
    }

    #[test]
    fn edits_across_a_varied_block_stay_separate() {
        // The rows between the two edits differ from each other, so neither edit
        // can move through them.
        let baseline = [1u64, 2, 3, 4];
        let current = [9u64, 1, 2, 3];

        let (_, segments) = segments_of(&baseline, &current, 12);

        assert_eq!(
            segments,
            vec![
                segment(RowSegmentKind::Insert, 0, 0, 1),
                segment(RowSegmentKind::Equal, 0, 1, 3),
                segment(RowSegmentKind::Delete, 3, 4, 1),
            ]
        );
    }

    #[test]
    fn repeated_rows_match_by_position() {
        // A page that is mostly blank background, with one row inserted into it.
        let mut baseline = vec![7u64; 25];
        baseline.push(42);
        let mut current = baseline.clone();
        current.insert(10, 43);

        let (edit_distance, segments) = segments_of(&baseline, &current, 12);

        assert_eq!(edit_distance, 1);
        assert_eq!(
            segments,
            vec![
                segment(RowSegmentKind::Equal, 0, 0, 10),
                segment(RowSegmentKind::Insert, 10, 10, 1),
                segment(RowSegmentKind::Equal, 10, 11, 16),
            ]
        );
    }
}
