/// Axis-aligned bounding box of a cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// A connected region of differing pixels.
#[derive(Debug, Clone)]
pub struct DiffCluster {
    pub bbox: BoundingBox,
    pub pixel_count: usize,
    pub centroid: (f64, f64),
    /// Number of raw clusters that were merged to produce this one (1 = no merge).
    pub merged_from: usize,
}

/// Options for cluster extraction.
pub struct ClusterOptions {
    /// Discard clusters with fewer than this many pixels.
    pub min_pixels: usize,
    /// Discard clusters where `min(bbox.width, bbox.height) < min_side`.
    pub min_side: usize,
    /// Dilate the diff mask by this many pixels before CCL.
    /// Merges nearby diff regions that belong to the same visual change.
    pub dilation: usize,
    /// Keep only the top N clusters by pixel count. `None` = no limit.
    pub max_clusters: Option<usize>,
    /// Post-CCL aligned-bbox merge: max gap on the perpendicular axis.
    /// 0 disables merging. Merges clusters that are aligned on one axis
    /// and within this distance on the other.
    pub merge_gap: usize,
    /// Minimum overlap ratio on the shared axis to consider two clusters aligned.
    pub merge_overlap: f64,
}

impl Default for ClusterOptions {
    fn default() -> Self {
        Self {
            min_pixels: 16,
            min_side: 0,
            dilation: 4,
            max_clusters: None,
            merge_gap: 0,
            merge_overlap: 0.5,
        }
    }
}

/// Result of cluster computation.
#[derive(Debug, Clone)]
pub struct ClustersOutput {
    /// Clusters sorted by `pixel_count` descending, possibly truncated.
    pub clusters: Vec<DiffCluster>,
    /// Total qualifying clusters before `max_clusters` truncation.
    pub total_clusters: usize,
}

fn axis_overlap_ratio(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> f64 {
    let overlap_start = a_start.max(b_start);
    let overlap_end = a_end.min(b_end);
    if overlap_start >= overlap_end {
        return 0.0;
    }
    let overlap = (overlap_end - overlap_start) as f64;
    let shorter = (a_end - a_start).min(b_end - b_start) as f64;
    if shorter == 0.0 {
        return 0.0;
    }
    overlap / shorter
}

fn perpendicular_gap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> usize {
    if a_end <= b_start {
        b_start - a_end
    } else {
        a_start.saturating_sub(b_end)
    }
}

fn should_merge(a: &BoundingBox, b: &BoundingBox, max_gap: usize, min_overlap: f64) -> bool {
    let x_overlap = axis_overlap_ratio(a.x, a.x + a.width, b.x, b.x + b.width);
    let y_overlap = axis_overlap_ratio(a.y, a.y + a.height, b.y, b.y + b.height);

    if x_overlap > 0.0 && y_overlap > 0.0 {
        return true;
    }

    // Aligned on X (horizontally overlapping), check vertical gap.
    if x_overlap >= min_overlap {
        let gap = perpendicular_gap(a.y, a.y + a.height, b.y, b.y + b.height);
        if gap <= max_gap {
            return true;
        }
    }

    // Aligned on Y (vertically overlapping), check horizontal gap.
    if y_overlap >= min_overlap {
        let gap = perpendicular_gap(a.x, a.x + a.width, b.x, b.x + b.width);
        if gap <= max_gap {
            return true;
        }
    }

    false
}

fn merge_two(a: &DiffCluster, b: &DiffCluster) -> DiffCluster {
    let x_min = a.bbox.x.min(b.bbox.x);
    let y_min = a.bbox.y.min(b.bbox.y);
    let x_max = (a.bbox.x + a.bbox.width).max(b.bbox.x + b.bbox.width);
    let y_max = (a.bbox.y + a.bbox.height).max(b.bbox.y + b.bbox.height);
    let total_pixels = a.pixel_count + b.pixel_count;
    let cx = (a.centroid.0 * a.pixel_count as f64 + b.centroid.0 * b.pixel_count as f64)
        / total_pixels as f64;
    let cy = (a.centroid.1 * a.pixel_count as f64 + b.centroid.1 * b.pixel_count as f64)
        / total_pixels as f64;
    DiffCluster {
        bbox: BoundingBox {
            x: x_min,
            y: y_min,
            width: x_max - x_min,
            height: y_max - y_min,
        },
        pixel_count: total_pixels,
        centroid: (cx, cy),
        merged_from: a.merged_from + b.merged_from,
    }
}

fn merge_aligned_clusters(clusters: &mut Vec<DiffCluster>, max_gap: usize, min_overlap: f64) {
    loop {
        let mut merged = false;
        let mut i = 0;
        while i < clusters.len() {
            let mut j = i + 1;
            while j < clusters.len() {
                if should_merge(&clusters[i].bbox, &clusters[j].bbox, max_gap, min_overlap) {
                    let b = clusters.remove(j);
                    clusters[i] = merge_two(&clusters[i], &b);
                    merged = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        if !merged {
            break;
        }
    }
    clusters.sort_by(|a, b| b.pixel_count.cmp(&a.pixel_count));
}

/// Compute connected-component clusters from a binary diff mask.
///
/// Uses optional dilation to merge nearby regions, two-pass CCL with
/// 8-connectivity and union-find, then filters by size constraints.
/// Results are sorted by `pixel_count` descending and optionally truncated.
pub fn compute_clusters(
    diff_mask: &[bool],
    width: usize,
    height: usize,
    options: &ClusterOptions,
) -> ClustersOutput {
    assert_eq!(diff_mask.len(), width * height);

    if width == 0 || height == 0 {
        return ClustersOutput {
            clusters: Vec::new(),
            total_clusters: 0,
        };
    }

    let mask = if options.dilation > 0 {
        dilate_mask(diff_mask, width, height, options.dilation)
    } else {
        diff_mask.to_vec()
    };

    let mut labels = vec![0u32; width * height];
    let mut uf = UnionFind::new();
    // Label 0 is reserved for "background" (no diff).
    uf.make_set(); // index 0, unused

    // Pass 1: assign provisional labels, record equivalences.
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !mask[idx] {
                continue;
            }

            let mut min_label = 0u32;

            // Check 8-connected neighbors that have already been visited
            // (those above and to the left in raster order).
            let neighbors: [(isize, isize); 4] = [(-1, -1), (0, -1), (1, -1), (-1, 0)];

            for (dx, dy) in neighbors {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                    continue;
                }
                let nlabel = labels[ny as usize * width + nx as usize];
                if nlabel != 0 {
                    if min_label == 0 {
                        min_label = nlabel;
                    } else {
                        let root_min = uf.find(min_label);
                        let root_n = uf.find(nlabel);
                        if root_min != root_n {
                            uf.union(root_min, root_n);
                        }
                        min_label = min_label.min(nlabel);
                    }
                }
            }

            if min_label == 0 {
                min_label = uf.make_set();
            }

            labels[idx] = min_label;
        }
    }

    // Pass 2: resolve all labels to canonical roots and accumulate stats.
    let num_labels = uf.len();
    let mut stats = vec![ClusterStats::default(); num_labels];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let label = labels[idx];
            if label == 0 {
                continue;
            }
            let root = uf.find(label) as usize;
            let s = &mut stats[root];
            s.pixel_count += 1;
            s.sum_x += x as u64;
            s.sum_y += y as u64;
            s.min_x = s.min_x.min(x);
            s.max_x = s.max_x.max(x);
            s.min_y = s.min_y.min(y);
            s.max_y = s.max_y.max(y);
        }
    }

    let min_pixels = options.min_pixels;
    let min_side = options.min_side;

    let mut clusters: Vec<DiffCluster> = stats
        .into_iter()
        .filter(|s| s.pixel_count > 0 && s.pixel_count >= min_pixels)
        .map(|s| DiffCluster {
            bbox: BoundingBox {
                x: s.min_x,
                y: s.min_y,
                width: s.max_x - s.min_x + 1,
                height: s.max_y - s.min_y + 1,
            },
            pixel_count: s.pixel_count,
            centroid: (
                s.sum_x as f64 / s.pixel_count as f64,
                s.sum_y as f64 / s.pixel_count as f64,
            ),
            merged_from: 1,
        })
        .filter(|c| min_side == 0 || c.bbox.width.min(c.bbox.height) >= min_side)
        .collect();

    clusters.sort_by(|a, b| b.pixel_count.cmp(&a.pixel_count));

    let total_clusters = clusters.len();
    if let Some(max) = options.max_clusters {
        clusters.truncate(max);
    }

    if options.merge_gap > 0 {
        merge_aligned_clusters(&mut clusters, options.merge_gap, options.merge_overlap);
    }

    ClustersOutput {
        clusters,
        total_clusters,
    }
}

/// Dilate a boolean mask by `radius` pixels (square structuring element).
fn dilate_mask(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    let mut dilated = mask.to_vec();
    for y in 0..height {
        for x in 0..width {
            if !mask[y * width + x] {
                continue;
            }
            let y_min = y.saturating_sub(radius);
            let y_max = (y + radius).min(height - 1);
            let x_min = x.saturating_sub(radius);
            let x_max = (x + radius).min(width - 1);
            for dy in y_min..=y_max {
                let row_start = dy * width;
                for dx in x_min..=x_max {
                    dilated[row_start + dx] = true;
                }
            }
        }
    }
    dilated
}

// -- Union-Find --------------------------------------------------------------

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.parent.len()
    }

    fn make_set(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let rank_a = self.rank[ra as usize];
        let rank_b = self.rank[rb as usize];
        if rank_a < rank_b {
            self.parent[ra as usize] = rb;
        } else if rank_a > rank_b {
            self.parent[rb as usize] = ra;
        } else {
            self.parent[rb as usize] = ra;
            self.rank[ra as usize] += 1;
        }
    }
}

// -- Per-cluster accumulator -------------------------------------------------

#[derive(Clone)]
struct ClusterStats {
    pixel_count: usize,
    sum_x: u64,
    sum_y: u64,
    min_x: usize,
    max_x: usize,
    min_y: usize,
    max_y: usize,
}

impl Default for ClusterStats {
    fn default() -> Self {
        Self {
            pixel_count: 0,
            sum_x: 0,
            sum_y: 0,
            min_x: usize::MAX,
            max_x: 0,
            min_y: usize::MAX,
            max_y: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(min_pixels: usize) -> ClusterOptions {
        ClusterOptions {
            min_pixels,
            min_side: 0,
            dilation: 0,
            max_clusters: None,
            merge_gap: 0,
            merge_overlap: 0.5,
        }
    }

    fn raw_opts(min_pixels: usize, min_side: usize, dilation: usize) -> ClusterOptions {
        ClusterOptions {
            min_pixels,
            min_side,
            dilation,
            max_clusters: None,
            merge_gap: 0,
            merge_overlap: 0.5,
        }
    }

    #[test]
    fn empty_mask_produces_no_clusters() {
        let mask = vec![false; 100];
        let out = compute_clusters(&mask, 10, 10, &opts(1));
        assert!(out.clusters.is_empty());
        assert_eq!(out.total_clusters, 0);
    }

    #[test]
    fn single_pixel_cluster() {
        let mut mask = vec![false; 25];
        mask[12] = true;
        let out = compute_clusters(&mask, 5, 5, &opts(1));
        assert_eq!(out.clusters.len(), 1);
        assert_eq!(out.clusters[0].pixel_count, 1);
        assert_eq!(out.clusters[0].bbox.x, 2);
        assert_eq!(out.clusters[0].bbox.y, 2);
        assert_eq!(out.clusters[0].bbox.width, 1);
        assert_eq!(out.clusters[0].bbox.height, 1);
    }

    #[test]
    fn two_separate_clusters() {
        let mut mask = vec![false; 100];
        mask[0] = true;
        mask[1] = true;
        mask[10] = true;
        mask[11] = true;
        mask[88] = true;
        mask[89] = true;
        mask[98] = true;
        mask[99] = true;

        let out = compute_clusters(&mask, 10, 10, &opts(1));
        assert_eq!(out.clusters.len(), 2);
        assert_eq!(out.total_clusters, 2);
        assert_eq!(out.clusters[0].pixel_count, 4);
        assert_eq!(out.clusters[1].pixel_count, 4);
    }

    #[test]
    fn diagonal_connectivity() {
        let mut mask = vec![false; 9];
        mask[0] = true;
        mask[4] = true;
        mask[8] = true;

        let out = compute_clusters(&mask, 3, 3, &opts(1));
        assert_eq!(out.clusters.len(), 1);
        assert_eq!(out.clusters[0].pixel_count, 3);
    }

    #[test]
    fn min_pixels_filters_small() {
        let mut mask = vec![false; 100];
        mask[5] = true;
        mask[50] = true;
        mask[51] = true;
        mask[52] = true;
        mask[60] = true;
        mask[61] = true;

        let out = compute_clusters(&mask, 10, 10, &opts(3));
        assert_eq!(out.clusters.len(), 1);
        assert_eq!(out.clusters[0].pixel_count, 5);
    }

    #[test]
    fn min_pixels_zero_no_panic() {
        let mut mask = vec![false; 25];
        mask[12] = true;
        let out = compute_clusters(&mask, 5, 5, &opts(0));
        assert_eq!(out.clusters.len(), 1);
    }

    #[test]
    fn empty_mask_min_pixels_zero_no_panic() {
        let mask = vec![false; 100];
        let out = compute_clusters(&mask, 10, 10, &opts(0));
        assert!(out.clusters.is_empty());
    }

    #[test]
    fn centroid_calculation() {
        let mut mask = vec![false; 100];
        mask[52] = true;
        mask[53] = true;
        mask[54] = true;

        let out = compute_clusters(&mask, 10, 10, &opts(1));
        assert_eq!(out.clusters.len(), 1);
        let c = &out.clusters[0];
        assert!((c.centroid.0 - 3.0).abs() < f64::EPSILON);
        assert!((c.centroid.1 - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sorted_by_pixel_count_desc() {
        let mut mask = vec![false; 100];
        mask[5] = true;
        mask[50] = true;
        mask[51] = true;
        mask[52] = true;

        let out = compute_clusters(&mask, 10, 10, &opts(1));
        assert_eq!(out.clusters.len(), 2);
        assert!(out.clusters[0].pixel_count >= out.clusters[1].pixel_count);
    }

    #[test]
    fn min_side_filters_thin_clusters() {
        let mut mask = vec![false; 100];
        for x in 0..10 {
            mask[50 + x] = true;
        }
        mask[0] = true;
        mask[1] = true;
        mask[2] = true;
        mask[10] = true;
        mask[11] = true;
        mask[12] = true;
        mask[20] = true;
        mask[21] = true;
        mask[22] = true;

        let no_filter = compute_clusters(&mask, 10, 10, &raw_opts(1, 0, 0));
        assert_eq!(no_filter.clusters.len(), 2);

        let with_min_side = compute_clusters(&mask, 10, 10, &raw_opts(1, 2, 0));
        assert_eq!(with_min_side.clusters.len(), 1);
        assert_eq!(with_min_side.clusters[0].pixel_count, 9);
    }

    #[test]
    fn dilation_merges_nearby_clusters() {
        let mut mask = vec![false; 100];
        mask[50] = true;
        mask[53] = true;

        let no_dilation = compute_clusters(&mask, 10, 10, &raw_opts(1, 0, 0));
        assert_eq!(no_dilation.clusters.len(), 2);

        let with_dilation = compute_clusters(&mask, 10, 10, &raw_opts(1, 0, 2));
        assert_eq!(with_dilation.clusters.len(), 1);
    }

    #[test]
    fn max_clusters_truncates() {
        let mut mask = vec![false; 100];
        // 3 separate single-pixel clusters
        mask[0] = true;
        mask[5] = true;
        mask[99] = true;

        let all = compute_clusters(&mask, 10, 10, &opts(1));
        assert_eq!(all.clusters.len(), 3);
        assert_eq!(all.total_clusters, 3);

        let capped = compute_clusters(
            &mask,
            10,
            10,
            &ClusterOptions {
                min_pixels: 1,
                min_side: 0,
                dilation: 0,
                max_clusters: Some(2),
                merge_gap: 0,
                merge_overlap: 0.5,
            },
        );
        assert_eq!(capped.clusters.len(), 2);
        assert_eq!(capped.total_clusters, 3);
    }

    #[test]
    fn aligned_merge_vertical_list() {
        // Three horizontally-aligned clusters stacked vertically with small gaps.
        // Simulates rows in a list shifting: same x-extent, separated by 2 rows.
        let mut mask = vec![false; 20 * 20];
        // Row 0-1, cols 2-7 (cluster A)
        for y in 0..2 {
            for x in 2..8 {
                mask[y * 20 + x] = true;
            }
        }
        // Row 4-5, cols 2-7 (cluster B, gap of 2 rows)
        for y in 4..6 {
            for x in 2..8 {
                mask[y * 20 + x] = true;
            }
        }
        // Row 8-9, cols 2-7 (cluster C, gap of 2 rows)
        for y in 8..10 {
            for x in 2..8 {
                mask[y * 20 + x] = true;
            }
        }

        // Without merge: 3 clusters.
        let no_merge = compute_clusters(&mask, 20, 20, &opts(1));
        assert_eq!(no_merge.clusters.len(), 3);

        // With merge (gap=3, overlap=0.5): all collapse into 1.
        let merged = compute_clusters(
            &mask,
            20,
            20,
            &ClusterOptions {
                min_pixels: 1,
                min_side: 0,
                dilation: 0,
                max_clusters: None,
                merge_gap: 3,
                merge_overlap: 0.5,
            },
        );
        assert_eq!(merged.clusters.len(), 1);
        assert_eq!(merged.clusters[0].pixel_count, 36);
        assert_eq!(merged.clusters[0].merged_from, 3);
    }

    #[test]
    fn aligned_merge_does_not_merge_unrelated() {
        // Two clusters at opposite corners — no axis alignment.
        let mut mask = vec![false; 20 * 20];
        // Top-left: rows 0-2, cols 0-2
        for y in 0..3 {
            for x in 0..3 {
                mask[y * 20 + x] = true;
            }
        }
        // Bottom-right: rows 17-19, cols 17-19
        for y in 17..20 {
            for x in 17..20 {
                mask[y * 20 + x] = true;
            }
        }

        let merged = compute_clusters(
            &mask,
            20,
            20,
            &ClusterOptions {
                min_pixels: 1,
                min_side: 0,
                dilation: 0,
                max_clusters: None,
                merge_gap: 60,
                merge_overlap: 0.5,
            },
        );
        assert_eq!(merged.clusters.len(), 2);
        assert_eq!(merged.clusters[0].merged_from, 1);
    }

    #[test]
    fn aligned_merge_horizontal_strip() {
        // Two vertically-aligned clusters side by side with a small horizontal gap.
        let mut mask = vec![false; 20 * 20];
        // Cols 0-3, rows 5-14 (cluster A)
        for y in 5..15 {
            for x in 0..4 {
                mask[y * 20 + x] = true;
            }
        }
        // Cols 6-9, rows 5-14 (cluster B, gap of 2 cols)
        for y in 5..15 {
            for x in 6..10 {
                mask[y * 20 + x] = true;
            }
        }

        let merged = compute_clusters(
            &mask,
            20,
            20,
            &ClusterOptions {
                min_pixels: 1,
                min_side: 0,
                dilation: 0,
                max_clusters: None,
                merge_gap: 3,
                merge_overlap: 0.5,
            },
        );
        assert_eq!(merged.clusters.len(), 1);
        assert_eq!(merged.clusters[0].merged_from, 2);
    }

    #[test]
    fn merge_gap_zero_disables() {
        let mut mask = vec![false; 20 * 20];
        for y in 0..2 {
            for x in 2..8 {
                mask[y * 20 + x] = true;
            }
        }
        for y in 4..6 {
            for x in 2..8 {
                mask[y * 20 + x] = true;
            }
        }

        let result = compute_clusters(
            &mask,
            20,
            20,
            &ClusterOptions {
                min_pixels: 1,
                min_side: 0,
                dilation: 0,
                max_clusters: None,
                merge_gap: 0,
                merge_overlap: 0.5,
            },
        );
        assert_eq!(result.clusters.len(), 2);
    }
}
