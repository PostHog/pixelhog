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
}

impl Default for ClusterOptions {
    fn default() -> Self {
        Self {
            min_pixels: 16,
            min_side: 0,
            dilation: 4,
            max_clusters: None,
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
        })
        .filter(|c| min_side == 0 || c.bbox.width.min(c.bbox.height) >= min_side)
        .collect();

    clusters.sort_by(|a, b| b.pixel_count.cmp(&a.pixel_count));

    let total_clusters = clusters.len();
    if let Some(max) = options.max_clusters {
        clusters.truncate(max);
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
        }
    }

    fn raw_opts(min_pixels: usize, min_side: usize, dilation: usize) -> ClusterOptions {
        ClusterOptions {
            min_pixels,
            min_side,
            dilation,
            max_clusters: None,
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
            },
        );
        assert_eq!(capped.clusters.len(), 2);
        assert_eq!(capped.total_clusters, 3);
    }
}
