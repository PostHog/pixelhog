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

/// Compute connected-component clusters from a binary diff mask.
///
/// Uses two-pass CCL with 8-connectivity and union-find.
/// Clusters smaller than `min_cluster_size` are discarded.
pub fn compute_clusters(
    diff_mask: &[bool],
    width: usize,
    height: usize,
    min_cluster_size: usize,
) -> Vec<DiffCluster> {
    assert_eq!(diff_mask.len(), width * height);

    if width == 0 || height == 0 {
        return Vec::new();
    }

    let mut labels = vec![0u32; width * height];
    let mut uf = UnionFind::new();
    // Label 0 is reserved for "background" (no diff).
    uf.make_set(); // index 0, unused

    // Pass 1: assign provisional labels, record equivalences.
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !diff_mask[idx] {
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

    // Collect clusters that meet the minimum size.
    stats
        .into_iter()
        .filter(|s| s.pixel_count > 0 && s.pixel_count >= min_cluster_size)
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
        .collect()
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

    #[test]
    fn empty_mask_produces_no_clusters() {
        let mask = vec![false; 100];
        let clusters = compute_clusters(&mask, 10, 10, 1);
        assert!(clusters.is_empty());
    }

    #[test]
    fn single_pixel_cluster() {
        let mut mask = vec![false; 25];
        mask[12] = true; // (2, 2) in a 5x5 grid
        let clusters = compute_clusters(&mask, 5, 5, 1);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].pixel_count, 1);
        assert_eq!(clusters[0].bbox.x, 2);
        assert_eq!(clusters[0].bbox.y, 2);
        assert_eq!(clusters[0].bbox.width, 1);
        assert_eq!(clusters[0].bbox.height, 1);
    }

    #[test]
    fn two_separate_clusters() {
        // 10x10 grid with two 2x2 blocks far apart
        let mut mask = vec![false; 100];
        // Block at (0,0)
        mask[0] = true;
        mask[1] = true;
        mask[10] = true;
        mask[11] = true;
        // Block at (8,8)
        mask[88] = true;
        mask[89] = true;
        mask[98] = true;
        mask[99] = true;

        let clusters = compute_clusters(&mask, 10, 10, 1);
        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].pixel_count, 4);
        assert_eq!(clusters[1].pixel_count, 4);
    }

    #[test]
    fn diagonal_connectivity() {
        // 8-connectivity: diagonally adjacent pixels should be in the same cluster
        let mut mask = vec![false; 9];
        mask[0] = true; // (0,0)
        mask[4] = true; // (1,1)
        mask[8] = true; // (2,2)

        let clusters = compute_clusters(&mask, 3, 3, 1);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].pixel_count, 3);
    }

    #[test]
    fn min_cluster_size_filters_small() {
        let mut mask = vec![false; 100];
        // 1-pixel cluster
        mask[5] = true;
        // 5-pixel cluster
        mask[50] = true;
        mask[51] = true;
        mask[52] = true;
        mask[60] = true;
        mask[61] = true;

        let clusters = compute_clusters(&mask, 10, 10, 3);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].pixel_count, 5);
    }

    #[test]
    fn min_cluster_size_zero_no_panic() {
        let mut mask = vec![false; 25];
        mask[12] = true;
        let clusters = compute_clusters(&mask, 5, 5, 0);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].pixel_count, 1);
    }

    #[test]
    fn empty_mask_min_cluster_size_zero_no_panic() {
        let mask = vec![false; 100];
        let clusters = compute_clusters(&mask, 10, 10, 0);
        assert!(clusters.is_empty());
    }

    #[test]
    fn centroid_calculation() {
        // Horizontal line at y=5, x=2..=4
        let mut mask = vec![false; 100];
        mask[52] = true; // (2,5)
        mask[53] = true; // (3,5)
        mask[54] = true; // (4,5)

        let clusters = compute_clusters(&mask, 10, 10, 1);
        assert_eq!(clusters.len(), 1);
        let c = &clusters[0];
        assert!((c.centroid.0 - 3.0).abs() < f64::EPSILON);
        assert!((c.centroid.1 - 5.0).abs() < f64::EPSILON);
    }
}
