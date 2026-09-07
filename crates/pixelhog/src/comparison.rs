use crate::alignment::{
    aligned_clusters, aligned_diff_image_png, aligned_diff_image_rgba, aligned_ssim,
    compute_row_alignment, RowAlignment, RowAlignmentOptions, RowImages,
};
use crate::clusters::{compute_clusters, ClusterOptions, ClustersOutput};
use crate::image_utils::{
    decode_png_rgba, encode_png, pad_images_to_largest_cow, thumbnail_webp_full,
};
use crate::pixelmatch::{
    pixelmatch_count_rgba, pixelmatch_count_rgba_capped, pixelmatch_mask_rgba, pixelmatch_rgba,
    PixelmatchOptions, PixelmatchOutput,
};
use crate::ssim::compute_ssim_rgba;
use crate::Error;
use rayon::join;

/// Holds a decoded, padded image pair ready for comparison.
///
/// Construct via [`Comparison::from_png`] or [`Comparison::from_rgba`],
/// then call individual methods to compute what you need. The PNG decode
/// and padding happen once at construction time.
pub struct Comparison {
    baseline_rgba: Vec<u8>,
    current_rgba: Vec<u8>,
    width: usize,
    height: usize,
    baseline_width: usize,
    baseline_height: usize,
    current_width: usize,
    current_height: usize,
    // Original current image for thumbnail generation when sizes differ.
    current_original: Option<Vec<u8>>,
}

impl Comparison {
    pub fn from_png(baseline_png: &[u8], current_png: &[u8]) -> Result<Self, Error> {
        let (baseline_decoded, current_decoded) = join(
            || decode_png_rgba(baseline_png),
            || decode_png_rgba(current_png),
        );
        let (baseline_rgba, bw, bh) = baseline_decoded?;
        let (current_rgba, cw, ch) = current_decoded?;

        Self::from_rgba_owned(baseline_rgba, bw, bh, current_rgba, cw, ch)
    }

    pub fn from_rgba(
        baseline: &[u8],
        baseline_width: usize,
        baseline_height: usize,
        current: &[u8],
        current_width: usize,
        current_height: usize,
    ) -> Result<Self, Error> {
        Self::from_rgba_owned(
            baseline.to_vec(),
            baseline_width,
            baseline_height,
            current.to_vec(),
            current_width,
            current_height,
        )
    }

    fn from_rgba_owned(
        baseline: Vec<u8>,
        baseline_width: usize,
        baseline_height: usize,
        current: Vec<u8>,
        current_width: usize,
        current_height: usize,
    ) -> Result<Self, Error> {
        let needs_padding = baseline_width != current_width || baseline_height != current_height;

        let (baseline_padded, current_padded, width, height) = pad_images_to_largest_cow(
            &baseline,
            baseline_width,
            baseline_height,
            &current,
            current_width,
            current_height,
        )?;

        // Materialize padded buffers first to release borrows.
        let baseline_rgba = baseline_padded.into_owned();
        let current_rgba = current_padded.into_owned();

        // Move the original current for thumbnailing at native dimensions.
        let current_original = if needs_padding { Some(current) } else { None };

        Ok(Self {
            baseline_rgba,
            current_rgba,
            width,
            height,
            baseline_width,
            baseline_height,
            current_width,
            current_height,
            current_original,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn size_mismatch(&self) -> bool {
        self.baseline_width != self.current_width || self.baseline_height != self.current_height
    }

    pub fn baseline_size(&self) -> (usize, usize) {
        (self.baseline_width, self.baseline_height)
    }

    pub fn current_size(&self) -> (usize, usize) {
        (self.current_width, self.current_height)
    }

    pub fn diff_count(&self, options: &PixelmatchOptions) -> Result<usize, Error> {
        pixelmatch_count_rgba(
            &self.baseline_rgba,
            &self.current_rgba,
            self.width,
            self.height,
            options,
        )
        .map(|o| o.diff_count)
    }

    pub fn diff_count_capped(
        &self,
        options: &PixelmatchOptions,
        max_diffs: usize,
    ) -> Result<usize, Error> {
        pixelmatch_count_rgba_capped(
            &self.baseline_rgba,
            &self.current_rgba,
            self.width,
            self.height,
            options,
            max_diffs,
        )
        .map(|o| o.diff_count)
    }

    pub fn ssim(&self) -> Result<f64, Error> {
        compute_ssim_rgba(
            &self.baseline_rgba,
            &self.current_rgba,
            self.width,
            self.height,
        )
    }

    pub fn clusters(
        &self,
        options: &PixelmatchOptions,
        cluster_options: &ClusterOptions,
    ) -> Result<ClustersOutput, Error> {
        let mask_output = pixelmatch_mask_rgba(
            &self.baseline_rgba,
            &self.current_rgba,
            self.width,
            self.height,
            options,
        )?;
        Ok(compute_clusters(
            &mask_output.diff_mask,
            self.width,
            self.height,
            cluster_options,
        ))
    }

    pub fn diff_image_rgba(&self, options: &PixelmatchOptions) -> Result<PixelmatchOutput, Error> {
        pixelmatch_rgba(
            &self.baseline_rgba,
            &self.current_rgba,
            self.width,
            self.height,
            options,
        )
    }

    pub fn diff_image_png(&self, options: &PixelmatchOptions) -> Result<Vec<u8>, Error> {
        let output = self.diff_image_rgba(options)?;
        encode_png(&output.diff_rgba, self.width, self.height)
    }

    /// Borrow both images as rows over the shared padded stride.
    fn row_images(&self) -> RowImages<'_> {
        RowImages {
            baseline_rgba: &self.baseline_rgba,
            current_rgba: &self.current_rgba,
            width: self.width,
            baseline_height: self.baseline_height,
            current_width: self.current_width,
            current_height: self.current_height,
        }
    }

    /// Align the rows of the two images to tell a vertical shift from real changes.
    pub fn row_alignment(
        &self,
        pixel_options: &PixelmatchOptions,
        options: &RowAlignmentOptions,
    ) -> Result<RowAlignment, Error> {
        compute_row_alignment(&self.row_images(), pixel_options, options)
    }

    /// Cluster the residual differences inside `Replace` segments.
    ///
    /// The mask is in current-image coordinates and leaves out the shift bands —
    /// a one-row band would not survive the `min_side` filter, so callers read
    /// [`RowAlignment::bands`] directly.
    pub fn aligned_clusters(
        &self,
        alignment: &RowAlignment,
        pixel_options: &PixelmatchOptions,
        cluster_options: &ClusterOptions,
    ) -> Result<ClustersOutput, Error> {
        aligned_clusters(
            &self.row_images(),
            alignment,
            pixel_options,
            cluster_options,
        )
    }

    /// Build the shift-aware diff image in current-image coordinates.
    pub fn aligned_diff_image_rgba(
        &self,
        alignment: &RowAlignment,
        pixel_options: &PixelmatchOptions,
    ) -> Result<PixelmatchOutput, Error> {
        aligned_diff_image_rgba(&self.row_images(), alignment, pixel_options)
    }

    /// PNG-encoded [`Comparison::aligned_diff_image_rgba`].
    pub fn aligned_diff_image_png(
        &self,
        alignment: &RowAlignment,
        pixel_options: &PixelmatchOptions,
    ) -> Result<Vec<u8>, Error> {
        aligned_diff_image_png(&self.row_images(), alignment, pixel_options)
    }

    /// SSIM over the matched rows only, so a vertical shift does not lower the score.
    ///
    /// Inserted and deleted rows are dropped from both images before scoring.
    /// Rows that were far apart become neighbors at the seams, so the SSIM
    /// window picks up small artifacts there.
    pub fn aligned_ssim(&self, alignment: &RowAlignment) -> Result<f64, Error> {
        aligned_ssim(&self.row_images(), alignment)
    }

    /// Generate a lossless WebP thumbnail of the current image.
    ///
    /// Uses the original (pre-padding) image so the thumbnail has the
    /// correct aspect ratio even when images were different sizes.
    /// When proportional scaling would make either dimension smaller than
    /// `min_width`/`min_height`, the original is top-left cropped instead.
    pub fn current_thumbnail(
        &self,
        max_width: usize,
        max_height: Option<usize>,
        min_width: Option<usize>,
        min_height: Option<usize>,
    ) -> Result<Vec<u8>, Error> {
        let (rgba, w, h) = match &self.current_original {
            Some(orig) => (orig.as_slice(), self.current_width, self.current_height),
            None => (self.current_rgba.as_slice(), self.width, self.height),
        };
        thumbnail_webp_full(rgba, w, h, max_width, max_height, min_width, min_height)
    }

    /// Generate a lossless WebP thumbnail of the baseline image.
    pub fn baseline_thumbnail(
        &self,
        max_width: usize,
        max_height: Option<usize>,
        min_width: Option<usize>,
        min_height: Option<usize>,
    ) -> Result<Vec<u8>, Error> {
        thumbnail_webp_full(
            &self.baseline_rgba,
            self.width,
            self.height,
            max_width,
            max_height,
            min_width,
            min_height,
        )
    }
}
