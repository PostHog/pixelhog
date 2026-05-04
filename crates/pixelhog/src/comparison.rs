use crate::clusters::{compute_clusters, DiffCluster};
use crate::image_utils::{decode_png_rgba, encode_png, pad_images_to_largest_cow, thumbnail_webp};
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
    // Original current image for thumbnail generation.
    // `None` when images were the same size (no padding happened),
    // so `current_rgba` can be used directly.
    current_original: Option<OriginalImage>,
}

struct OriginalImage {
    rgba: Vec<u8>,
    width: usize,
    height: usize,
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

        let current_original = if needs_padding {
            Some(OriginalImage {
                rgba: current.clone(),
                width: current_width,
                height: current_height,
            })
        } else {
            None
        };

        let (baseline_padded, current_padded, width, height) = pad_images_to_largest_cow(
            &baseline,
            baseline_width,
            baseline_height,
            &current,
            current_width,
            current_height,
        )?;

        Ok(Self {
            baseline_rgba: baseline_padded.into_owned(),
            current_rgba: current_padded.into_owned(),
            width,
            height,
            current_original,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
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
        min_cluster_size: usize,
    ) -> Result<Vec<DiffCluster>, Error> {
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
            min_cluster_size,
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

    /// Generate a lossless WebP thumbnail of the current image.
    ///
    /// Uses the original (pre-padding) image so the thumbnail has the
    /// correct aspect ratio even when images were different sizes.
    pub fn thumbnail(&self, max_width: usize, max_height: Option<usize>) -> Result<Vec<u8>, Error> {
        let (rgba, w, h) = match &self.current_original {
            Some(orig) => (orig.rgba.as_slice(), orig.width, orig.height),
            None => (self.current_rgba.as_slice(), self.width, self.height),
        };
        thumbnail_webp(rgba, w, h, max_width, max_height)
    }
}
