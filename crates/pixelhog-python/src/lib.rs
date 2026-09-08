#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use ::pixelhog::{
    compare_png, create_thumbnail, diff_count_png, diff_png, ssim_png, ClusterOptions,
    ClustersOutput, Comparison as RustComparison, PixelmatchOptions,
    RowAlignment as RustRowAlignment, RowAlignmentOptions, RowSegmentKind, ShiftBandKind,
    ThumbnailOptions,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

fn to_py_err(e: ::pixelhog::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn pixelmatch_options(
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
) -> PyResult<PixelmatchOptions> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(PyValueError::new_err(
            "threshold must be in the range [0.0, 1.0]",
        ));
    }

    if !(0.0..=1.0).contains(&alpha) {
        return Err(PyValueError::new_err(
            "alpha must be in the range [0.0, 1.0]",
        ));
    }

    Ok(PixelmatchOptions {
        threshold,
        alpha,
        include_aa,
        diff_color: [diff_color.0, diff_color.1, diff_color.2],
        aa_color: [aa_color.0, aa_color.1, aa_color.2],
        diff_color_alt: diff_color_alt.map(|c| [c.0, c.1, c.2]),
    })
}

fn pixelmatch_count_options(threshold: f64, include_aa: bool) -> PyResult<PixelmatchOptions> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(PyValueError::new_err(
            "threshold must be in the range [0.0, 1.0]",
        ));
    }

    Ok(PixelmatchOptions {
        threshold,
        include_aa,
        ..PixelmatchOptions::default()
    })
}

fn row_alignment_options(
    max_edit_ratio: f64,
    max_edit_rows: usize,
) -> PyResult<RowAlignmentOptions> {
    if !(0.0..=1.0).contains(&max_edit_ratio) {
        return Err(PyValueError::new_err(
            "max_edit_ratio must be in the range [0.0, 1.0]",
        ));
    }

    Ok(RowAlignmentOptions {
        max_edit_ratio,
        max_edit_rows,
    })
}

fn thumbnail_options(
    thumbnail_width: Option<usize>,
    thumbnail_height: Option<usize>,
) -> Option<ThumbnailOptions> {
    thumbnail_width.map(|max_width| ThumbnailOptions {
        max_width,
        max_height: thumbnail_height,
        min_width: None,
        min_height: None,
    })
}

/// Create a lossless WebP thumbnail from a PNG image.
///
/// Scales down to `width`, preserving aspect ratio. If `height` is set,
/// crops from the top after resizing — useful for fixed-size grid cells.
/// Images already within bounds are re-encoded without resizing.
#[pyfunction]
#[pyo3(name = "thumbnail", signature = (png_bytes, width = 200, height = None))]
fn thumbnail_py(
    py: Python<'_>,
    png_bytes: &[u8],
    width: usize,
    height: Option<usize>,
) -> PyResult<Py<PyBytes>> {
    let result = py
        .allow_threads(|| create_thumbnail(png_bytes, width, height))
        .map_err(to_py_err)?;
    Ok(PyBytes::new(py, &result).into())
}

/// Diff multiple image pairs in parallel.
#[pyfunction]
#[pyo3(name = "diff_batch", signature = (
    pairs,
    threshold = 0.1,
    alpha = 0.1,
    include_aa = false,
    diff_color = (255, 0, 0),
    aa_color = (255, 255, 0),
    diff_color_alt = None,
))]
fn diff_batch_py(
    py: Python<'_>,
    pairs: Vec<(Vec<u8>, Vec<u8>)>,
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
) -> PyResult<Vec<(Py<PyBytes>, usize, usize, usize)>> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;

    let results = py
        .allow_threads(|| {
            let r: Result<Vec<_>, ::pixelhog::Error> = pairs
                .into_par_iter()
                .map(|(baseline_png, current_png)| diff_png(&baseline_png, &current_png, &options))
                .collect();
            r
        })
        .map_err(to_py_err)?;

    Ok(results
        .into_iter()
        .map(|(diff_png, diff_count, width, height)| {
            (
                PyBytes::new(py, &diff_png).into(),
                diff_count,
                width,
                height,
            )
        })
        .collect())
}

/// Count mismatched pixels for multiple image pairs in parallel.
#[pyfunction]
#[pyo3(name = "diff_count_batch", signature = (
    pairs,
    threshold = 0.1,
    include_aa = false,
))]
fn diff_count_batch_py(
    py: Python<'_>,
    pairs: Vec<(Vec<u8>, Vec<u8>)>,
    threshold: f64,
    include_aa: bool,
) -> PyResult<Vec<(usize, usize, usize)>> {
    let options = pixelmatch_count_options(threshold, include_aa)?;

    py.allow_threads(|| {
        let r: Result<Vec<_>, ::pixelhog::Error> = pairs
            .into_par_iter()
            .map(|(baseline_png, current_png)| {
                diff_count_png(&baseline_png, &current_png, &options)
            })
            .collect();
        r
    })
    .map_err(to_py_err)
}

/// Compute SSIM for multiple image pairs in parallel.
#[pyfunction]
#[pyo3(name = "ssim_batch")]
fn ssim_batch_py(py: Python<'_>, pairs: Vec<(Vec<u8>, Vec<u8>)>) -> PyResult<Vec<f64>> {
    py.allow_threads(|| {
        let r: Result<Vec<_>, ::pixelhog::Error> = pairs
            .into_par_iter()
            .map(|(baseline_png, current_png)| ssim_png(&baseline_png, &current_png))
            .collect();
        r
    })
    .map_err(to_py_err)
}

/// Compare multiple image pairs in parallel (diff count + SSIM per pair).
///
/// Batch version of `compare()`. Set `thumbnail_width` to generate a
/// thumbnail per pair.
#[pyfunction]
#[pyo3(name = "compare_batch", signature = (
    pairs,
    threshold = 0.1,
    alpha = 0.1,
    include_aa = false,
    diff_color = (255, 0, 0),
    aa_color = (255, 255, 0),
    diff_color_alt = None,
    return_diff = false,
    thumbnail_width = None,
    thumbnail_height = None,
))]
fn compare_batch_py(
    py: Python<'_>,
    pairs: Vec<(Vec<u8>, Vec<u8>)>,
    threshold: f64,
    alpha: f64,
    include_aa: bool,
    diff_color: (u8, u8, u8),
    aa_color: (u8, u8, u8),
    diff_color_alt: Option<(u8, u8, u8)>,
    return_diff: bool,
    thumbnail_width: Option<usize>,
    thumbnail_height: Option<usize>,
) -> PyResult<
    Vec<(
        usize,
        f64,
        usize,
        usize,
        Option<Py<PyBytes>>,
        Option<Py<PyBytes>>,
    )>,
> {
    let options = pixelmatch_options(
        threshold,
        alpha,
        include_aa,
        diff_color,
        aa_color,
        diff_color_alt,
    )?;
    let thumb = thumbnail_options(thumbnail_width, thumbnail_height);

    let results = py
        .allow_threads(|| {
            let r: Result<Vec<_>, ::pixelhog::Error> = pairs
                .into_par_iter()
                .map(|(baseline_png, current_png)| {
                    compare_png(
                        &baseline_png,
                        &current_png,
                        &options,
                        return_diff,
                        thumb.as_ref(),
                    )
                })
                .collect();
            r
        })
        .map_err(to_py_err)?;

    Ok(results
        .into_iter()
        .map(|(diff_png, diff_count, ssim, width, height, thumb_webp)| {
            let diff_bytes = diff_png.map(|bytes| PyBytes::new(py, &bytes).into());
            let thumb_bytes = thumb_webp.map(|bytes| PyBytes::new(py, &bytes).into());
            (diff_count, ssim, width, height, diff_bytes, thumb_bytes)
        })
        .collect())
}

// -- Stateful Comparison API -------------------------------------------------

#[pyclass(frozen, name = "BoundingBox")]
struct BoundingBoxPy {
    #[pyo3(get)]
    x: usize,
    #[pyo3(get)]
    y: usize,
    #[pyo3(get)]
    width: usize,
    #[pyo3(get)]
    height: usize,
}

#[pymethods]
impl BoundingBoxPy {
    fn __repr__(&self) -> String {
        format!(
            "BoundingBox(x={}, y={}, width={}, height={})",
            self.x, self.y, self.width, self.height
        )
    }
}

#[pyclass(frozen, name = "Cluster")]
struct ClusterPy {
    #[pyo3(get)]
    bbox: Py<BoundingBoxPy>,
    #[pyo3(get)]
    pixel_count: usize,
    #[pyo3(get)]
    centroid: (f64, f64),
    #[pyo3(get)]
    merged_from: usize,
}

#[pymethods]
impl ClusterPy {
    fn __repr__(&self) -> String {
        format!(
            "Cluster(pixel_count={}, centroid=({:.1}, {:.1}), merged_from={})",
            self.pixel_count, self.centroid.0, self.centroid.1, self.merged_from
        )
    }
}

#[pyclass(frozen, name = "ClustersResult")]
struct ClustersResultPy {
    #[pyo3(get)]
    clusters: Vec<Py<ClusterPy>>,
    #[pyo3(get)]
    total_clusters: usize,
    #[pyo3(get)]
    truncated: bool,
}

#[pymethods]
impl ClustersResultPy {
    fn __repr__(&self) -> String {
        if self.truncated {
            format!(
                "ClustersResult({} of {}, truncated)",
                self.clusters.len(),
                self.total_clusters
            )
        } else {
            format!("ClustersResult({})", self.clusters.len())
        }
    }

    fn __len__(&self) -> usize {
        self.clusters.len()
    }
}

fn clusters_result_py(py: Python<'_>, output: ClustersOutput) -> PyResult<Py<ClustersResultPy>> {
    let clusters = output
        .clusters
        .into_iter()
        .map(|c| {
            let bbox = Py::new(
                py,
                BoundingBoxPy {
                    x: c.bbox.x,
                    y: c.bbox.y,
                    width: c.bbox.width,
                    height: c.bbox.height,
                },
            )?;
            Py::new(
                py,
                ClusterPy {
                    bbox,
                    pixel_count: c.pixel_count,
                    centroid: c.centroid,
                    merged_from: c.merged_from,
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;

    Py::new(
        py,
        ClustersResultPy {
            clusters,
            total_clusters: output.total_clusters,
            truncated: output.truncated,
        },
    )
}

#[pyclass(frozen, name = "ShiftBand")]
struct ShiftBandPy {
    #[pyo3(get)]
    y: usize,
    #[pyo3(get)]
    rows: usize,
    #[pyo3(get)]
    kind: &'static str,
}

#[pymethods]
impl ShiftBandPy {
    fn __repr__(&self) -> String {
        format!(
            "ShiftBand(kind={}, y={}, rows={})",
            self.kind, self.y, self.rows
        )
    }
}

#[pyclass(frozen, name = "RowSegment")]
struct RowSegmentPy {
    #[pyo3(get)]
    kind: &'static str,
    #[pyo3(get)]
    baseline_start: usize,
    #[pyo3(get)]
    current_start: usize,
    #[pyo3(get)]
    len: usize,
}

#[pymethods]
impl RowSegmentPy {
    fn __repr__(&self) -> String {
        format!(
            "RowSegment(kind={}, baseline_start={}, current_start={}, len={})",
            self.kind, self.baseline_start, self.current_start, self.len
        )
    }
}

#[pyclass(frozen, name = "RowAlignment")]
struct RowAlignmentPy {
    #[pyo3(get)]
    aligned: bool,
    #[pyo3(get)]
    edit_distance: usize,
    #[pyo3(get)]
    inserted_rows: usize,
    #[pyo3(get)]
    deleted_rows: usize,
    #[pyo3(get)]
    changed_rows: usize,
    #[pyo3(get)]
    residual_count: usize,
    #[pyo3(get)]
    segments: Vec<Py<RowSegmentPy>>,
    #[pyo3(get)]
    bands: Vec<Py<ShiftBandPy>>,
    // Kept so the aligned_* methods can reuse the Rust result.
    inner: RustRowAlignment,
}

#[pymethods]
impl RowAlignmentPy {
    fn __repr__(&self) -> String {
        if !self.aligned {
            return "RowAlignment(aligned=False)".to_string();
        }
        format!(
            "RowAlignment(edit_distance={}, inserted_rows={}, deleted_rows={}, changed_rows={}, residual_count={})",
            self.edit_distance,
            self.inserted_rows,
            self.deleted_rows,
            self.changed_rows,
            self.residual_count
        )
    }
}

fn segment_kind_name(kind: RowSegmentKind) -> &'static str {
    match kind {
        RowSegmentKind::Equal => "equal",
        RowSegmentKind::Replace => "replace",
        RowSegmentKind::Insert => "insert",
        RowSegmentKind::Delete => "delete",
    }
}

fn band_kind_name(kind: ShiftBandKind) -> &'static str {
    match kind {
        ShiftBandKind::Inserted => "inserted",
        ShiftBandKind::Deleted => "deleted",
    }
}

fn row_alignment_py(py: Python<'_>, alignment: RustRowAlignment) -> PyResult<RowAlignmentPy> {
    let segments = alignment
        .segments
        .iter()
        .map(|s| {
            Py::new(
                py,
                RowSegmentPy {
                    kind: segment_kind_name(s.kind),
                    baseline_start: s.baseline_start,
                    current_start: s.current_start,
                    len: s.len,
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;

    let bands = alignment
        .bands
        .iter()
        .map(|b| {
            Py::new(
                py,
                ShiftBandPy {
                    y: b.y,
                    rows: b.rows,
                    kind: band_kind_name(b.kind),
                },
            )
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(RowAlignmentPy {
        aligned: alignment.aligned,
        edit_distance: alignment.edit_distance,
        inserted_rows: alignment.inserted_rows,
        deleted_rows: alignment.deleted_rows,
        changed_rows: alignment.changed_rows,
        residual_count: alignment.residual_count,
        segments,
        bands,
        inner: alignment,
    })
}

#[pyclass(frozen, name = "Comparison")]
struct ComparisonPy {
    inner: RustComparison,
}

#[pymethods]
impl ComparisonPy {
    /// Create a Comparison from two PNG images.
    #[new]
    #[pyo3(signature = (baseline_png, current_png))]
    fn new(py: Python<'_>, baseline_png: &[u8], current_png: &[u8]) -> PyResult<Self> {
        let inner = py
            .allow_threads(|| RustComparison::from_png(baseline_png, current_png))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Create a Comparison from pre-decoded RGBA buffers.
    #[staticmethod]
    #[pyo3(name = "from_rgba", signature = (
        baseline_rgba, baseline_width, baseline_height,
        current_rgba, current_width, current_height,
    ))]
    fn from_rgba_py(
        py: Python<'_>,
        baseline_rgba: &[u8],
        baseline_width: usize,
        baseline_height: usize,
        current_rgba: &[u8],
        current_width: usize,
        current_height: usize,
    ) -> PyResult<Self> {
        let inner = py
            .allow_threads(|| {
                RustComparison::from_rgba(
                    baseline_rgba,
                    baseline_width,
                    baseline_height,
                    current_rgba,
                    current_width,
                    current_height,
                )
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Decode multiple PNG pairs in parallel.
    #[staticmethod]
    #[pyo3(signature = (pairs,))]
    fn batch(py: Python<'_>, pairs: Vec<(Vec<u8>, Vec<u8>)>) -> PyResult<Vec<Self>> {
        let results = py
            .allow_threads(|| {
                pairs
                    .into_par_iter()
                    .map(|(b, c)| RustComparison::from_png(&b, &c))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(to_py_err)?;
        Ok(results.into_iter().map(|inner| Self { inner }).collect())
    }

    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    #[getter]
    fn size_mismatch(&self) -> bool {
        self.inner.size_mismatch()
    }

    #[getter]
    fn baseline_size(&self) -> (usize, usize) {
        self.inner.baseline_size()
    }

    #[getter]
    fn current_size(&self) -> (usize, usize) {
        self.inner.current_size()
    }

    /// Count differing pixels.
    #[pyo3(signature = (threshold = 0.1, include_aa = false))]
    fn diff_count(&self, py: Python<'_>, threshold: f64, include_aa: bool) -> PyResult<usize> {
        let options = pixelmatch_count_options(threshold, include_aa)?;
        py.allow_threads(|| self.inner.diff_count(&options))
            .map_err(to_py_err)
    }

    /// Count differing pixels with early exit.
    #[pyo3(signature = (max_diffs, threshold = 0.1, include_aa = false))]
    fn diff_count_capped(
        &self,
        py: Python<'_>,
        max_diffs: usize,
        threshold: f64,
        include_aa: bool,
    ) -> PyResult<usize> {
        let options = pixelmatch_count_options(threshold, include_aa)?;
        py.allow_threads(|| self.inner.diff_count_capped(&options, max_diffs))
            .map_err(to_py_err)
    }

    /// Compute SSIM (structural similarity) score.
    fn ssim(&self, py: Python<'_>) -> PyResult<f64> {
        py.allow_threads(|| self.inner.ssim()).map_err(to_py_err)
    }

    /// Compute connected-component clusters of differing pixels.
    #[pyo3(signature = (threshold = 0.1, include_aa = false, min_pixels = 16, min_side = 0, dilation = 4, max_clusters = None, merge_gap = 0, merge_overlap = 0.5))]
    fn clusters(
        &self,
        py: Python<'_>,
        threshold: f64,
        include_aa: bool,
        min_pixels: usize,
        min_side: usize,
        dilation: usize,
        max_clusters: Option<usize>,
        merge_gap: usize,
        merge_overlap: f64,
    ) -> PyResult<Py<ClustersResultPy>> {
        let options = pixelmatch_count_options(threshold, include_aa)?;
        let cluster_opts = ClusterOptions {
            min_pixels,
            min_side,
            dilation,
            max_clusters,
            merge_gap,
            merge_overlap,
        };
        let output = py
            .allow_threads(|| self.inner.clusters(&options, &cluster_opts))
            .map_err(to_py_err)?;

        clusters_result_py(py, output)
    }

    /// Generate the diff image as PNG bytes.
    #[pyo3(signature = (
        threshold = 0.1,
        alpha = 0.1,
        include_aa = false,
        diff_color = (255, 0, 0),
        aa_color = (255, 255, 0),
        diff_color_alt = None,
    ))]
    fn diff_image(
        &self,
        py: Python<'_>,
        threshold: f64,
        alpha: f64,
        include_aa: bool,
        diff_color: (u8, u8, u8),
        aa_color: (u8, u8, u8),
        diff_color_alt: Option<(u8, u8, u8)>,
    ) -> PyResult<Py<PyBytes>> {
        let options = pixelmatch_options(
            threshold,
            alpha,
            include_aa,
            diff_color,
            aa_color,
            diff_color_alt,
        )?;
        let png = py
            .allow_threads(|| self.inner.diff_image_png(&options))
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &png).into())
    }

    /// Align the rows of the two images to tell a vertical shift from real changes.
    ///
    /// `aligned` is False when the pair could not be aligned — the edit budget
    /// was exceeded, or the widths differ, since alignment is vertical only.
    /// Every other field is then zero or empty.
    #[pyo3(signature = (threshold = 0.1, include_aa = false, max_edit_ratio = 0.25, max_edit_rows = 2048))]
    fn row_alignment(
        &self,
        py: Python<'_>,
        threshold: f64,
        include_aa: bool,
        max_edit_ratio: f64,
        max_edit_rows: usize,
    ) -> PyResult<Py<RowAlignmentPy>> {
        let options = pixelmatch_count_options(threshold, include_aa)?;
        let alignment_opts = row_alignment_options(max_edit_ratio, max_edit_rows)?;
        let alignment = py
            .allow_threads(|| self.inner.row_alignment(&options, &alignment_opts))
            .map_err(to_py_err)?;

        Py::new(py, row_alignment_py(py, alignment)?)
    }

    /// Cluster the residual differences, ignoring the shifted rows.
    ///
    /// The mask holds the residual only. The shift bands are the other half of
    /// the answer: read `alignment.bands` to decide whether to absorb a shift or
    /// flag it. Raises if the alignment failed or came from another image pair.
    #[pyo3(signature = (alignment, threshold = 0.1, include_aa = false, min_pixels = 16, min_side = 0, dilation = 4, max_clusters = None, merge_gap = 0, merge_overlap = 0.5))]
    fn aligned_clusters(
        &self,
        py: Python<'_>,
        alignment: PyRef<'_, RowAlignmentPy>,
        threshold: f64,
        include_aa: bool,
        min_pixels: usize,
        min_side: usize,
        dilation: usize,
        max_clusters: Option<usize>,
        merge_gap: usize,
        merge_overlap: f64,
    ) -> PyResult<Py<ClustersResultPy>> {
        let options = pixelmatch_count_options(threshold, include_aa)?;
        let cluster_opts = ClusterOptions {
            min_pixels,
            min_side,
            dilation,
            max_clusters,
            merge_gap,
            merge_overlap,
        };
        let inner = &alignment.inner;
        let output = py
            .allow_threads(|| self.inner.aligned_clusters(inner, &options, &cluster_opts))
            .map_err(to_py_err)?;

        clusters_result_py(py, output)
    }

    /// Generate the shift-aware diff image as PNG bytes, in current-image coordinates.
    #[pyo3(signature = (
        alignment,
        threshold = 0.1,
        alpha = 0.1,
        include_aa = false,
        diff_color = (255, 0, 0),
        aa_color = (255, 255, 0),
        diff_color_alt = None,
    ))]
    fn aligned_diff_image(
        &self,
        py: Python<'_>,
        alignment: PyRef<'_, RowAlignmentPy>,
        threshold: f64,
        alpha: f64,
        include_aa: bool,
        diff_color: (u8, u8, u8),
        aa_color: (u8, u8, u8),
        diff_color_alt: Option<(u8, u8, u8)>,
    ) -> PyResult<Py<PyBytes>> {
        let options = pixelmatch_options(
            threshold,
            alpha,
            include_aa,
            diff_color,
            aa_color,
            diff_color_alt,
        )?;
        let inner = &alignment.inner;
        let png = py
            .allow_threads(|| self.inner.aligned_diff_image_png(inner, &options))
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &png).into())
    }

    /// Compute SSIM over the matched rows only, so a shift does not lower the score.
    #[pyo3(signature = (alignment,))]
    fn aligned_ssim(&self, py: Python<'_>, alignment: PyRef<'_, RowAlignmentPy>) -> PyResult<f64> {
        let inner = &alignment.inner;
        py.allow_threads(|| self.inner.aligned_ssim(inner))
            .map_err(to_py_err)
    }

    /// Generate a lossless WebP thumbnail of the current image.
    ///
    /// When scaling would make either dimension smaller than `min_width` /
    /// `min_height`, the original is top-left cropped instead (no upscaling).
    #[pyo3(signature = (width = 200, height = None, min_width = None, min_height = None))]
    fn current_thumbnail(
        &self,
        py: Python<'_>,
        width: usize,
        height: Option<usize>,
        min_width: Option<usize>,
        min_height: Option<usize>,
    ) -> PyResult<Py<PyBytes>> {
        let result = py
            .allow_threads(|| {
                self.inner
                    .current_thumbnail(width, height, min_width, min_height)
            })
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &result).into())
    }

    /// Generate a lossless WebP thumbnail of the baseline image.
    #[pyo3(signature = (width = 200, height = None, min_width = None, min_height = None))]
    fn baseline_thumbnail(
        &self,
        py: Python<'_>,
        width: usize,
        height: Option<usize>,
        min_width: Option<usize>,
        min_height: Option<usize>,
    ) -> PyResult<Py<PyBytes>> {
        let result = py
            .allow_threads(|| {
                self.inner
                    .baseline_thumbnail(width, height, min_width, min_height)
            })
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &result).into())
    }

    fn __repr__(&self) -> String {
        format!(
            "Comparison({}x{}, {} pixels)",
            self.inner.width(),
            self.inner.height(),
            self.inner.width() * self.inner.height()
        )
    }
}

#[pymodule]
fn pixelhog(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<ComparisonPy>()?;
    m.add_class::<ClustersResultPy>()?;
    m.add_class::<ClusterPy>()?;
    m.add_class::<BoundingBoxPy>()?;
    m.add_class::<RowAlignmentPy>()?;
    m.add_class::<RowSegmentPy>()?;
    m.add_class::<ShiftBandPy>()?;

    m.add_function(wrap_pyfunction!(thumbnail_py, m)?)?;
    m.add_function(wrap_pyfunction!(diff_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(diff_count_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(ssim_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(compare_batch_py, m)?)?;

    Ok(())
}
