from __future__ import annotations

from typing import Optional, Sequence

PngPair = tuple[bytes, bytes]
DiffResult = tuple[bytes, int, int, int]
DiffCountResult = tuple[int, int, int]
CompareResult = tuple[int, float, int, int, Optional[bytes], Optional[bytes]]

def thumbnail(
    png_bytes: bytes,
    width: int = 200,
    height: Optional[int] = None,
) -> bytes: ...
def diff(
    baseline_png: bytes,
    current_png: bytes,
    threshold: float = 0.1,
    alpha: float = 0.1,
    include_aa: bool = False,
    diff_color: tuple[int, int, int] = (255, 0, 0),
    aa_color: tuple[int, int, int] = (255, 255, 0),
    diff_color_alt: Optional[tuple[int, int, int]] = None,
) -> DiffResult: ...
def diff_count(
    baseline_png: bytes,
    current_png: bytes,
    threshold: float = 0.1,
    include_aa: bool = False,
) -> DiffCountResult: ...
def ssim(
    baseline_png: bytes,
    current_png: bytes,
) -> float: ...
def compare(
    baseline_png: bytes,
    current_png: bytes,
    threshold: float = 0.1,
    alpha: float = 0.1,
    include_aa: bool = False,
    diff_color: tuple[int, int, int] = (255, 0, 0),
    aa_color: tuple[int, int, int] = (255, 255, 0),
    diff_color_alt: Optional[tuple[int, int, int]] = None,
    return_diff: bool = False,
    thumbnail_width: Optional[int] = None,
    thumbnail_height: Optional[int] = None,
) -> CompareResult: ...
def diff_rgba(
    baseline_rgba: bytes,
    baseline_width: int,
    baseline_height: int,
    current_rgba: bytes,
    current_width: int,
    current_height: int,
    threshold: float = 0.1,
    alpha: float = 0.1,
    include_aa: bool = False,
    diff_color: tuple[int, int, int] = (255, 0, 0),
    aa_color: tuple[int, int, int] = (255, 255, 0),
    diff_color_alt: Optional[tuple[int, int, int]] = None,
) -> DiffResult: ...
def diff_count_rgba(
    baseline_rgba: bytes,
    baseline_width: int,
    baseline_height: int,
    current_rgba: bytes,
    current_width: int,
    current_height: int,
    threshold: float = 0.1,
    include_aa: bool = False,
) -> DiffCountResult: ...
def ssim_rgba(
    baseline_rgba: bytes,
    baseline_width: int,
    baseline_height: int,
    current_rgba: bytes,
    current_width: int,
    current_height: int,
) -> float: ...
def compare_rgba(
    baseline_rgba: bytes,
    baseline_width: int,
    baseline_height: int,
    current_rgba: bytes,
    current_width: int,
    current_height: int,
    threshold: float = 0.1,
    alpha: float = 0.1,
    include_aa: bool = False,
    diff_color: tuple[int, int, int] = (255, 0, 0),
    aa_color: tuple[int, int, int] = (255, 255, 0),
    diff_color_alt: Optional[tuple[int, int, int]] = None,
    return_diff: bool = False,
    thumbnail_width: Optional[int] = None,
    thumbnail_height: Optional[int] = None,
) -> CompareResult: ...
def diff_batch(
    pairs: Sequence[PngPair],
    threshold: float = 0.1,
    alpha: float = 0.1,
    include_aa: bool = False,
    diff_color: tuple[int, int, int] = (255, 0, 0),
    aa_color: tuple[int, int, int] = (255, 255, 0),
    diff_color_alt: Optional[tuple[int, int, int]] = None,
) -> list[DiffResult]: ...
def diff_count_batch(
    pairs: Sequence[PngPair],
    threshold: float = 0.1,
    include_aa: bool = False,
) -> list[DiffCountResult]: ...
def ssim_batch(
    pairs: Sequence[PngPair],
) -> list[float]: ...
def compare_batch(
    pairs: Sequence[PngPair],
    threshold: float = 0.1,
    alpha: float = 0.1,
    include_aa: bool = False,
    diff_color: tuple[int, int, int] = (255, 0, 0),
    aa_color: tuple[int, int, int] = (255, 255, 0),
    diff_color_alt: Optional[tuple[int, int, int]] = None,
    return_diff: bool = False,
    thumbnail_width: Optional[int] = None,
    thumbnail_height: Optional[int] = None,
) -> list[CompareResult]: ...

class BoundingBox:
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

class Cluster:
    @property
    def bbox(self) -> BoundingBox: ...
    @property
    def pixel_count(self) -> int: ...
    @property
    def centroid(self) -> tuple[float, float]: ...
    @property
    def merged_from(self) -> int: ...

class ClustersResult:
    @property
    def clusters(self) -> list[Cluster]: ...
    @property
    def total_clusters(self) -> int: ...
    @property
    def truncated(self) -> bool: ...
    def __len__(self) -> int: ...

class Comparison:
    def __init__(self, baseline_png: bytes, current_png: bytes) -> None: ...
    @staticmethod
    def from_rgba(
        baseline_rgba: bytes,
        baseline_width: int,
        baseline_height: int,
        current_rgba: bytes,
        current_width: int,
        current_height: int,
    ) -> Comparison: ...
    @staticmethod
    def batch(pairs: Sequence[PngPair]) -> list[Comparison]: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def size_mismatch(self) -> bool: ...
    @property
    def baseline_size(self) -> tuple[int, int]: ...
    @property
    def current_size(self) -> tuple[int, int]: ...
    def diff_count(
        self,
        threshold: float = 0.1,
        include_aa: bool = False,
    ) -> int: ...
    def diff_count_capped(
        self,
        max_diffs: int,
        threshold: float = 0.1,
        include_aa: bool = False,
    ) -> int: ...
    def ssim(self) -> float: ...
    def clusters(
        self,
        threshold: float = 0.1,
        include_aa: bool = False,
        min_pixels: int = 16,
        min_side: int = 0,
        dilation: int = 4,
        max_clusters: Optional[int] = None,
        merge_gap: int = 0,
        merge_overlap: float = 0.5,
    ) -> ClustersResult: ...
    def diff_image(
        self,
        threshold: float = 0.1,
        alpha: float = 0.1,
        include_aa: bool = False,
        diff_color: tuple[int, int, int] = (255, 0, 0),
        aa_color: tuple[int, int, int] = (255, 255, 0),
        diff_color_alt: Optional[tuple[int, int, int]] = None,
    ) -> bytes: ...
    def current_thumbnail(
        self,
        width: int = 200,
        height: Optional[int] = None,
        min_width: Optional[int] = None,
        min_height: Optional[int] = None,
    ) -> bytes: ...
    def baseline_thumbnail(
        self,
        width: int = 200,
        height: Optional[int] = None,
        min_width: Optional[int] = None,
        min_height: Optional[int] = None,
    ) -> bytes: ...

__version__: str

__all__ = [
    "BoundingBox",
    "Cluster",
    "ClustersResult",
    "Comparison",
    "thumbnail",
    "diff",
    "diff_count",
    "ssim",
    "compare",
    "diff_rgba",
    "diff_count_rgba",
    "ssim_rgba",
    "compare_rgba",
    "diff_batch",
    "diff_count_batch",
    "ssim_batch",
    "compare_batch",
]
