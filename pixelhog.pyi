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

__version__: str

__all__ = [
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
