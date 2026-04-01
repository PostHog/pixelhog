import io
import math

import pytest
from PIL import Image, ImageDraw

from pixelhog import compute_ssim, pixelmatch


def decode_png_rgba(png_bytes: bytes) -> tuple[bytes, int, int]:
    with Image.open(io.BytesIO(png_bytes)) as img:
        rgba = img.convert("RGBA")
        return rgba.tobytes(), rgba.width, rgba.height


def encode_png_rgba(raw: bytes, width: int, height: int) -> bytes:
    img = Image.frombytes("RGBA", (width, height), raw)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def solid_png(width: int, height: int, color: tuple[int, int, int, int]) -> bytes:
    img = Image.new("RGBA", (width, height), color)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def pad_rgba(raw: bytes, width: int, height: int, target_width: int, target_height: int) -> bytes:
    out = bytearray(target_width * target_height * 4)
    src_stride = width * 4
    dst_stride = target_width * 4
    for row in range(height):
        src_start = row * src_stride
        src_end = src_start + src_stride
        dst_start = row * dst_stride
        out[dst_start : dst_start + src_stride] = raw[src_start:src_end]
    return bytes(out)


def color_delta_reference(img1: bytes, img2: bytes, k: int, m: int, y_only: bool) -> float:
    r1, g1, b1, a1 = img1[k], img1[k + 1], img1[k + 2], img1[k + 3]
    r2, g2, b2, a2 = img2[m], img2[m + 1], img2[m + 2], img2[m + 3]

    dr = float(r1 - r2)
    dg = float(g1 - g2)
    db = float(b1 - b2)
    da = float(a1 - a2)

    if dr == 0.0 and dg == 0.0 and db == 0.0 and da == 0.0:
        return 0.0

    if a1 < 255 or a2 < 255:
        rb = 48.0 + 159.0 * (k % 2)
        gb = 48.0 + 159.0 * (int(math.floor(k / 1.618033988749895)) % 2)
        bb = 48.0 + 159.0 * (int(math.floor(k / 2.618033988749895)) % 2)

        dr = (r1 * a1 - r2 * a2 - rb * da) / 255.0
        dg = (g1 * a1 - g2 * a2 - gb * da) / 255.0
        db = (b1 * a1 - b2 * a2 - bb * da) / 255.0

    y = dr * 0.29889531 + dg * 0.58662247 + db * 0.11448223
    if y_only:
        return y

    i = dr * 0.59597799 - dg * 0.27417610 - db * 0.32180189
    q = dr * 0.21147017 - dg * 0.52261711 + db * 0.31114694
    delta = 0.5053 * y * y + 0.299 * i * i + 0.1957 * q * q
    return -delta if y > 0 else delta


def has_many_siblings_reference(img: list[int], x1: int, y1: int, width: int, height: int) -> bool:
    x0 = max(x1 - 1, 0)
    y0 = max(y1 - 1, 0)
    x2 = min(x1 + 1, width - 1)
    y2 = min(y1 + 1, height - 1)

    val = img[y1 * width + x1]
    zeroes = 1 if (x1 == x0 or x1 == x2 or y1 == y0 or y1 == y2) else 0

    for x in range(x0, x2 + 1):
        for y in range(y0, y2 + 1):
            if x == x1 and y == y1:
                continue
            if val == img[y * width + x]:
                zeroes += 1
                if zeroes > 2:
                    return True
    return False


def antialiased_reference(
    img: bytes,
    x1: int,
    y1: int,
    width: int,
    height: int,
    a32: list[int],
    b32: list[int],
) -> bool:
    x0 = max(x1 - 1, 0)
    y0 = max(y1 - 1, 0)
    x2 = min(x1 + 1, width - 1)
    y2 = min(y1 + 1, height - 1)

    pos = y1 * width + x1
    zeroes = 1 if (x1 == x0 or x1 == x2 or y1 == y0 or y1 == y2) else 0

    min_delta = 0.0
    max_delta = 0.0
    min_x = min_y = max_x = max_y = 0

    for x in range(x0, x2 + 1):
        for y in range(y0, y2 + 1):
            if x == x1 and y == y1:
                continue

            delta = color_delta_reference(img, img, pos * 4, (y * width + x) * 4, True)
            if delta == 0.0:
                zeroes += 1
                if zeroes > 2:
                    return False
            elif delta < min_delta:
                min_delta = delta
                min_x, min_y = x, y
            elif delta > max_delta:
                max_delta = delta
                max_x, max_y = x, y

    if min_delta == 0.0 or max_delta == 0.0:
        return False

    return (
        has_many_siblings_reference(a32, min_x, min_y, width, height)
        and has_many_siblings_reference(b32, min_x, min_y, width, height)
    ) or (
        has_many_siblings_reference(a32, max_x, max_y, width, height)
        and has_many_siblings_reference(b32, max_x, max_y, width, height)
    )


def u32_values(raw: bytes) -> list[int]:
    return [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]


def pixelmatch_reference_png(
    baseline_png: bytes,
    current_png: bytes,
    threshold: float = 0.1,
    include_aa: bool = False,
) -> tuple[int, int, int]:
    raw1, w1, h1 = decode_png_rgba(baseline_png)
    raw2, w2, h2 = decode_png_rgba(current_png)
    width = max(w1, w2)
    height = max(h1, h2)

    img1 = pad_rgba(raw1, w1, h1, width, height)
    img2 = pad_rgba(raw2, w2, h2, width, height)

    a32 = u32_values(img1)
    b32 = u32_values(img2)

    max_delta = 35215.0 * threshold * threshold
    diff = 0

    for y in range(height):
        for x in range(width):
            i = y * width + x
            pos = i * 4
            delta = 0.0 if a32[i] == b32[i] else color_delta_reference(img1, img2, pos, pos, False)

            if abs(delta) > max_delta:
                excluded_aa = (not include_aa) and (
                    antialiased_reference(img1, x, y, width, height, a32, b32)
                    or antialiased_reference(img2, x, y, width, height, b32, a32)
                )
                if not excluded_aa:
                    diff += 1

    return diff, width, height


def rgba_to_gray(raw: bytes) -> list[float]:
    out: list[float] = []
    for i in range(0, len(raw), 4):
        r = raw[i]
        g = raw[i + 1]
        b = raw[i + 2]
        out.append(r * 0.299 + g * 0.587 + b * 0.114)
    return out


def reflect_index(i: int, length: int) -> int:
    if length <= 1:
        return 0

    idx = i
    while idx < 0 or idx >= length:
        if idx < 0:
            idx = -idx
        else:
            idx = 2 * length - idx - 2
    return idx


def ssim_reference_png(baseline_png: bytes, current_png: bytes) -> float:
    c1 = 6.5025
    c2 = 58.5225
    win = 11
    half = win // 2

    raw1, w1, h1 = decode_png_rgba(baseline_png)
    raw2, w2, h2 = decode_png_rgba(current_png)
    width = max(w1, w2)
    height = max(h1, h2)

    img1 = pad_rgba(raw1, w1, h1, width, height)
    img2 = pad_rgba(raw2, w2, h2, width, height)

    g1 = rgba_to_gray(img1)
    g2 = rgba_to_gray(img2)

    if width < win or height < win:
        n = len(g1)
        mu1 = sum(g1) / n
        mu2 = sum(g2) / n

        var1 = sum((v - mu1) ** 2 for v in g1) / n
        var2 = sum((v - mu2) ** 2 for v in g2) / n
        cov = sum((a - mu1) * (b - mu2) for a, b in zip(g1, g2)) / n

        numerator = (2 * mu1 * mu2 + c1) * (2 * cov + c2)
        denominator = (mu1 * mu1 + mu2 * mu2 + c1) * (var1 + var2 + c2)
        return max(0.0, min(1.0, numerator / denominator if denominator else 1.0))

    score = 0.0
    total = width * height

    for y in range(height):
        for x in range(width):
            vals1 = []
            vals2 = []
            for dy in range(-half, half + 1):
                yy = reflect_index(y + dy, height)
                row_offset = yy * width
                for dx in range(-half, half + 1):
                    xx = reflect_index(x + dx, width)
                    idx = row_offset + xx
                    vals1.append(g1[idx])
                    vals2.append(g2[idx])

            n = float(len(vals1))
            mu1 = sum(vals1) / n
            mu2 = sum(vals2) / n

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = sum(v * v for v in vals1) / n - mu1_sq
            sigma2_sq = sum(v * v for v in vals2) / n - mu2_sq
            sigma12 = sum(a * b for a, b in zip(vals1, vals2)) / n - mu1_mu2

            numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            score += numerator / denominator if denominator else 1.0

    return max(0.0, min(1.0, score / total))


def test_identical_images_zero_diff() -> None:
    baseline = solid_png(20, 15, (30, 40, 50, 255))
    current = baseline

    diff_png, diff_count, width, height = pixelmatch(baseline, current)

    assert diff_count == 0
    assert (width, height) == (20, 15)

    with Image.open(io.BytesIO(diff_png)) as img:
        assert img.size == (20, 15)


def test_completely_different_images_full_diff() -> None:
    baseline = solid_png(10, 10, (0, 0, 0, 255))
    current = solid_png(10, 10, (255, 255, 255, 255))

    _, diff_count, width, height = pixelmatch(baseline, current)

    assert (width, height) == (10, 10)
    assert diff_count == 100


def test_partial_diff() -> None:
    width, height = 24, 12
    baseline_img = Image.new("RGBA", (width, height), (255, 0, 0, 255))
    current_img = Image.new("RGBA", (width, height), (255, 0, 0, 255))

    draw = ImageDraw.Draw(current_img)
    draw.rectangle((width // 2, 0, width - 1, height - 1), fill=(0, 0, 255, 255))

    baseline_io = io.BytesIO()
    current_io = io.BytesIO()
    baseline_img.save(baseline_io, format="PNG")
    current_img.save(current_io, format="PNG")

    _, diff_count, _, _ = pixelmatch(baseline_io.getvalue(), current_io.getvalue())

    assert diff_count == (width * height) // 2


def test_different_sizes_pads_to_larger() -> None:
    baseline = solid_png(10, 8, (200, 0, 0, 255))
    current = solid_png(12, 10, (200, 0, 0, 255))

    _, diff_count, width, height = pixelmatch(baseline, current)

    assert (width, height) == (12, 10)
    assert diff_count == 40


def test_threshold_controls_sensitivity() -> None:
    baseline = solid_png(16, 16, (120, 120, 120, 255))
    current = solid_png(16, 16, (132, 132, 132, 255))

    _, low_count, _, _ = pixelmatch(baseline, current, threshold=0.01)
    _, high_count, _, _ = pixelmatch(baseline, current, threshold=0.3)

    assert low_count > high_count
    assert high_count == 0


def test_diff_image_is_valid_png() -> None:
    baseline = solid_png(8, 8, (0, 255, 0, 255))
    current = solid_png(8, 8, (255, 0, 0, 255))

    diff_png, _, _, _ = pixelmatch(baseline, current)

    with Image.open(io.BytesIO(diff_png)) as img:
        assert img.format == "PNG"
        assert img.size == (8, 8)


def test_identical_images_score_one() -> None:
    baseline = solid_png(48, 48, (120, 130, 140, 255))
    current = baseline

    score = compute_ssim(baseline, current)
    assert score == pytest.approx(1.0, abs=1e-12)


def test_completely_different_images_low_score() -> None:
    baseline = solid_png(48, 48, (0, 0, 0, 255))
    current = solid_png(48, 48, (255, 255, 255, 255))

    score = compute_ssim(baseline, current)
    assert score < 0.1


def test_slight_difference_high_score() -> None:
    img1 = Image.new("RGBA", (120, 80), (180, 180, 180, 255))
    img2 = img1.copy()
    draw = ImageDraw.Draw(img2)
    draw.rectangle((59, 39, 60, 40), fill=(170, 170, 170, 255))

    out1 = io.BytesIO()
    out2 = io.BytesIO()
    img1.save(out1, format="PNG")
    img2.save(out2, format="PNG")

    score = compute_ssim(out1.getvalue(), out2.getvalue())
    assert score > 0.98


def test_small_images_below_window_size() -> None:
    baseline = solid_png(5, 5, (100, 100, 100, 255))
    current = solid_png(5, 5, (130, 130, 130, 255))

    score = compute_ssim(baseline, current)

    assert 0.0 <= score <= 1.0
    assert score < 1.0


def test_different_sizes_pads_to_larger_ssim() -> None:
    baseline = solid_png(9, 9, (255, 255, 255, 255))
    current = solid_png(14, 14, (255, 255, 255, 255))

    score = compute_ssim(baseline, current)

    assert 0.0 <= score <= 1.0
    assert score < 1.0


def test_tall_page_change_caught_by_ssim() -> None:
    width, height = 240, 3000

    baseline_img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    current_img = baseline_img.copy()

    draw = ImageDraw.Draw(current_img)
    draw.rectangle((160, 2900, 219, 2919), fill=(0, 0, 0, 255))

    baseline_io = io.BytesIO()
    current_io = io.BytesIO()
    baseline_img.save(baseline_io, format="PNG")
    current_img.save(current_io, format="PNG")

    baseline = baseline_io.getvalue()
    current = current_io.getvalue()

    _, diff_count, _, _ = pixelmatch(baseline, current)
    score = compute_ssim(baseline, current)

    diff_ratio = diff_count / (width * height)
    assert diff_ratio < 0.005
    assert score < 0.999


def test_new_element_below_pixelmatch_threshold() -> None:
    width, height = 500, 1500

    baseline_img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    current_img = baseline_img.copy()

    draw = ImageDraw.Draw(current_img)
    draw.rectangle((470, 1470, 481, 1481), fill=(20, 20, 20, 255))

    baseline_io = io.BytesIO()
    current_io = io.BytesIO()
    baseline_img.save(baseline_io, format="PNG")
    current_img.save(current_io, format="PNG")

    baseline = baseline_io.getvalue()
    current = current_io.getvalue()

    _, diff_count, _, _ = pixelmatch(baseline, current)
    score = compute_ssim(baseline, current)

    diff_ratio = diff_count / (width * height)
    assert diff_ratio < 0.001
    assert score < 1.0


def test_cross_validation_against_python_reference() -> None:
    width, height = 37, 29

    baseline_pixels: list[tuple[int, int, int, int]] = []
    current_pixels: list[tuple[int, int, int, int]] = []

    for y in range(height):
        for x in range(width):
            r = (x * 7 + y * 3) % 256
            g = (x * 5 + y * 11) % 256
            b = (x * 13 + y * 17) % 256
            a = 255 if (x + y) % 5 else 180

            baseline_pixels.append((r, g, b, a))

            r2 = r
            g2 = g
            b2 = b
            a2 = a

            if 8 <= x <= 28 and 6 <= y <= 22:
                r2 = (r + 30) % 256
                g2 = (g + 10) % 256

            if (x + y) % 9 == 0:
                a2 = max(60, a - 80)

            current_pixels.append((r2, g2, b2, a2))

    baseline_img = Image.new("RGBA", (width, height))
    baseline_img.putdata(baseline_pixels)
    current_img = Image.new("RGBA", (width, height))
    current_img.putdata(current_pixels)

    baseline_io = io.BytesIO()
    current_io = io.BytesIO()
    baseline_img.save(baseline_io, format="PNG")
    current_img.save(current_io, format="PNG")

    baseline = baseline_io.getvalue()
    current = current_io.getvalue()

    _, rust_diff_count, rust_w, rust_h = pixelmatch(baseline, current)
    ref_diff_count, ref_w, ref_h = pixelmatch_reference_png(baseline, current)

    assert (rust_w, rust_h) == (ref_w, ref_h)
    assert rust_diff_count == ref_diff_count

    rust_ssim = compute_ssim(baseline, current)
    ref_ssim = ssim_reference_png(baseline, current)

    assert abs(rust_ssim - ref_ssim) <= 0.01


def test_invalid_inputs_raise_value_error() -> None:
    valid = solid_png(2, 2, (10, 20, 30, 255))

    with pytest.raises(ValueError):
        pixelmatch(valid, valid, threshold=-0.01)

    with pytest.raises(ValueError):
        pixelmatch(valid, valid, alpha=1.2)

    with pytest.raises(ValueError):
        pixelmatch(b"not-a-png", valid)

    with pytest.raises(ValueError):
        compute_ssim(valid, b"not-a-png")
