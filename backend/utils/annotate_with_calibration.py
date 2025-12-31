# roomprintz-compositor/backend/utils/annotate_with_calibration.py

from __future__ import annotations

import io
import math
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageDraw


RED = (255, 0, 0, 255)  # pure red #FF0000


def annotate_with_calibration(
    image_bytes: bytes,
    calibration: Optional[Dict[str, Any]],
    *,
    line_width_px: Optional[int] = None,
    arrow_length_px: Optional[int] = None,
    arrow_width_px: Optional[int] = None,
    output_format: Optional[str] = None,
) -> bytes:
    """
    Bake ONLY the user-confirmed Segment measurement into the image:
      - solid pure red line (#FF0000)
      - arrowheads at both ends
      - no text, no labels, no height line

    Supports segment endpoints in either:
      A) Normalized coords: p1/p2 as {x: 0..1, y: 0..1}
      B) Pixel coords:      p1/p2 as {x: px,   y: px}  when meta.coordSpace == "px"
      C) Auto-detect: if values look > 1.5 we treat as pixels

    Returns annotated bytes (same format as input unless output_format provided).
    """
    img = Image.open(io.BytesIO(image_bytes))
    orig_format = img.format  # capture before any conversion
    img = _ensure_rgba(img)

    seg = _pick_segment(calibration)
    if not seg:
        return _save_image_bytes(img, fmt=output_format or orig_format)

    w, h = img.size

    p1 = seg.get("p1") or {}
    p2 = seg.get("p2") or {}

    try:
        x1 = float(p1.get("x", 0.0))
        y1 = float(p1.get("y", 0.0))
        x2 = float(p2.get("x", 0.0))
        y2 = float(p2.get("y", 0.0))
    except Exception:
        return _save_image_bytes(img, fmt=output_format or orig_format)

    # coord space detection
    meta = seg.get("meta") or {}
    coord_space = str(meta.get("coordSpace") or meta.get("coord_space") or "").strip().lower()

    is_pixels = False
    if coord_space in ("px", "pixel", "pixels"):
        is_pixels = True
    elif max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        # heuristic: normalized coords should be ~0..1
        is_pixels = True

    if is_pixels:
        ax = _clamp(x1, 0.0, float(w))
        ay = _clamp(y1, 0.0, float(h))
        bx = _clamp(x2, 0.0, float(w))
        by = _clamp(y2, 0.0, float(h))
    else:
        # clamp normalized -> pixels
        nx1, ny1, nx2, ny2 = _clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)
        ax, ay = nx1 * w, ny1 * h
        bx, by = nx2 * w, ny2 * h

    a = (ax, ay)
    b = (bx, by)

    # ignore degenerate segments
    if math.hypot(bx - ax, by - ay) < 2.0:
        return _save_image_bytes(img, fmt=output_format or orig_format)

    # default sizes scale with image
    base = max(w, h)
    lw = int(line_width_px if line_width_px is not None else max(3, round(base * 0.004)))
    al = int(arrow_length_px if arrow_length_px is not None else max(16, round(base * 0.02)))
    aw = float(arrow_width_px if arrow_width_px is not None else max(10, round(base * 0.012)))

    draw = ImageDraw.Draw(img)

    # main line
    draw.line([a, b], fill=RED, width=lw)

    # arrowheads (filled triangles)
    _draw_arrowhead(draw, tip=a, tail=b, length=al, half_width=aw / 2.0, fill=RED)
    _draw_arrowhead(draw, tip=b, tail=a, length=al, half_width=aw / 2.0, fill=RED)

    return _save_image_bytes(img, fmt=output_format or orig_format)


def _pick_segment(calibration: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(calibration, dict):
        return None
    ms = calibration.get("measurements") or []
    for m in ms:
        if isinstance(m, dict) and (m.get("kind") or "").strip().lower() == "segment":
            p1 = m.get("p1") or {}
            p2 = m.get("p2") or {}
            if isinstance(p1, dict) and isinstance(p2, dict) and ("x" in p1 and "y" in p1 and "x" in p2 and "y" in p2):
                return m
    return None


def _draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    *,
    tip: Tuple[float, float],
    tail: Tuple[float, float],
    length: float,
    half_width: float,
    fill: Tuple[int, int, int, int],
) -> None:
    tx, ty = tip
    sx, sy = tail

    dx = tx - sx
    dy = ty - sy
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return

    # unit direction from tail -> tip
    ux = dx / d
    uy = dy / d

    # base center point of triangle
    bx = tx - ux * length
    by = ty - uy * length

    # perpendicular unit
    px = -uy
    py = ux

    p1 = (bx + px * half_width, by + py * half_width)
    p2 = (bx - px * half_width, by - py * half_width)
    p3 = (tx, ty)

    draw.polygon([p1, p2, p3], fill=fill)


def _ensure_rgba(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        return img
    return img.convert("RGBA")


def _save_image_bytes(img: Image.Image, fmt: Optional[str] = None) -> bytes:
    out = io.BytesIO()
    format_to_use = (fmt or "PNG")
    format_to_use = str(format_to_use).upper()

    if format_to_use in ("JPG", "JPEG"):
        img_rgb = img.convert("RGB")
        img_rgb.save(out, format="JPEG", quality=95, subsampling=0, optimize=True)
    else:
        # Default: PNG is safest for RGBA
        img.save(out, format=format_to_use)

    return out.getvalue()


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
