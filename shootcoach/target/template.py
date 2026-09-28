"""Render the printable target (rings + ArUco markers) in paper-mm coordinates."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from shootcoach.config import TargetSpec


def _aruco_dict(spec: TargetSpec):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.aruco_dict))


def render_target(spec: TargetSpec, px_per_mm: float | None = None, with_markers: bool = True) -> np.ndarray:
    """Return a BGR image of the blank target at `px_per_mm` resolution."""
    ppm = px_per_mm or spec.px_per_mm
    w, h = int(round(spec.paper_mm[0] * ppm)), int(round(spec.paper_mm[1] * ppm))
    img = np.full((h, w, 3), 255, np.uint8)
    cx, cy = spec.center_mm[0] * ppm, spec.center_mm[1] * ppm
    center = (int(round(cx)), int(round(cy)))
    line = max(1, int(round(0.35 * ppm)))

    black_r = spec.ring_radius_mm(spec.black_from_ring) * ppm
    cv2.circle(img, center, int(round(black_r)), (0, 0, 0), -1, cv2.LINE_AA)
    for ring in range(10, 10 - spec.ring_count, -1):
        r = int(round(spec.ring_radius_mm(ring) * ppm))
        color = (255, 255, 255) if ring >= spec.black_from_ring else (0, 0, 0)
        cv2.circle(img, center, r, color, line, cv2.LINE_AA)
    # Ring numbers on the horizontal axis (distractors for the detector, like real targets).
    font_scale = 0.9 * ppm / 4
    for ring in range(10 - spec.ring_count + 1, 10):
        r_mid = (spec.ring_radius_mm(ring) + spec.ring_radius_mm(ring + 1)) / 2 * ppm
        color = (255, 255, 255) if ring >= spec.black_from_ring else (0, 0, 0)
        for sign in (-1, 1):
            org = (int(cx + sign * r_mid - 6 * ppm / 4), int(cy + 6 * ppm / 4))
            cv2.putText(img, str(ring), org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, line // 1), cv2.LINE_AA)

    if with_markers:
        d = _aruco_dict(spec)
        for mid, (mx, my, size) in spec.markers.items():
            side = int(round(size * ppm))
            marker = cv2.aruco.generateImageMarker(d, mid, side)
            x0 = int(round((mx - size / 2) * ppm))
            y0 = int(round((my - size / 2) * ppm))
            img[y0:y0 + side, x0:x0 + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            cv2.putText(img, f"ID{mid}", (x0, y0 + side + int(4 * ppm)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * ppm / 4 * 1.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def save_printable(spec: TargetSpec, out_dir: str | Path, dpi: int = 300) -> tuple[Path, Path]:
    """Write PNG + PDF at true physical size (print at 100%, no scaling)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ppm = dpi / 25.4
    img = render_target(spec, px_per_mm=ppm)
    png = out_dir / f"{spec.name}_target.png"
    pdf = out_dir / f"{spec.name}_target.pdf"
    cv2.imwrite(str(png), img)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(pdf, "PDF", resolution=dpi)
    return png, pdf
