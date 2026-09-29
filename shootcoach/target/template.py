"""Render a clean target image (rings, numbers, centre cross) in paper-mm coordinates — used for synthetic tests and demos."""
from __future__ import annotations


import cv2
import numpy as np

from shootcoach.config import TargetSpec


def render_target(spec: TargetSpec, px_per_mm: float | None = None) -> np.ndarray:
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
    # Ring numbers on both axes and a centre cross, like the real KCG/police target.
    font_scale = spec.ring_step_mm * ppm / 45
    thick = max(1, int(round(font_scale * 1.6)))
    for ring in range(10 - spec.ring_count + 1, 10):
        r_mid = (spec.ring_radius_mm(ring) + spec.ring_radius_mm(ring + 1)) / 2 * ppm
        color = (255, 255, 255) if ring >= spec.black_from_ring else (0, 0, 0)
        for sign in (-1, 1):
            org = (int(cx + sign * r_mid - 9 * font_scale), int(cy + 10 * font_scale))
            cv2.putText(img, str(ring), org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)
            org = (int(cx - 9 * font_scale), int(cy + sign * r_mid + 10 * font_scale))
            cv2.putText(img, str(ring), org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)

    arm = int(round(spec.ring10_radius_mm * 0.45 * ppm))                 # centre cross (+)
    cv2.line(img, (center[0] - arm, center[1]), (center[0] + arm, center[1]), (255, 255, 255), line + 1, cv2.LINE_AA)
    cv2.line(img, (center[0], center[1] - arm), (center[0], center[1] + arm), (255, 255, 255), line + 1, cv2.LINE_AA)
    return img
