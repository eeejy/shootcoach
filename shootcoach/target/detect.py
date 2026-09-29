"""Hole type (target mm coordinates) and the classic no-training detector.

The main detector is the photo-trained YOLO in shootcoach/target/photo.py; ClassicHoleDetector is a
fallback that works on the flattened target image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shootcoach.config import TargetSpec



@dataclass
class Hole:
    x_mm: float          # paper mm, x → right
    y_mm: float          # paper mm, y → down
    r_mm: float
    conf: float = 1.0

    def as_dict(self):
        return {"x_mm": round(float(self.x_mm), 2), "y_mm": round(float(self.y_mm), 2),
                "r_mm": round(float(self.r_mm), 2), "conf": round(float(self.conf), 3)}


def _nms_holes(holes: list[Hole], min_dist_mm: float) -> list[Hole]:
    holes = sorted(holes, key=lambda h: -h.conf)
    kept: list[Hole] = []
    for h in holes:
        if all(np.hypot(h.x_mm - k.x_mm, h.y_mm - k.y_mm) >= min_dist_mm for k in kept):
            kept.append(h)
    return kept


class ClassicHoleDetector:
    """Fallback without a trained model (morphological top-hat).

    On white paper a hole is a small blob darker than its surroundings; inside the black
    aiming area it is lighter. Closing/opening with a kernel larger than a hole recovers the
    local background, so lighting gradients cancel out and thin ring lines are filtered away.
    """

    def detect(self, rect: np.ndarray, spec: TargetSpec) -> list[Hole]:
        ppm = spec.px_per_mm
        gray = cv2.GaussianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        r_px = spec.bullet_diameter_mm / 2 * ppm
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(r_px * 5) | 1, int(r_px * 5) | 1))
        dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)      # dark blobs on light paper
        light = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)       # light blobs inside the black
        # The narrow white bands between ring lines look like "light blobs" too, so the top-hat
        # is only valid inside the black aiming disc (known from the target geometry).
        yy, xx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
        rr = np.hypot(xx - spec.center_mm[0] * ppm, yy - spec.center_mm[1] * ppm)
        r_black = spec.ring_radius_mm(spec.black_from_ring) * ppm
        inside = rr < r_black - 1.0 * ppm
        resp = np.where(inside, light, dark)
        thr = 45.0
        mask = (resp > thr).astype(np.uint8) * 255
        # Remove thin structures (ring lines, numerals), then fill hole interiors (rim-only holes).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        filled = mask.copy()
        ff = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
        cv2.floodFill(filled, ff, (0, 0), 255)
        mask = mask | cv2.bitwise_not(filled)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area1 = np.pi * r_px ** 2
        holes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.4 * area1 or area > 12 * area1:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            fill_ratio = area / (np.pi * r * r + 1e-6)
            n_est = int(np.clip(round(area / (0.85 * area1)), 1, 10))
            if n_est == 1:
                if fill_ratio < 0.45:
                    continue
                holes.append(Hole(float(x) / ppm, float(y) / ppm, float(r) / ppm, 0.6))
            else:  # overlapping holes merged into one blob: split with k-means on its pixels
                blob = np.zeros_like(mask)
                cv2.drawContours(blob, [c], -1, 255, -1)
                pts = np.column_stack(np.nonzero(blob)[::-1]).astype(np.float32)
                crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
                _, _, centers = cv2.kmeans(pts, n_est, None, crit, 3, cv2.KMEANS_PP_CENTERS)
                for cx, cy in centers:
                    holes.append(Hole(float(cx) / ppm, float(cy) / ppm, r_px / ppm, 0.4))
        return _nms_holes(holes, spec.bullet_diameter_mm * 0.3)
