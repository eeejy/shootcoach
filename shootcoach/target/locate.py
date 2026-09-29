"""마커 없이 원형 표적 찾기: 검은 조준 원(7점 경계)을 타원으로 맞추고, 그 타원을 원으로 펴는 변환을 만든다.

- 표적은 세로로 걸려 있다고 보고, 사진의 '위'를 표적의 '위'로 유지한다.
- 원근 왜곡이 심하지 않은 사진(정면에서 조금 비스듬한 정도)에서는 타원→원 변환으로 충분하다.
- 탄공은 원본 사진에서 찾고, 좌표만 이 변환으로 표적 mm 좌표로 옮긴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shootcoach.config import TargetSpec


class TargetNotFound(RuntimeError):
    pass


@dataclass
class TargetFrame:
    center_px: tuple[float, float]     # 검은 원 중심 (사진 px)
    axes_px: tuple[float, float]       # 타원 반지름 (장축, 단축) px
    angle_deg: float
    A: np.ndarray                      # 2x3 affine: 사진 px → 표적 mm (중심 기준, x 오른쪽, y 아래)
    fill: float                        # 타원 채움 정도 (1에 가까울수록 확실)
    mm_per_px: float

    def to_target_mm(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, float).reshape(-1, 2)
        return pts @ self.A[:, :2].T + self.A[:, 2]

    def paper_mm(self, pts_px: np.ndarray, spec: TargetSpec) -> np.ndarray:
        """Target-centred mm → paper mm (x right, y down) used by scoring.py."""
        return self.to_target_mm(pts_px) + np.asarray(spec.center_mm)


def _refine_ellipse(pts: np.ndarray, ell, iters: int = 4):
    """Refit on contour points that lie on the ellipse.

    Holes on the edge of the black disc either bite into it (bright holes inside) or stick out of it
    (dark holes just outside merge with the disc at a bright threshold). Both pull a plain or hull fit
    towards the shots. Dropping points whose normalised radius is off, then refitting, removes both.
    """
    pts = pts.reshape(-1, 2).astype(np.float64)
    for tol in np.linspace(0.08, 0.025, iters):
        (cx, cy), (MA, ma), ang = ell
        t = np.deg2rad(ang)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u = dx * np.cos(t) + dy * np.sin(t)
        v = -dx * np.sin(t) + dy * np.cos(t)
        rho = np.hypot(u / (MA / 2 + 1e-9), v / (ma / 2 + 1e-9))
        keep = np.abs(rho - np.median(rho)) < tol
        if keep.sum() < max(20, 0.3 * len(pts)):
            break
        ell = cv2.fitEllipse(pts[keep].astype(np.float32))
    return ell


def _candidate_discs(gray: np.ndarray):
    """Dark, filled, ellipse-shaped regions. Tries several thresholds and looks inside other regions too,
    so a dark wall or table behind the paper does not swallow the black aiming disc."""
    h, w = gray.shape
    k = max(5, int(round(min(h, w) / 60)) | 1)
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seen = []
    for t in sorted({int(otsu), 60, 90, 120}):
        bw = (g < t).astype(np.uint8) * 255
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            area = cv2.contourArea(c)
            if area < 0.0015 * h * w or area > 0.8 * h * w or len(c) < 20:
                continue
            hull = cv2.convexHull(c)                  # 가장자리 탄공이 파먹은 부분을 무시
            if len(hull) < 5:
                continue
            (cx, cy), (MA, ma), ang = cv2.fitEllipse(hull)
            ell_area = np.pi * MA * ma / 4
            fill = area / (ell_area + 1e-6)
            ratio = min(MA, ma) / max(MA, ma)
            if fill < 0.75 or ratio < 0.45:
                continue
            (cx, cy), (MA, ma), ang = _refine_ellipse(c, ((cx, cy), (MA, ma), ang))
            if cx - MA / 2 < -0.15 * MA or cy - ma / 2 < -0.15 * ma or cx + MA / 2 > w + 0.15 * MA or cy + ma / 2 > h + 0.15 * ma:
                continue
            # 안쪽이 실제로 어두운지 (밝은 종이 영역을 뒤집어 잡은 경우 제외)
            m = np.zeros_like(g)
            cv2.ellipse(m, ((cx, cy), (MA * 0.9, ma * 0.9), ang), 255, -1)
            inner = float(g[m > 0].mean()) if (m > 0).any() else 255.0
            ring = np.zeros_like(g)
            cv2.ellipse(ring, ((cx, cy), (MA * 1.35, ma * 1.35), ang), 255, -1)
            cv2.ellipse(ring, ((cx, cy), (MA * 1.08, ma * 1.08), ang), 0, -1)
            outer = float(g[ring > 0].mean()) if (ring > 0).any() else 0.0
            if outer - inner < 40:                    # 검은 원 바로 바깥은 흰 종이여야 한다
                continue
            if any(np.hypot(cx - x, cy - y) < 0.1 * MA and abs(MA - M2) < 0.1 * MA for x, y, M2 in seen):
                continue
            seen.append((cx, cy, MA))
            yield area, fill, ((cx, cy), (MA, ma), ang)


def locate_target(image: np.ndarray, spec: TargetSpec) -> TargetFrame:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    best = None
    for area, fill, ell in _candidate_discs(gray):
        (cx, cy), _, _ = ell
        centrality = 1 - min(1.0, np.hypot(cx - w / 2, cy - h / 2) / (0.75 * np.hypot(w / 2, h / 2)))
        score = area * (0.5 + fill) * (0.5 + centrality)
        if best is None or score > best[0]:
            best = (score, fill, ell)
    if best is None:
        raise TargetNotFound("표적의 검은 원을 찾지 못했습니다. 표적 전체가 나오게 정면에서 다시 찍어 주세요.")
    _, fill, ((cx, cy), (d1, d2), ang) = best
    r_black_mm = spec.ring_radius_mm(spec.black_from_ring)
    # 타원 축을 원 반지름으로 맞추는 변환: 회전 → 축별 스케일 → 역회전 (사진의 위 = 표적의 위)
    t = np.radians(ang)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])   # 사진 → 타원 축 좌표
    S = np.diag([r_black_mm / (d1 / 2), r_black_mm / (d2 / 2)])
    M = R.T @ S @ R
    A = np.hstack([M, (-M @ np.array([cx, cy])).reshape(2, 1)])
    return TargetFrame((cx, cy), (d1 / 2, d2 / 2), ang, A, float(fill),
                       mm_per_px=float(r_black_mm / np.sqrt((d1 / 2) * (d2 / 2))))


def rectified_view(image: np.ndarray, frame: TargetFrame, spec: TargetSpec, px_per_mm: float | None = None) -> np.ndarray:
    """정면으로 편 표적 이미지 (표시·라벨링용)."""
    ppm = px_per_mm or spec.px_per_mm
    W, H = int(spec.paper_mm[0] * ppm), int(spec.paper_mm[1] * ppm)
    A3 = np.vstack([frame.A, [0, 0, 1]])
    to_canvas = np.array([[ppm, 0, spec.center_mm[0] * ppm], [0, ppm, spec.center_mm[1] * ppm], [0, 0, 1]])
    H_img_to_canvas = to_canvas @ A3
    return cv2.warpAffine(image, H_img_to_canvas[:2], (W, H), flags=cv2.INTER_LINEAR, borderValue=(235, 235, 235))
