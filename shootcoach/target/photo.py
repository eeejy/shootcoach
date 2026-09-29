"""원본 사진에서 탄공 찾기 (마커 없음) + 발수 보정.

- 모델은 공개 실사진 데이터(Roboflow, 표적을 폰으로 찍은 사진)로 학습한 YOLO. 원본 사진을 그대로 넣는다.
- 발수 보정 (서울청 STARS 과제의 핵심 요령): 한 장에 쏜 발수를 알면,
  검출 수가 모자랄 때 신뢰도 기준을 낮추고, 넘칠 때 높여 다시 고른다.
  그래도 모자라면 크기가 유난히 큰 탄공 박스(여러 발이 뭉친 것)를 여러 발로 나눈다.
- 결과는 사진 px 좌표. 표적 좌표 변환은 locate.TargetFrame 이 맡는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import REPO_ROOT

PHOTO_WEIGHTS = REPO_ROOT / "models" / "hole_detector_photo.pt"


@dataclass
class PxHole:
    x: float
    y: float
    r: float          # 반지름 px
    conf: float
    split: bool = False


def _nms_px(holes: list[PxHole], min_dist: float) -> list[PxHole]:
    kept: list[PxHole] = []
    for h in sorted(holes, key=lambda h: -h.conf):
        if all(np.hypot(h.x - k.x, h.y - k.y) >= min_dist for k in kept):
            kept.append(h)
    return kept


class PhotoHoleDetector:
    def __init__(self, weights: str | Path = PHOTO_WEIGHTS, imgsz: int = 960, device=None, augment: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = device
        self.augment = augment        # 좌우 뒤집기·배율 TTA: 해경 실사진 F1 0.85 → 0.88 (약 2~3배 느림)

    def raw(self, image: np.ndarray, conf_floor: float = 0.05) -> list[PxHole]:
        """All candidates down to a low confidence (filtered later)."""
        r = self.model.predict(image, conf=conf_floor, iou=0.6, imgsz=self.imgsz, device=self.device, augment=self.augment, verbose=False,
                               max_det=300)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append(PxHole((x1 + x2) / 2, (y1 + y2) / 2, ((x2 - x1) + (y2 - y1)) / 4, float(c)))
        return out

    def detect(self, image: np.ndarray, expected: int | None = None, inside=None, base_conf: float = 0.25) -> tuple[list[PxHole], dict]:
        """Returns holes and a small log of what the shot-count correction did.

        inside: optional callable (x, y) -> bool, keeps only candidates on the target paper.
        """
        cands = self.raw(image)
        if inside is not None:
            cands = [h for h in cands if inside(h.x, h.y)]
        med_r = float(np.median([h.r for h in cands])) if cands else 5.0
        cands = _nms_px(cands, med_r * 0.8)
        log = {"expected": expected, "base_conf": base_conf}
        picked = [h for h in cands if h.conf >= base_conf]
        if expected:
            if len(picked) < expected:                                 # 미탐: 기준을 낮춰 채운다
                for c in (0.18, 0.12, 0.08, 0.05):
                    picked = [h for h in cands if h.conf >= c]
                    log["conf"] = c
                    if len(picked) >= expected:
                        picked = sorted(picked, key=lambda h: -h.conf)[:expected]
                        break
            elif len(picked) > expected:                               # 과탐: 기준을 높인다
                for c in (0.35, 0.45, 0.55, 0.65):
                    p2 = [h for h in cands if h.conf >= c]
                    log["conf"] = c
                    if len(p2) <= expected:
                        picked = p2 if len(p2) == expected else sorted(picked, key=lambda h: -h.conf)[:expected]
                        break
            if len(picked) < expected:                                 # 뭉친 탄공 나누기
                picked, n_split = split_merged(image, picked, expected)
                log["split"] = n_split
        log["found"] = len(picked)
        return picked, log


def split_merged(image: np.ndarray, holes: list[PxHole], expected: int) -> tuple[list[PxHole], int]:
    """Split unusually large boxes (several overlapping shots) into k holes by k-means on the torn pixels."""
    if not holes:
        return holes, 0
    need = expected - len(holes)
    r_med = float(np.median([h.r for h in holes]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    big = sorted([h for h in holes if h.r > 1.5 * r_med], key=lambda h: -h.r)
    out = [h for h in holes if h not in big]
    n_split = 0
    for h in big:
        k_area = int(round((h.r / r_med) ** 2))
        k = int(np.clip(min(k_area, need + 1), 1, 6))
        if k <= 1:
            out.append(h)
            continue
        x0, y0 = int(max(0, h.x - h.r)), int(max(0, h.y - h.r))
        x1, y1 = int(min(gray.shape[1], h.x + h.r)), int(min(gray.shape[0], h.y + h.r))
        roi = gray[y0:y1, x0:x1]
        if roi.size < 20:
            out.append(h)
            continue
        # 탄공은 주변과 밝기가 다른 픽셀 (검은 바탕의 흰 찢김 / 흰 바탕의 검은 구멍)
        dev = np.abs(roi.astype(np.float32) - np.median(roi))
        mask = dev > max(20.0, np.percentile(dev, 60))
        pts = np.column_stack(np.nonzero(mask)[::-1]).astype(np.float32)
        if len(pts) < k * 5:
            out.append(h)
            continue
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, _, centers = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        for cx, cy in centers:
            out.append(PxHole(x0 + float(cx), y0 + float(cy), r_med, h.conf * 0.9, split=True))
        n_split += k - 1
        need -= k - 1
        if need <= 0:
            break
    return out, n_split


def _on_printed_digit(frame, spec, x: float, y: float) -> bool:
    """점수 숫자와 가운데 + 는 가로·세로 축 위에 인쇄되어 있다 (검은 원 안의 흰 숫자는 탄공처럼 밝다)."""
    tx, ty = frame.to_target_mm(np.array([[x, y]]))[0]
    r = float(np.hypot(tx, ty))
    band = spec.ring_step_mm * 0.28
    if r < spec.ring10_radius_mm * 0.7:                                   # 가운데 +
        return abs(tx) < band or abs(ty) < band
    return abs(tx) < band or abs(ty) < band


def bright_holes_in_disc(image: np.ndarray, frame, spec, existing: list[PxHole], need: int) -> list[PxHole]:
    """검은 조준 원 안의 탄공은 뒤로 빛이 비쳐 '밝은 점'으로 보인다. AI가 발수를 못 채웠을 때 이것으로 보충한다.

    - 탄공 크기보다 큰 창으로 top-hat → 주변보다 밝은 작은 영역만 남김
    - 링 선·가운데 + 는 탄공보다 가늘어서 열기(opening)로 지워짐
    - 이미 찾은 탄공 근처는 제외, 밝기 대비가 큰 순서로 need 개까지
    """
    if need <= 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    r_px = spec.bullet_diameter_mm / 2 / frame.mm_per_px
    if r_px < 2:
        return []
    (cx, cy), (a, b), ang = frame.center_px, frame.axes_px, frame.angle_deg
    disc = np.zeros_like(gray)
    cv2.ellipse(disc, ((cx, cy), (2 * a * 0.97, 2 * b * 0.97), ang), 255, -1)
    k = int(r_px * 5) | 1
    th = cv2.morphologyEx(cv2.GaussianBlur(gray, (3, 3), 0), cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    th[disc == 0] = 0
    vals = th[disc > 0]
    if vals.size == 0:
        return []
    t = max(25.0, float(np.percentile(vals, 97)) * 0.6)
    mask = (th > t).astype(np.uint8) * 255
    ko = max(3, int(r_px * 0.9) | 1)                                   # 링 선·+ 표시(가는 선) 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    area1 = np.pi * r_px ** 2
    cands = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.25 * area1 or area > 8 * area1:
            continue
        x, y = cent[i]
        if _on_printed_digit(frame, spec, x, y):
            continue
        contrast = float(th[lab == i].mean())
        k_holes = int(np.clip(round(area / (2.2 * area1)), 1, 4))          # 흐린 사진은 탄공이 번져 커 보인다
        if k_holes == 1:
            cands.append((contrast, PxHole(float(x), float(y), r_px, 0.2, split=False)))
        else:
            pts = np.column_stack(np.nonzero(lab == i)[::-1]).astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
            _, _, centers = cv2.kmeans(pts, k_holes, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            for px_, py_ in centers:
                cands.append((contrast, PxHole(float(px_), float(py_), r_px, 0.18, split=True)))
    out = []
    for _, h in sorted(cands, key=lambda c: -c[0]):
        if all(np.hypot(h.x - e.x, h.y - e.y) > 1.9 * r_px for e in existing + out):
            out.append(h)
        if len(out) >= need:
            break
    return out
