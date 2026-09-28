"""Scoring, shot-group statistics and group-shape classification (1단계의 입력)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from shootcoach.config import TargetSpec
from shootcoach.target.detect import Hole

CLOCK_LABELS_KO = {
    12: "12시(위)", 1.5: "1시 반(오른쪽 위)", 3: "3시(오른쪽)", 4.5: "4시 반(오른쪽 아래)",
    6: "6시(아래)", 7.5: "7시 반(왼쪽 아래)", 9: "9시(왼쪽)", 10.5: "10시 반(왼쪽 위)",
}
SECTORS = [12, 1.5, 3, 4.5, 6, 7.5, 9, 10.5]


def to_target_xy(holes: list[Hole], spec: TargetSpec) -> np.ndarray:
    """Hole centres relative to target centre in mm, x → right, y → UP (shooter's view)."""
    if not holes:
        return np.zeros((0, 2))
    p = np.array([[h.x_mm, h.y_mm] for h in holes], float)
    return np.c_[p[:, 0] - spec.center_mm[0], spec.center_mm[1] - p[:, 1]]


def score_hole(dist_mm: float, spec: TargetSpec) -> int:
    """Ring value using the outward-gauging rule: any contact with a ring line counts for that ring."""
    d = dist_mm - spec.bullet_diameter_mm / 2
    if d <= spec.ring10_radius_mm:
        return 10
    ring = 10 - math.ceil((d - spec.ring10_radius_mm) / spec.ring_step_mm)
    lowest = 10 - spec.ring_count + 1
    return ring if ring >= lowest else 0


def clock_of(vec: np.ndarray) -> float:
    """Clock position (12 = up, 3 = right) of a vector, continuous 0-12."""
    ang = math.degrees(math.atan2(vec[0], vec[1])) % 360  # 0° = up, clockwise
    c = ang / 30.0
    return 12.0 if c < 0.5 else c


def sector_of(vec: np.ndarray) -> float:
    ang = math.degrees(math.atan2(vec[0], vec[1])) % 360
    return SECTORS[int(((ang + 22.5) % 360) // 45)]


@dataclass
class GroupStats:
    n: int
    n_fliers: int
    scores: list[int]
    total_score: int
    center_mm: tuple[float, float]          # x right, y up
    offset_mm: float
    clock: float
    sector: float
    sector_ko: str
    extreme_spread_mm: float
    mean_radius_mm: float
    std_x_mm: float
    std_y_mm: float
    shape: str
    shape_ko: str

    def as_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 2)
        d["center_mm"] = [round(v, 2) for v in self.center_mm]
        return d


SHAPE_KO = {
    "insufficient": "탄 수 부족 (최소 3발, 권장 5발 이상)",
    "tight_centered": "밀집 · 중앙 (양호)",
    "tight_offset": "밀집 · 한쪽 쏠림 (영점/조준 문제 가능성)",
    "scattered_centered": "산개 · 중앙 (전반적 불안정)",
    "scattered_offset": "산개 · 한쪽 쏠림 (특정 동작 오류 가능성)",
    "vertical_string": "세로로 길게 퍼짐 (호흡·앞뒤 흔들림)",
    "horizontal_string": "가로로 길게 퍼짐 (자세·파지·좌우 흔들림)",
}


@dataclass
class ShapeThresholds:
    """Relative to the target's ring width so they transfer across target sizes."""
    tight_mean_radius_rings: float = 1.6     # mean radius ≤ 1.6 ring widths → tight
    offset_min_rings: float = 1.0            # centre offset ≥ 1 ring width → offset
    offset_vs_spread: float = 0.9            # … and ≥ 0.9 × mean radius
    string_ratio: float = 2.2                # std ratio for stringing
    string_min_rings: float = 1.0            # … and long-axis std ≥ 1 ring width
    flier_factor: float = 2.5                # > 2.5 × median distance-to-centroid → flier
    flier_min_rings: float = 3.0             # … and > 3 ring widths from the group


def group_stats(holes: list[Hole], spec: TargetSpec, th: ShapeThresholds | None = None) -> GroupStats:
    th = th or ShapeThresholds()
    xy = to_target_xy(holes, spec)
    n = len(xy)
    scores = [score_hole(float(np.hypot(*p)), spec) for p in xy]
    if n == 0:
        return GroupStats(0, 0, [], 0, (0.0, 0.0), 0.0, 12.0, 12, CLOCK_LABELS_KO[12], 0.0, 0.0, 0.0, 0.0,
                          "insufficient", SHAPE_KO["insufficient"])
    # Robust centroid: drop fliers (only with ≥5 shots, as coaches judge the group, not single shots).
    keep = np.ones(n, bool)
    if n >= 5:
        # A flier must be far from the group AND rare: at most 1 per 5 shots, farthest first.
        med = np.median(xy, axis=0)
        d = np.linalg.norm(xy - med, axis=1)
        limit = max(th.flier_factor * np.median(d), th.flier_min_rings * spec.ring_step_mm)
        far = [i for i in np.argsort(-d) if d[i] > limit][: n // 5]
        keep[far] = False
    g = xy[keep]
    c = g.mean(axis=0)
    offset = float(np.hypot(*c))
    rad = np.linalg.norm(g - c, axis=1)
    mean_r = float(rad.mean()) if len(g) else 0.0
    es = float(max((np.linalg.norm(a - b) for i, a in enumerate(g) for b in g[i + 1:]), default=0.0))
    sx, sy = (float(v) for v in g.std(axis=0)) if len(g) > 1 else (0.0, 0.0)
    ring_w = spec.ring_step_mm

    if n < 3:
        shape = "insufficient"
    else:
        tight = mean_r <= th.tight_mean_radius_rings * ring_w
        is_offset = offset >= th.offset_min_rings * ring_w and offset >= th.offset_vs_spread * mean_r
        ratio = (sy + 1e-6) / (sx + 1e-6)
        # Stringing is judged before tightness: a narrow but long line is a breathing/sway
        # pattern even when its mean radius is small.
        long_min = th.string_min_rings * ring_w
        if not is_offset and ratio >= th.string_ratio and sy >= long_min:
            shape = "vertical_string"
        elif not is_offset and ratio <= 1 / th.string_ratio and sx >= long_min:
            shape = "horizontal_string"
        elif tight:
            shape = "tight_offset" if is_offset else "tight_centered"
        else:
            shape = "scattered_offset" if is_offset else "scattered_centered"
    sector = sector_of(c) if offset > 1e-6 else 12
    return GroupStats(
        n=n, n_fliers=int((~keep).sum()), scores=scores, total_score=int(sum(scores)),
        center_mm=(float(c[0]), float(c[1])), offset_mm=offset, clock=clock_of(c), sector=sector,
        sector_ko=CLOCK_LABELS_KO[sector], extreme_spread_mm=es, mean_radius_mm=mean_r,
        std_x_mm=sx, std_y_mm=sy, shape=shape, shape_ko=SHAPE_KO[shape],
    )
