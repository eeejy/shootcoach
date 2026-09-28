"""Target geometry configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_CONFIG = REPO_ROOT / "configs" / "target_a4.yaml"


@dataclass
class TargetSpec:
    name: str
    paper_mm: tuple[float, float]
    center_mm: tuple[float, float]
    ring_count: int
    ring10_radius_mm: float
    ring_step_mm: float
    black_from_ring: int
    bullet_diameter_mm: float
    px_per_mm: float
    aruco_dict: str
    markers: dict[int, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def canvas_px(self) -> tuple[int, int]:
        """(width, height) of the rectified image in pixels."""
        w, h = self.paper_mm
        return int(round(w * self.px_per_mm)), int(round(h * self.px_per_mm))

    def ring_radius_mm(self, ring: int) -> float:
        """Outer radius of the scoring ring `ring` (10 = centre)."""
        return self.ring10_radius_mm + (10 - ring) * self.ring_step_mm

    @property
    def outer_radius_mm(self) -> float:
        return self.ring_radius_mm(10 - self.ring_count + 1)

    def marker_corners_mm(self, marker_id: int) -> list[tuple[float, float]]:
        """Corners in ArUco order (TL, TR, BR, BL) for a marker, in paper mm."""
        cx, cy, s = self.markers[marker_id]
        h = s / 2
        return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


def load_target_spec(path: str | Path | None = None) -> TargetSpec:
    path = Path(path) if path else DEFAULT_TARGET_CONFIG
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TargetSpec(
        name=raw["name"],
        paper_mm=tuple(raw["paper_mm"]),
        center_mm=tuple(raw["center_mm"]),
        ring_count=int(raw["ring_count"]),
        ring10_radius_mm=float(raw["ring10_radius_mm"]),
        ring_step_mm=float(raw["ring_step_mm"]),
        black_from_ring=int(raw["black_from_ring"]),
        bullet_diameter_mm=float(raw["bullet_diameter_mm"]),
        px_per_mm=float(raw["px_per_mm"]),
        aruco_dict=raw["aruco_dict"],
        markers={int(k): tuple(float(x) for x in v) for k, v in raw["markers"].items()},
    )
