"""Hole type in target mm coordinates. Detection itself lives in shootcoach/target/photo.py (photo-trained YOLO)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hole:
    x_mm: float          # paper mm, x → right
    y_mm: float          # paper mm, y → down
    r_mm: float
    conf: float = 1.0

    def as_dict(self):
        return {"x_mm": round(float(self.x_mm), 2), "y_mm": round(float(self.y_mm), 2),
                "r_mm": round(float(self.r_mm), 2), "conf": round(float(self.conf), 3)}
