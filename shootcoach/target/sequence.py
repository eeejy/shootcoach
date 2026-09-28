"""Shot-by-shot capture: find the NEW holes by matching against the previous photo (중첩 탄공 대응)."""
from __future__ import annotations

import numpy as np

from shootcoach.target.detect import Hole


def new_holes(prev: list[Hole], curr: list[Hole], match_mm: float) -> list[Hole]:
    """Holes in `curr` not within `match_mm` of any hole in `prev` (greedy one-to-one)."""
    if not prev:
        return list(curr)
    used = set()
    out = []
    P = np.array([[h.x_mm, h.y_mm] for h in prev])
    for h in sorted(curr, key=lambda h: -h.conf):
        d = np.hypot(P[:, 0] - h.x_mm, P[:, 1] - h.y_mm)
        order = [i for i in np.argsort(d) if i not in used]
        if order and d[order[0]] <= match_mm:
            used.add(order[0])
        else:
            out.append(h)
    return out
