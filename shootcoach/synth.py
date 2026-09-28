"""Synthetic shot targets: realistic bullet holes, shot-group patterns, and phone-photo simulation.

Used to (1) train the hole detector before any real range data exists and
(2) test the full pipeline end-to-end against known ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from shootcoach.config import TargetSpec
from shootcoach.target.template import render_target


@dataclass
class ShotGroupSpec:
    """Where and how tightly the shooter grouped, in mm relative to the target centre (x→right, y→up)."""
    n_shots: int = 5
    offset_mm: tuple[float, float] = (0.0, 0.0)
    sigma_mm: tuple[float, float] = (6.0, 6.0)   # per-axis spread (stringing when unequal)
    fliers: int = 0                              # uncalled wild shots


@dataclass
class SynthTarget:
    rect: np.ndarray                   # rectified image with holes
    holes_mm: np.ndarray               # (N,2) hole centres, paper mm (x right, y down)
    radii_mm: np.ndarray               # (N,)
    group: ShotGroupSpec | None = None
    meta: dict = field(default_factory=dict)


def sample_group(spec: TargetSpec, g: ShotGroupSpec, rng: np.random.Generator) -> np.ndarray:
    """Hole centres in paper mm (y down) for a shot-group pattern."""
    cx, cy = spec.center_mm
    pts = rng.normal([g.offset_mm[0], g.offset_mm[1]], g.sigma_mm, size=(g.n_shots, 2))
    if g.fliers:
        ang = rng.uniform(0, 2 * np.pi, g.fliers)
        rad = rng.uniform(25, spec.outer_radius_mm * 0.95, g.fliers)
        pts = np.vstack([pts, np.c_[rad * np.cos(ang), rad * np.sin(ang)]])
    xy = np.c_[cx + pts[:, 0], cy - pts[:, 1]]
    w, h = spec.paper_mm
    return np.clip(xy, [8, 8], [w - 8, h - 8])


def _hole_polygon(center, radius, rng, n=28):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = radius * (1 + rng.normal(0, 0.06, n))
    rr = cv2.GaussianBlur(rr.reshape(-1, 1).astype(np.float32), (1, 5), 0).ravel()
    return np.c_[center[0] + rr * np.cos(ang), center[1] + rr * np.sin(ang)].astype(np.int32)


def draw_holes(img: np.ndarray, holes_px: np.ndarray, radii_px: np.ndarray, rng: np.random.Generator,
               backer: str | None = None) -> np.ndarray:
    """Paint torn bullet holes. Inside the hole shows the backer (cardboard/dark/white)."""
    out = img.copy()
    backer = backer or rng.choice(["cardboard", "dark", "white", "cardboard"])
    fill = {"cardboard": (95, 135, 170), "dark": (40, 40, 45), "white": (225, 225, 225)}[backer]
    for (x, y), r in zip(holes_px, radii_px):
        poly = _hole_polygon((x, y), r, rng)
        local = img[int(max(0, y - 2)):int(y + 3), int(max(0, x - 2)):int(x + 3)]
        on_black = local.size and local.mean() < 90
        # Bullet wipe / torn fibres: grey ring on white paper, light fuzzy ring on black.
        rim = (70, 70, 70) if not on_black else (150, 150, 150)
        cv2.polylines(out, [_hole_polygon((x, y), r * 1.08, rng)], True, rim,
                      max(1, int(r * 0.18)), cv2.LINE_AA)
        f = np.array(fill, float) * rng.uniform(0.85, 1.1)
        cv2.fillPoly(out, [poly], tuple(int(v) for v in np.clip(f, 0, 255)), cv2.LINE_AA)
        # small paper flaps
        for _ in range(rng.integers(0, 2)):
            a = rng.uniform(0, 2 * np.pi)
            tip = (int(x + r * 0.72 * np.cos(a)), int(y + r * 0.72 * np.sin(a)))
            b1 = (int(x + r * np.cos(a - 0.35)), int(y + r * np.sin(a - 0.35)))
            b2 = (int(x + r * np.cos(a + 0.35)), int(y + r * np.sin(a + 0.35)))
            cv2.fillPoly(out, [np.array([tip, b1, b2])], (225, 225, 225) if not on_black else (35, 35, 35), cv2.LINE_AA)
    return out


def make_synth_target(spec: TargetSpec, rng: np.random.Generator, group: ShotGroupSpec | None = None,
                      overlap_prob: float = 0.15) -> SynthTarget:
    if group is None:
        n = int(rng.integers(3, 16))
        group = ShotGroupSpec(
            n_shots=n,
            offset_mm=tuple(rng.normal(0, 22, 2)),
            sigma_mm=tuple(rng.uniform(3, 22, 2)),
            fliers=int(rng.random() < 0.2),
        )
    holes = sample_group(spec, group, rng)
    # Deliberate near-overlaps (the hard case for detectors).
    extra = []
    for p in holes:
        if rng.random() < overlap_prob:
            a = rng.uniform(0, 2 * np.pi)
            d = spec.bullet_diameter_mm * rng.uniform(0.35, 0.95)
            extra.append(p + d * np.array([np.cos(a), np.sin(a)]))
    if extra:
        holes = np.vstack([holes, extra])
    radii = spec.bullet_diameter_mm / 2 * rng.uniform(0.85, 1.1, len(holes))
    ppm = spec.px_per_mm
    base = render_target(spec)
    rect = draw_holes(base, holes * ppm, radii * ppm, rng)
    return SynthTarget(rect, holes, radii, group)


def augment_rectified(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Mimic imperfect rectification + camera effects. Returns image and 2x3 affine used."""
    h, w = img.shape[:2]
    ang = rng.uniform(-2.5, 2.5)
    sc = rng.uniform(0.97, 1.03)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += rng.uniform(-12, 12, 2)
    out = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    out = _photometric(out, rng)
    return out, M


def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    # paper tint + contrast/brightness
    out = out * rng.uniform(0.75, 1.1) + rng.uniform(-25, 20)
    out *= np.array(rng.uniform(0.9, 1.05, 3), np.float32)
    # lighting gradient / soft shadow
    gx, gy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    a, b = rng.uniform(-1, 1, 2)
    grad = 1 + rng.uniform(0.0, 0.35) * (a * gx + b * gy)
    out *= grad[..., None].astype(np.float32)
    if rng.random() < 0.3:  # hard shadow band (hand/phone)
        x0 = rng.integers(0, w)
        band = (np.abs(gx * w / 2 + w / 2 - x0) < rng.integers(40, 200)).astype(np.float32)
        band = cv2.GaussianBlur(band, (0, 0), 25)
        out *= (1 - 0.35 * band)[..., None]
    out += rng.normal(0, rng.uniform(1, 7), out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    k = rng.choice([0, 0, 3, 5])
    if k:
        out = cv2.GaussianBlur(out, (k, k), 0)
    q = int(rng.integers(55, 95))
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def simulate_photo(rect: np.ndarray, rng: np.random.Generator, out_size=(1600, 2000),
                   max_tilt: float = 0.22) -> tuple[np.ndarray, np.ndarray]:
    """Place the rectified target in a random scene under a random perspective.

    Returns the photo and H (rectified px → photo px) for ground truth.
    """
    h, w = rect.shape[:2]
    W, Hh = out_size
    bg_color = rng.integers(40, 200, 3)
    bg = np.full((Hh, W, 3), bg_color, np.uint8)
    noise = rng.normal(0, 18, (Hh // 8, W // 8, 3))
    bg = np.clip(bg + cv2.resize(noise, (W, Hh)), 0, 255).astype(np.uint8)
    scale = rng.uniform(0.62, 0.85) * min(W / w, Hh / h)
    tw, th = w * scale, h * scale
    ox, oy = (W - tw) / 2 + rng.uniform(-0.08, 0.08) * W, (Hh - th) / 2 + rng.uniform(-0.06, 0.06) * Hh
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[ox, oy], [ox + tw, oy], [ox + tw, oy + th], [ox, oy + th]])
    dst += rng.uniform(-max_tilt, max_tilt, (4, 2)).astype(np.float32) * np.float32([tw, th]) * 0.5
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(rect, Hm, (W, Hh), borderMode=cv2.BORDER_TRANSPARENT, dst=bg.copy())
    return _photometric(warped, rng), Hm


def yolo_labels(holes_px: np.ndarray, radii_px: np.ndarray, w: int, h: int, M: np.ndarray | None = None) -> list[str]:
    lines = []
    for (x, y), r in zip(holes_px, radii_px):
        if M is not None:
            x, y = M @ np.array([x, y, 1.0])
            r = r * np.sqrt(abs(np.linalg.det(M[:, :2])))
        bw = bh = 2.2 * r
        if not (0 <= x < w and 0 <= y < h):
            continue
        lines.append(f"0 {x / w:.6f} {y / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def render_lineart_target(spec: TargetSpec, rng: np.random.Generator) -> np.ndarray:
    """Silhouette/qualification-style paper: random line art + markers (no rings).

    Teaches the detector that line ends, junctions and printed strokes are NOT holes,
    which was the main false-positive mode on real range photos.
    """
    from shootcoach.target.template import render_target

    base = render_target(spec, with_markers=True)
    w, h = spec.canvas_px
    ppm = spec.px_per_mm
    img = base.copy()
    # wipe the rings, keep the markers
    keep = np.zeros((h, w), bool)
    for mid, (mx, my, s) in spec.markers.items():
        x0, y0 = int((mx - s / 2 - 2) * ppm), int((my - s / 2 - 2) * ppm)
        x1, y1 = int((mx + s / 2 + 8) * ppm), int((my + s / 2 + 8) * ppm)
        keep[max(0, y0):y1, max(0, x0):x1] = True
    tint = rng.choice([255, 245, 235, 225])
    img[~keep] = tint
    if rng.random() < 0.35:  # grey silhouette body
        cx, cy = w / 2 + rng.normal(0, 30), h * rng.uniform(0.45, 0.6)
        cv2.ellipse(img, (int(cx), int(cy - 0.28 * h)), (int(0.13 * w), int(0.09 * h)), 0, 0, 360, (200, 200, 200), -1)
        cv2.ellipse(img, (int(cx), int(cy + 0.1 * h)), (int(0.36 * w), int(0.3 * h)), 0, 180, 360, (200, 200, 200), -1)
    for _ in range(int(rng.integers(6, 22))):
        kind = rng.choice(["poly", "curve", "ellipse", "line"])
        th = int(rng.integers(2, 14))
        col = int(rng.choice([0, 20, 40, 180]))
        col = (col, col, col)
        if kind == "ellipse":
            c = (int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.15, 0.85) * h))
            ax = (int(rng.uniform(20, 0.35 * w)), int(rng.uniform(20, 0.3 * h)))
            a0 = rng.uniform(0, 360)
            cv2.ellipse(img, c, ax, rng.uniform(0, 180), a0, a0 + rng.uniform(60, 360), col, th, cv2.LINE_AA)
        else:
            n = 2 if kind == "line" else int(rng.integers(3, 7))
            pts = np.c_[rng.uniform(0.08, 0.92, n) * w, rng.uniform(0.12, 0.88, n) * h].astype(np.int32)
            if kind == "curve":
                t = np.linspace(0, 1, 40)[:, None]
                p0, p1, p2 = pts[0], pts[1], pts[-1]
                pts = ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2).astype(np.int32)
            cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA)
    if rng.random() < 0.3:  # printed numerals / text blocks
        for _ in range(int(rng.integers(1, 5))):
            org = (int(rng.uniform(0.1, 0.8) * w), int(rng.uniform(0.15, 0.9) * h))
            cv2.putText(img, str(int(rng.integers(1, 100))), org, cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.8, 2.5),
                        (0, 0, 0), int(rng.integers(2, 5)), cv2.LINE_AA)
    if rng.random() < 0.3:  # cardboard backer visible in a band
        x0 = int(rng.choice([0, w * 0.75]))
        img[:, x0:x0 + int(w * 0.25)] = (60 + rng.integers(0, 30), 110 + rng.integers(0, 30), 150 + rng.integers(0, 30))
        img[keep] = base[keep]
    return img


def make_lineart_synth(spec: TargetSpec, rng: np.random.Generator) -> SynthTarget:
    w, h = spec.paper_mm
    n = int(rng.integers(0, 18))
    holes = np.c_[rng.uniform(15, w - 15, n), rng.uniform(30, h - 30, n)] if n else np.zeros((0, 2))
    radii = spec.bullet_diameter_mm / 2 * rng.uniform(0.7, 1.1, n)
    ppm = spec.px_per_mm
    base = render_lineart_target(spec, rng)
    backer = rng.choice(["dark", "dark", "cardboard", "white"])
    rect = draw_holes(base, holes * ppm, radii * ppm, rng, backer=backer) if n else base
    return SynthTarget(rect, holes, radii, None, {"style": "lineart"})
