"""로컬 라벨링: 실제 표적지 사진 → (정면 보정 + 모델 사전 검출) → 사람이 수정 → YOLO 데이터셋.

해경 사진은 외부 라벨링 서비스(Roboflow 등)에 올리지 않고 이 PC 안에서만 라벨링한다.
라벨은 학습 데이터와 같은 규칙(정면 보정 이미지, 박스 = 탄공 반지름 × 2.2)으로 저장한다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import TargetSpec
from shootcoach.synth import yolo_labels
from shootcoach.target.detect import Hole
from shootcoach.target.markers import rectify

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTS) if folder.exists() else []


def split_for(stem: str, val_frac: float = 0.2) -> str:
    """Deterministic train/val split so re-saving an image never moves it between splits."""
    h = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def prelabel(image: np.ndarray, spec: TargetSpec, detector, already_rectified: bool = False):
    rect = cv2.resize(image, spec.canvas_px) if already_rectified else rectify(image, spec).image
    return rect, detector.detect(rect, spec)


def save_sample(rect: np.ndarray, holes: list[Hole], spec: TargetSpec, out_dir: str | Path, stem: str,
                split: str | None = None) -> Path:
    out = Path(out_dir)
    split = split or split_for(stem)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    ppm = spec.px_per_mm
    h, w = rect.shape[:2]
    pts = np.array([[hh.x_mm, hh.y_mm] for hh in holes]).reshape(-1, 2) * ppm
    radii = np.array([hh.r_mm for hh in holes]) * ppm
    lines = yolo_labels(pts, radii, w, h) if len(holes) else []
    img_path = out / "images" / split / f"{stem}.jpg"
    cv2.imwrite(str(img_path), rect, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    write_data_yaml(out)
    return img_path


def load_sample(out_dir: str | Path, stem: str, spec: TargetSpec) -> list[Hole] | None:
    out = Path(out_dir)
    for split in ("train", "val"):
        lab = out / "labels" / split / f"{stem}.txt"
        img = out / "images" / split / f"{stem}.jpg"
        if lab.exists() and img.exists():
            h, w = cv2.imread(str(img)).shape[:2]
            ppm = spec.px_per_mm
            holes = []
            for line in lab.read_text().split("\n"):
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = map(float, line.split())
                r = (bw * w + bh * h) / 4 / 1.1
                holes.append(Hole(cx * w / ppm, cy * h / ppm, r / ppm))
            return holes
    return None


def write_data_yaml(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    p = out / "data.yaml"
    p.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: bullet_hole\n")
    return p


def dataset_counts(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    res = {}
    for split in ("train", "val"):
        labs = list((out / "labels" / split).glob("*.txt"))
        res[split] = {"images": len(labs), "holes": sum(len([x for x in f.read_text().split("\n") if x.strip()]) for f in labs)}
    return res


def draw_label_view(rect: np.ndarray, holes: list[Hole], spec: TargetSpec) -> np.ndarray:
    out = rect.copy()
    ppm = spec.px_per_mm
    for i, h in enumerate(holes, 1):
        c = (int(h.x_mm * ppm), int(h.y_mm * ppm))
        color = (0, 170, 255) if h.conf < 1.0 else (60, 200, 60)      # 주황 = 모델 제안, 초록 = 사람 추가
        cv2.circle(out, c, int(h.r_mm * ppm * 1.2), color, 3, cv2.LINE_AA)
        cv2.putText(out, str(i), (c[0] + int(h.r_mm * ppm) + 3, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 180), 2, cv2.LINE_AA)
    return out
