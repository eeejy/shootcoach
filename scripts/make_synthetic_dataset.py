"""Generate a YOLO dataset of rectified targets with synthetic bullet holes.

    python scripts/make_synthetic_dataset.py --out data/synth --train 1600 --val 300
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.synth import augment_rectified, make_synth_target, yolo_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--train", type=int, default=1600)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    spec = load_target_spec(args.config)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    ppm = spec.px_per_mm
    for split, n in (("train", args.train), ("val", args.val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            t = make_synth_target(spec, rng)
            img, M = augment_rectified(t.rect, rng)
            h, w = img.shape[:2]
            labels = yolo_labels(t.holes_mm * ppm, t.radii_mm * ppm, w, h, M)
            stem = f"{split}_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n")
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: bullet_hole\n"
    )
    print("done", out.resolve())


if __name__ == "__main__":
    main()
