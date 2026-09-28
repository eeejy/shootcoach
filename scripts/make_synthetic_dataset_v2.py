"""v2 dataset: add silhouette/line-art targets (hard negatives for line ends) to the ring-target set.

    python scripts/make_synthetic_dataset_v2.py --base data/synth --out data/synth_v2 --lineart 1400 --val 250
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.synth import augment_rectified, make_lineart_synth, yolo_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/synth")
    ap.add_argument("--out", default="data/synth_v2")
    ap.add_argument("--lineart", type=int, default=1400)
    ap.add_argument("--val", type=int, default=250)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    spec = load_target_spec()
    rng = np.random.default_rng(a.seed)
    base, out = Path(a.base).resolve(), Path(a.out)
    ppm = spec.px_per_mm
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for f in (base / "images" / split).glob("*.jpg"):          # reuse v1 ring targets via symlinks
            for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
                dst = out / sub / split / (f.stem + ext)
                if not dst.exists():
                    os.symlink(base / sub / split / (f.stem + ext), dst)
        n = a.lineart if split == "train" else a.val
        for i in range(n):
            t = make_lineart_synth(spec, rng)
            img, M = augment_rectified(t.rect, rng)
            hh, ww = img.shape[:2]
            labels = yolo_labels(t.holes_mm * ppm, t.radii_mm * ppm, ww, hh, M) if len(t.holes_mm) else []
            stem = f"la_{split}_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: bullet_hole\n")
    print("done", out.resolve())


if __name__ == "__main__":
    main()
