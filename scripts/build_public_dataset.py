"""공개 탄공 데이터셋(Roboflow, CC BY 4.0 / Public Domain)을 '탄공' 한 클래스로 합친다.

    python scripts/build_public_dataset.py --out data/real_v1
원본: data/public/<name>/{train,valid,test}/{images,labels}  (scripts/download_public_datasets.py 로 받음)
점수별 클래스(0~10점)는 모두 bullet_hole 로 합치고, 표적·검은 원 박스는 버린다.
"""
import argparse
import os
from pathlib import Path

import yaml

HOLE_NAMES = {str(i) for i in range(11)} | {"bullet_hole", "bulletholes", "Bullet hole", "bullet hole", "Bullet Hole"}
SOURCES = ["butt_bullet_holes", "knsa", "carbine", "kanat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/public")
    ap.add_argument("--out", default="data/real_v1")
    a = ap.parse_args()
    out = Path(a.out)
    counts = {}
    for name in SOURCES:
        d = Path(a.src) / name
        if not (d / "data.yaml").exists():
            print("skip (없음):", name)
            continue
        names = yaml.safe_load(open(d / "data.yaml"))["names"]
        hole_ids = {i for i, n in enumerate(names) if str(n) in HOLE_NAMES}
        for split_in, split_out in (("train", "train"), ("valid", "val"), ("test", "test")):
            (out / "images" / split_out).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split_out).mkdir(parents=True, exist_ok=True)
            for img in sorted((d / split_in / "images").glob("*")):
                lab = d / split_in / "labels" / (img.stem + ".txt")
                lines = []
                if lab.exists():
                    for l in lab.read_text().split("\n"):
                        p = l.split()
                        if len(p) == 5 and int(p[0]) in hole_ids:
                            lines.append("0 " + " ".join(p[1:]))
                        elif len(p) > 5 and int(p[0]) in hole_ids:           # polygon → bbox
                            xs, ys = list(map(float, p[1::2])), list(map(float, p[2::2]))
                            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                            lines.append(f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
                stem = f"{name}__{img.stem}"
                dst = out / "images" / split_out / (stem + img.suffix)
                if not dst.exists():
                    os.symlink(img.resolve(), dst)
                (out / "labels" / split_out / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
                c = counts.setdefault((name, split_out), [0, 0])
                c[0] += 1
                c[1] += len(lines)
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: bullet_hole\n")
    for (n, s), (i, h) in sorted(counts.items()):
        print(f"{n:18s} {s:5s} images={i:5d} holes={h}")


if __name__ == "__main__":
    main()
