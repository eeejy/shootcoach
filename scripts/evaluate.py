"""성능 평가 (정직한 수치용).

    python scripts/evaluate.py                      # 공개 테스트셋 + 해경 실사진(정답 있으면) + 자세 합성 시나리오
    python scripts/evaluate.py --real data/kcg_real  # 해경 실사진 폴더 (이미지 + labels/*.txt YOLO 형식 정답)

- 공개 테스트셋: data/real_v1 test split (학습에 쓰지 않은 사진)
- 해경 실사진: 사진마다 같은 이름의 YOLO 라벨(.txt). 표적 찾기 성공률, 탄공 정밀도/재현율, 발별 점수 일치,
  발수 보정 전후 비교를 낸다.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shootcoach.diagnosis.stage1 import diagnose_stage1
from shootcoach.diagnosis.stage2 import diagnose_stage2
from shootcoach.pipeline import default_spec, detect_holes_px, get_photo_detector
from shootcoach.pose.simulate import Faults, simulate_side_view
from shootcoach.target.detect import Hole
from shootcoach.target.locate import TargetNotFound, locate_target
from shootcoach.target.scoring import group_stats, score_hole, to_target_xy


def read_yolo(lab: Path, w: int, h: int) -> np.ndarray:
    pts = []
    if lab.exists():
        for l in lab.read_text().split("\n"):
            p = l.split()
            if len(p) == 5:
                pts.append((float(p[1]) * w, float(p[2]) * h, (float(p[3]) * w + float(p[4]) * h) / 4))
    return np.array(pts).reshape(-1, 3)


def match(gt: np.ndarray, pred: np.ndarray, tol_factor: float = 1.0):
    """Greedy one-to-one match within tol_factor × GT radius (min 3 px)."""
    used, pairs = set(), []
    for gi, (gx, gy, gr) in enumerate(gt):
        if not len(pred):
            break
        d = np.hypot(pred[:, 0] - gx, pred[:, 1] - gy)
        for pi in np.argsort(d):
            if pi in used:
                continue
            if d[pi] <= max(3.0, tol_factor * gr * 1.5):
                used.add(pi)
                pairs.append((gi, int(pi)))
            break
    return pairs


def prf(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3), "tp": tp, "fp": fp, "fn": fn}


def eval_public(det, root: Path):
    tp = fp = fn = 0
    imgs = sorted((root / "images" / "test").glob("*"))
    for img_p in imgs:
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]
        gt = read_yolo(root / "labels" / "test" / (img_p.stem + ".txt"), w, h)
        pred, _ = det.detect(im, None)
        P = np.array([[q.x, q.y] for q in pred]).reshape(-1, 2)
        pairs = match(gt, P)
        tp += len(pairs)
        fp += len(P) - len(pairs)
        fn += len(gt) - len(pairs)
    return {"images": len(imgs), **prf(tp, fp, fn)}


def eval_real(det, folder: Path, spec, shots: int | None):
    rows = []
    agg = {"raw": [0, 0, 0], "count": [0, 0, 0]}
    score_ok = score_n = located = 0
    imgs = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for img_p in imgs:
        im = cv2.imread(str(img_p))
        h, w = im.shape[:2]
        gt = read_yolo(folder / "labels" / (img_p.stem + ".txt"), w, h)
        try:
            frame = locate_target(im, spec)
            located += 1
        except TargetNotFound:
            rows.append({"image": img_p.name, "located": False})
            continue
        res = {"image": img_p.name, "located": True, "gt": len(gt)}
        n_exp = shots or len(gt)
        for mode, expected in (("raw", None), ("count", n_exp)):
            pred, log = detect_holes_px(det, im, frame, spec, expected)
            P = np.array([[q.x, q.y] for q in pred]).reshape(-1, 2)
            pairs = match(gt, P)
            a = agg[mode]
            a[0] += len(pairs); a[1] += len(P) - len(pairs); a[2] += len(gt) - len(pairs)
            res[mode] = {"found": len(P), "tp": len(pairs)}
            if mode == "count":
                g_mm = frame.paper_mm(gt[:, :2], spec) if len(gt) else np.zeros((0, 2))
                p_mm = frame.paper_mm(P, spec) if len(P) else np.zeros((0, 2))
                gs = [score_hole(float(np.hypot(*v)), spec) for v in to_target_xy([Hole(x, y, 4.5) for x, y in g_mm], spec)]
                ps = [score_hole(float(np.hypot(*v)), spec) for v in to_target_xy([Hole(x, y, 4.5) for x, y in p_mm], spec)]
                for gi, pi in pairs:
                    score_n += 1
                    score_ok += int(gs[gi] == ps[pi])
                res["total_gt"], res["total_pred"] = int(sum(gs)), int(sum(ps))
        rows.append(res)
    out = {"images": len(imgs), "located": located,
           "detect_raw": prf(*agg["raw"]), "detect_with_count_correction": prf(*agg["count"]),
           "score_exact_acc": round(score_ok / max(1, score_n), 3), "per_image": rows}
    return out


FAULT_CASES = {  # stage-1 group (A4 synthetic, mm), injected posture fault, expected stage-2 verdict
    "low-left + muzzle dip": ((-22, -22), (13, 13), Faults(dip_deg=4), {"L1", "L2", "L3"}),   # 측면 영상으로는 하방 원인 3종 구분 어려움
    "low-left, clean posture": ((-22, -22), (13, 13), Faults(), None),
    "high-right + muzzle up": ((22, 22), (13, 13), Faults(heel_deg=4, shrug=0.02), {"H1", "H2", "H4"}),
    "high + early drop": ((0, 24), (13, 13), Faults(early_drop=0.12), {"H4"}),
    "vertical + breath": ((0, 0), (2.5, 16), Faults(breath_amp=0.03), {"S1"}),
    "vertical, clean": ((0, 0), (2.5, 16), Faults(), None),
}


def eval_stage2(seeds=20):
    from shootcoach.config import load_target_spec

    spec = load_target_spec("configs/target_a4.yaml")
    rows = {}
    for name, (off, sig, faults, expect) in FAULT_CASES.items():
        n = 10
        k = np.arange(n) + 0.5
        a = k * np.pi * (3 - np.sqrt(5))
        u = np.c_[np.sqrt(k / n) * np.cos(a), np.sqrt(k / n) * np.sin(a)]
        u = (u - u.mean(0)) / u.std(0)
        pts = np.asarray(off) + u * np.asarray(sig)
        s1 = diagnose_stage1(group_stats([Hole(spec.center_mm[0] + x, spec.center_mm[1] - y, 4.5) for x, y in pts], spec), spec)
        ok = 0
        for sd in range(seeds):
            f = Faults(**{kk: v * float(np.random.default_rng(sd).uniform(0.8, 1.3)) if kk != "tremor" else v
                          for kk, v in faults.__dict__.items()})
            got = diagnose_stage2(s1, simulate_side_view((2.5, 5.0, 7.5), faults=f, seed=sd)).final_cause_id
            ok += int(got in expect) if expect else int(got is None)
        rows[name] = {"expected": sorted(expect) if expect else "보류", "accuracy": ok / seeds}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default="data/real_v1")
    ap.add_argument("--real", default="data/kcg_real")
    ap.add_argument("--shots", type=int, default=None, help="장당 발수 (없으면 정답 개수를 발수로 사용)")
    ap.add_argument("--out", default="docs/results.json")
    a = ap.parse_args()
    det = get_photo_detector()
    res = {"detector": type(det).__name__}
    if Path(a.public, "images", "test").exists() and hasattr(det, "raw"):
        res["public_test"] = eval_public(det, Path(a.public))
    if Path(a.real).exists() and hasattr(det, "raw"):
        res["kcg_real_photos"] = eval_real(det, Path(a.real), default_spec(), a.shots)
    res["stage2_posture_synthetic"] = eval_stage2()
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "kcg_real_photos"} |
                     ({"kcg_real_photos": {k: v for k, v in res["kcg_real_photos"].items() if k != "per_image"}}
                      if "kcg_real_photos" in res else {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
