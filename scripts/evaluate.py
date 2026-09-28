"""Evaluate the MVP on held-out synthetic data (photos with perspective + lighting, never seen in training).

    python scripts/evaluate.py --n 200 --out docs/results.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.diagnosis.stage1 import diagnose_stage1
from shootcoach.diagnosis.stage2 import diagnose_stage2
from shootcoach.pose.simulate import Faults, simulate_side_view
from shootcoach.synth import ShotGroupSpec, draw_holes, sample_group, simulate_photo
from shootcoach.target.detect import ClassicHoleDetector, Hole, YoloHoleDetector
from shootcoach.target.markers import MarkerError, rectify
from shootcoach.target.scoring import group_stats, score_hole, to_target_xy
from shootcoach.target.template import render_target


def match(gt, pred, tol):
    used, pairs = set(), []
    for gi, g in enumerate(gt):
        if not len(pred):
            break
        d = np.hypot(*(pred - g).T)
        for pi in np.argsort(d):
            if pi in used:
                continue
            if d[pi] <= tol:
                used.add(pi)
                pairs.append((gi, pi, d[pi]))
            break
    return pairs


def random_group(rng):
    kind = rng.choice(["tight", "offset", "scatter", "vstring", "hstring"])
    n = int(rng.integers(5, 11))
    off = tuple(rng.normal(0, 18, 2)) if kind in ("offset", "scatter") else tuple(rng.normal(0, 2, 2))
    sig = {"tight": (4, 4), "offset": (4, 4), "scatter": (14, 14), "vstring": (3.5, 18), "hstring": (18, 3.5)}[kind]
    sig = tuple(np.asarray(sig) * rng.uniform(0.85, 1.15, 2))
    return ShotGroupSpec(n, off, sig)


def eval_target(spec, n, seed):
    rng = np.random.default_rng(seed)
    dets = {"yolo": YoloHoleDetector(), "classic": ClassicHoleDetector()}
    agg = {k: {"tp": 0, "fp": 0, "fn": 0, "err": [], "score_ok": 0, "score_n": 0, "shape_ok": 0, "t": []} for k in dets}
    rect_fail, rect_t, reproj = 0, [], []
    tol = spec.bullet_diameter_mm / 2
    for i in range(n):
        g = random_group(rng)
        holes = sample_group(spec, g, rng)
        radii = np.full(len(holes), spec.bullet_diameter_mm / 2) * rng.uniform(0.9, 1.05, len(holes))
        rect = draw_holes(render_target(spec), holes * spec.px_per_mm, radii * spec.px_per_mm, rng)
        photo, _ = simulate_photo(rect, rng, max_tilt=float(rng.uniform(0.05, 0.22)))
        t0 = time.perf_counter()
        try:
            r = rectify(photo, spec)
        except MarkerError:
            rect_fail += 1
            continue
        rect_t.append(time.perf_counter() - t0)
        reproj.append(r.reprojection_mm)
        gt_holes = [Hole(x, y, spec.bullet_diameter_mm / 2) for x, y in holes]
        gt_shape = group_stats(gt_holes, spec).shape
        gt_scores = [score_hole(float(np.hypot(*p)), spec) for p in to_target_xy(gt_holes, spec)]
        for name, det in dets.items():
            t1 = time.perf_counter()
            pred = det.detect(r.image, spec)
            agg[name]["t"].append(time.perf_counter() - t1)
            P = np.array([[h.x_mm, h.y_mm] for h in pred]).reshape(-1, 2)
            pairs = match(holes, P, tol)
            a = agg[name]
            a["tp"] += len(pairs)
            a["fp"] += len(P) - len(pairs)
            a["fn"] += len(holes) - len(pairs)
            a["err"] += [d for _, _, d in pairs]
            pred_scores = [score_hole(float(np.hypot(*p)), spec) for p in to_target_xy(pred, spec)]
            for gi, pi, _ in pairs:
                a["score_n"] += 1
                a["score_ok"] += int(gt_scores[gi] == pred_scores[pi])
            a["shape_ok"] += int(group_stats(pred, spec).shape == gt_shape)
    n_ok = n - rect_fail
    out = {"n_photos": n, "rectify_fail": rect_fail, "rectify_ms_mean": round(1000 * float(np.mean(rect_t)), 1),
           "reprojection_mm_mean": round(float(np.mean(reproj)), 3), "detectors": {}}
    for name, a in agg.items():
        p = a["tp"] / max(1, a["tp"] + a["fp"])
        rc = a["tp"] / max(1, a["tp"] + a["fn"])
        out["detectors"][name] = {
            "precision": round(p, 3), "recall": round(rc, 3), "f1": round(2 * p * rc / max(1e-9, p + rc), 3),
            "loc_err_mm_mean": round(float(np.mean(a["err"])), 2) if a["err"] else None,
            "score_exact_acc": round(a["score_ok"] / max(1, a["score_n"]), 3),
            "shape_acc": round(a["shape_ok"] / max(1, n_ok), 3),
            "detect_ms_mean": round(1000 * float(np.mean(a["t"])), 1),
        }
    return out


FAULT_CASES = {
    # stage-1 group (offset, sigma), injected posture fault, expected stage-2 verdict
    "jerking+dip": ((-22, -22), (13, 13), Faults(dip_deg=4), "jerking"),
    "low-left, clean posture": ((-22, -22), (13, 13), Faults(), None),
    "heeling+up": ((22, 22), (13, 13), Faults(heel_deg=4, shrug=0.02), {"heeling", "anticipation_high"}),
    "low+early drop": ((0, -24), (13, 13), Faults(early_drop=0.12), "follow_through"),
    "low+dip": ((0, -24), (13, 13), Faults(dip_deg=4), "anticipation_low"),
    "vertical+breath": ((0, 0), (4, 22), Faults(breath_amp=0.03), "breathing"),
    "vertical, clean": ((0, 0), (4, 22), Faults(), None),
}


def sunflower(spec, offset, sigma, n=10):
    k = np.arange(n) + 0.5
    a = k * np.pi * (3 - np.sqrt(5))
    u = np.c_[np.sqrt(k / n) * np.cos(a), np.sqrt(k / n) * np.sin(a)]
    u = (u - u.mean(0)) / u.std(0)
    pts = np.asarray(offset) + u * np.asarray(sigma)
    return [Hole(spec.center_mm[0] + x, spec.center_mm[1] - y, 4.5) for x, y in pts]


def eval_stage2(spec, seeds=20):
    rows = {}
    for name, (off, sig, faults, expect) in FAULT_CASES.items():
        s1 = diagnose_stage1(group_stats(sunflower(spec, off, sig), spec), spec)
        ok = 0
        for sd in range(seeds):
            f = Faults(**{k: v * float(np.random.default_rng(sd).uniform(0.8, 1.3)) if k != "tremor" else v
                          for k, v in faults.__dict__.items()})
            seq = simulate_side_view((2.5, 5.0, 7.5), faults=f, seed=sd)
            s2 = diagnose_stage2(s1, seq)       # shot times from recoil motion (no audio)
            got = s2.final_cause_id
            ok += int(got in expect) if isinstance(expect, set) else int(got == expect)
        rows[name] = {"stage1_top": s1.candidates[0].cause_id if s1.candidates else None,
                      "expected": sorted(expect) if isinstance(expect, set) else expect, "accuracy": ok / seeds}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="docs/results.json")
    a = ap.parse_args()
    spec = load_target_spec()
    res = {"stage1_target": eval_target(spec, a.n, a.seed), "stage2_posture_synthetic": eval_stage2(spec)}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
