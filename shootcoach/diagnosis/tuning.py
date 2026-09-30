"""교관 라벨로 자세 신호 임계값 보정.

입력: 영상(또는 관절 .npz)별로 교관이 신호 유무(1/0)를 표시한 표.
각 신호의 특징값(영상 내 격발 평균)을 계산하고, 교관 판정과 가장 잘 맞는 임계값(Youden J 최대)을 추천한다.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shootcoach.diagnosis.stage2 import load_signals
from shootcoach.pose.features import arm_series, shot_features
from shootcoach.pose.keypoints import KeypointSeq
from shootcoach.pose.shots import motion_shot_times


def load_sequence(path: str | Path) -> tuple[KeypointSeq, list[float]]:
    """.npz (xy, conf, fps[, shots]) or a video file. Returns the sequence and shot times."""
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=False)   # never unpickle files from others
        seq = KeypointSeq(float(d["fps"]), d["xy"], d["conf"])
        shots = [float(x) for x in d["shots"]] if "shots" in d.files else []
    else:
        from shootcoach.pose.keypoints import extract_keypoints
        from shootcoach.pose.shots import audio_shot_times

        seq = extract_keypoints(path)
        shots = audio_shot_times(path)
    if not shots:
        _, pitch, _ = arm_series(seq)
        shots = motion_shot_times(pitch, seq.fps)
        cuts = seq.meta.get("scene_cuts", [])
        shots = [t for t in shots if not cuts or min(abs(t - c) for c in cuts) > 0.3]
    return seq, shots


def video_feature_means(seq: KeypointSeq, shots: list[float], handedness: str = "right") -> dict[str, float]:
    feats = [shot_features(seq, t, handedness).values for t in shots]
    keys = feats[0].keys() if feats else []
    return {k: float(np.nanmean([f[k] for f in feats])) for k in keys}


@dataclass
class TuneRow:
    signal_id: str
    feature: str
    compare: str
    current: float
    suggested: float | None
    n_pos: int
    n_neg: int
    acc_current: float | None
    acc_suggested: float | None
    note: str = ""


def _score(x: np.ndarray, compare: str) -> np.ndarray:
    """Map feature values so that 'larger = more present' for every compare mode."""
    return {"gt": x, "lt": -x, "abs_gt": np.abs(x)}[compare]


def best_threshold(values: np.ndarray, labels: np.ndarray, compare: str) -> tuple[float, float]:
    """Threshold (in the feature's own units) maximising Youden's J; returns (threshold, accuracy)."""
    s = _score(values, compare)
    order = np.unique(s)
    cands = (order[:-1] + order[1:]) / 2 if len(order) > 1 else order
    best = (-2.0, float(cands[0]), 0.0)
    pos, neg = labels == 1, labels == 0
    for c in cands:
        pred = s >= c
        tpr = (pred & pos).sum() / max(1, pos.sum())
        fpr = (pred & neg).sum() / max(1, neg.sum())
        j = tpr - fpr
        acc = (pred == pos).mean()
        if j > best[0] or (j == best[0] and acc > best[2]):
            best = (j, float(c), float(acc))
    thr = {"gt": best[1], "lt": -best[1], "abs_gt": best[1]}[compare]
    return thr, best[2]


def tune(rows: list[dict], min_each: int = 2) -> list[TuneRow]:
    """rows: [{"features": {feature: value}, "labels": {signal_id: 0/1}}, ...]"""
    out = []
    for sid, sig in load_signals().items():
        vals, labs = [], []
        for r in rows:
            lab = r["labels"].get(sid)
            v = r["features"].get(sig.feature, np.nan)
            if lab in (0, 1) and np.isfinite(v):
                vals.append(v)
                labs.append(lab)
        vals, labs = np.asarray(vals, float), np.asarray(labs, int)
        n_pos, n_neg = int((labs == 1).sum()), int((labs == 0).sum())
        if n_pos < min_each or n_neg < min_each:
            out.append(TuneRow(sid, sig.feature, sig.compare, sig.threshold, None, n_pos, n_neg, None, None,
                               f"라벨 부족 (양성·음성 각 {min_each}개 이상 필요)"))
            continue
        cur_pred = _score(vals, sig.compare) >= _score(np.array([sig.threshold]), sig.compare)[0]
        acc_cur = float((cur_pred == (labs == 1)).mean())
        thr, acc = best_threshold(vals, labs, sig.compare)
        out.append(TuneRow(sid, sig.feature, sig.compare, sig.threshold, round(thr, 4), n_pos, n_neg,
                           round(acc_cur, 3), round(acc, 3)))
    return out


def write_suggested_csv(results: list[TuneRow], src: Path, dst: Path) -> None:
    with open(src, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sug = {r.signal_id: r.suggested for r in results if r.suggested is not None}
    for r in rows:
        if r["signal_id"] in sug:
            r["threshold"] = f"{sug[r['signal_id']]:g}"
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
