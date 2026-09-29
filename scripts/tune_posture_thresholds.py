"""교관 라벨로 자세 신호 임계값 추천.

라벨 CSV (예: data/posture_labels.csv):
    video,handedness,muzzle_dip_preshot,muzzle_up_preshot,early_drop_post,breath_sway
    data/haegyeong/video/kim_01.mp4,right,1,0,0,
    ...
(비워 둔 칸은 판정 안 함. 1 = 교관이 그 신호를 봤음, 0 = 없었음)

    python scripts/tune_posture_thresholds.py data/posture_labels.csv
→ rules/posture_signals.suggested.csv (기존 규칙 파일은 건드리지 않음) + docs/threshold_tuning.md
"""
import argparse
import csv
from pathlib import Path

from shootcoach.diagnosis.rules import RULES_DIR
from shootcoach.diagnosis.tuning import load_sequence, tune, video_feature_means, write_suggested_csv

ap = argparse.ArgumentParser()
ap.add_argument("labels_csv")
ap.add_argument("--out-csv", default=str(RULES_DIR / "posture_signals.suggested.csv"))
ap.add_argument("--report", default="docs/threshold_tuning.md")
a = ap.parse_args()

rows = []
with open(a.labels_csv, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        seq, shots = load_sequence(r["video"])
        if not shots:
            print(f"[건너뜀] {r['video']}: 격발 시점을 찾지 못함")
            continue
        labels = {k: int(v) for k, v in r.items() if k not in ("video", "handedness", "view") and v.strip() in ("0", "1")}
        rows.append({"features": video_feature_means(seq, shots, r.get("handedness") or "right"), "labels": labels})
        print(f"[분석] {r['video']}: 격발 {len(shots)}회")

res = tune(rows)
write_suggested_csv(res, RULES_DIR / "posture_signals.csv", Path(a.out_csv))
lines = ["# 자세 신호 임계값 보정 결과", "", f"영상 {len(rows)}개 · 라벨 파일 `{a.labels_csv}`", "",
         "| 신호 | 현재 임계값 | 추천 | 양성/음성 | 현재 일치율 | 추천 일치율 | 비고 |", "|---|---|---|---|---|---|---|"]
for r in res:
    lines.append(f"| {r.signal_id} | {r.current:g} | {'' if r.suggested is None else f'{r.suggested:g}'} | {r.n_pos}/{r.n_neg} | "
                 f"{'' if r.acc_current is None else f'{r.acc_current:.0%}'} | {'' if r.acc_suggested is None else f'{r.acc_suggested:.0%}'} | {r.note} |")
lines += ["", f"추천값은 `{a.out_csv}`에 저장했다. 검토 후 `rules/posture_signals.csv`에 반영한다.",
          "영상 수가 적으면 추천값이 과적합될 수 있다 — 신호당 양성·음성 각 5개 이상을 권장한다."]
Path(a.report).write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
