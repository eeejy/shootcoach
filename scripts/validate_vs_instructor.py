"""교관 판정 대비 검증 (3일차 '검증 10건').

입력 CSV (예: data/validation.csv):
    target_image,video,handedness,instructor_cause,note
    data/haegyeong/raw/kim_01.jpg,data/haegyeong/video/kim_01.mp4,right,jerking,
    data/haegyeong/raw/lee_02.jpg,,right,sight_zero,영상 없음
instructor_cause 는 rules/causes.csv 의 cause_id (예: jerking, heeling, sight_zero, breathing ...)

    python scripts/validate_vs_instructor.py data/validation.csv
→ docs/validation_report.md, docs/validation_rows.csv
목표에 못 미쳐도 그대로 기록한다 (발표에서 수치를 부풀리지 않는다).
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from shootcoach.diagnosis.rules import load_causes
from shootcoach.pipeline import analyze_posture, analyze_target
from shootcoach.target.markers import MarkerError

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--report", default="docs/validation_report.md")
ap.add_argument("--rows-out", default="docs/validation_rows.csv")
a = ap.parse_args()
causes = load_causes()
out_rows = []
with open(a.csv, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        truth = r["instructor_cause"].strip()
        if truth not in causes:
            raise SystemExit(f"알 수 없는 원인 ID '{truth}' — rules/causes.csv 의 cause_id를 쓰세요")
        hand = (r.get("handedness") or "right").strip()
        row = {"target_image": r["target_image"], "instructor": truth, "top1": "", "top3": "", "final": "",
               "stage2": "", "hit_top1": 0, "hit_top3": 0, "hit_final": "", "error": ""}
        try:
            rep = analyze_target(r["target_image"], handedness=hand).report
        except (MarkerError, FileNotFoundError) as e:
            row["error"] = str(e)
            out_rows.append(row)
            continue
        cands = [c["cause_id"] for c in rep["stage1"]["candidates"]]
        row.update(top1=cands[0] if cands else "", top3=";".join(cands[:3]),
                   hit_top1=int(bool(cands) and cands[0] == truth), hit_top3=int(truth in cands[:3]))
        final = cands[0] if cands else ""
        if (r.get("video") or "").strip():
            rep = analyze_posture(rep, r["video"].strip(), hand)
            s2 = rep["stage2"]
            row["stage2"] = s2["final_ko"]
            final = s2["final_cause_id"] or ""          # 보류 = 빈 값
            row["hit_final"] = int(final == truth)
        row["final"] = final
        out_rows.append(row)

Path(a.rows_out).parent.mkdir(parents=True, exist_ok=True)
with open(a.rows_out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

ok = [r for r in out_rows if not r["error"]]
vid = [r for r in ok if r["hit_final"] != ""]
n = len(ok)
pct = lambda k, rows: f"{sum(int(r[k]) for r in rows) / len(rows):.0%} ({sum(int(r[k]) for r in rows)}/{len(rows)})" if rows else "—"
held = sum(1 for r in vid if not r["final"])
lines = ["# 교관 판정 대비 검증", "", f"표적지 {len(out_rows)}건 (분석 실패 {len(out_rows) - n}건) · 자세 영상 포함 {len(vid)}건", "",
         "| 지표 | 결과 |", "|---|---|",
         f"| 1단계 1순위 = 교관 판정 | {pct('hit_top1', ok)} |",
         f"| 1단계 상위 3개 안에 교관 판정 | {pct('hit_top3', ok)} |",
         f"| 2단계 최종 = 교관 판정 (영상 있는 건) | {pct('hit_final', vid)} |",
         f"| 2단계 보류 | {held}/{len(vid)} |" if vid else "| 2단계 보류 | — |", "",
         "## 건별", "", "| 표적지 | 교관 | 1단계 1순위 | 상위 3 | 2단계 | 판정 |", "|---|---|---|---|---|---|"]
for r in out_rows:
    if r["error"]:
        lines.append(f"| {Path(r['target_image']).name} | {r['instructor']} | 오류: {r['error']} | | | ❌ |")
        continue
    mark = "✅" if (r["hit_final"] == 1 or (r["hit_final"] == "" and r["hit_top1"])) else ("🟡" if r["hit_top3"] else "❌")
    lines.append(f"| {Path(r['target_image']).name} | {causes[r['instructor']].cause_ko} | "
                 f"{causes[r['top1']].cause_ko if r['top1'] else '—'} | {r['top3']} | {r['stage2'] or '—'} | {mark} |")
miss = Counter((r["instructor"], r["top1"]) for r in ok if not r["hit_top1"])
if miss:
    lines += ["", "## 자주 틀린 조합 (교관 → 시스템 1순위)", ""] + [f"- {causes[t].cause_ko} → {causes[p].cause_ko if p else '없음'}: {c}건" for (t, p), c in miss.most_common(5)]
lines += ["", "🟡 = 1순위는 아니지만 상위 3개 안에 있음. 목표에 못 미쳐도 그대로 기록한다."]
Path(a.report).write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
