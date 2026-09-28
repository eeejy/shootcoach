"""CLI: 표적지 사진(+자세 영상) 분석.

    python scripts/analyze.py samples/demo_photo_jerking.jpg --out out/ [--video shot.mp4] [--vlm]
"""
import argparse
from pathlib import Path

import cv2

from shootcoach.explain.vlm import template_explanation, vlm_explanation
from shootcoach.pipeline import analyze_posture, analyze_target, to_json

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--video")
ap.add_argument("--hand", default="right", choices=["right", "left"])
ap.add_argument("--distance", type=float)
ap.add_argument("--click-mm-per-10m", type=float)
ap.add_argument("--vlm", action="store_true", help="로컬 Ollama VLM으로 설명 문장 생성")
ap.add_argument("--out", default="out")
a = ap.parse_args()

res = analyze_target(a.image, handedness=a.hand, distance_m=a.distance, click_mm_per_10m=a.click_mm_per_10m)
rep = res.report
if a.video:
    rep = analyze_posture(rep, a.video, a.hand)
rep["explanation"] = (vlm_explanation(rep, [res.overlay]) if a.vlm
                      else {"text": template_explanation(rep), "source": "template"})
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
stem = Path(a.image).stem
cv2.imwrite(str(out / f"{stem}_overlay.jpg"), res.overlay)
(out / f"{stem}_report.json").write_text(to_json(rep))
g, s1 = rep["group"], rep["stage1"]
print(f"[채점] {g['n']}발 · {g['total_score']}점 · 모양: {s1['shape_ko']} · 방향: {s1['sector_ko']}")
for c in s1["candidates"]:
    print(f"  - 후보 {c['cause_ko']} ({c['score']:.2f})")
if "stage2" in rep:
    print(f"[자세] {rep['stage2']['final_ko']}")
print(f"[설명:{rep['explanation']['source']}] {rep['explanation']['text']}")
print(f"[시간] {rep['timings']}")
