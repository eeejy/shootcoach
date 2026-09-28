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
ap.add_argument("--prev", help="같은 표적지의 이전 사진 (회차별 촬영: 새 탄공만 표시)")
ap.add_argument("--profile", help="사수 캘리브레이션 프로필 JSON (profiles/<이름>.json)")
ap.add_argument("--calibrate", metavar="CAUSE_ID", help="진단 세션: 이 사진을 CAUSE_ID 오류를 일부러 낸 결과로 프로필에 기록")
ap.add_argument("--out", default="out")
a = ap.parse_args()

res = analyze_target(a.image, handedness=a.hand, distance_m=a.distance, click_mm_per_10m=a.click_mm_per_10m)
rep = res.report
if a.profile:
    from pathlib import Path as _P

    from shootcoach.config import load_target_spec
    from shootcoach.diagnosis.calibration import ShooterProfile, apply_profile
    from shootcoach.diagnosis.stage1 import diagnose_stage1
    from shootcoach.target.scoring import group_stats

    spec = load_target_spec()
    prof = ShooterProfile.load(a.profile) if _P(a.profile).exists() else ShooterProfile(_P(a.profile).stem, a.hand)
    st = group_stats(res.holes, spec)
    if a.calibrate:
        prof.add_session(a.calibrate, st)
        prof.save(a.profile)
        print(f"[캘리브레이션] {a.calibrate} 세션 기록 → {a.profile}")
    else:
        s1 = apply_profile(diagnose_stage1(st, spec, a.hand, a.distance, a.click_mm_per_10m), st, prof)
        rep["stage1"] = s1.as_dict()
if a.prev:
    from shootcoach.config import load_target_spec
    from shootcoach.target.sequence import new_holes

    prev = analyze_target(a.prev, handedness=a.hand)
    fresh = new_holes(prev.holes, res.holes, match_mm=load_target_spec().bullet_diameter_mm * 0.35)
    rep["new_holes_since_prev"] = [h.as_dict() for h in fresh]
    print(f"[회차] 이전 사진 대비 새 탄공 {len(fresh)}개")
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
