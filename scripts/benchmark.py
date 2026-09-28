"""Measure end-to-end latency on this machine (target pipeline, pose, local LLM explanation).

    python scripts/benchmark.py --out docs/benchmark.json
"""
import argparse
import json
import platform
import statistics as stats
import subprocess
import time
from pathlib import Path

import cv2

from shootcoach.explain.vlm import installed_models, vlm_explanation
from shootcoach.pipeline import analyze_target
from shootcoach.target.detect import ClassicHoleDetector, YoloHoleDetector


def chip():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/benchmark.json")
    ap.add_argument("--video", default="data/real_video/clip_army_5_9.mp4")
    a = ap.parse_args()
    res = {"machine": chip(), "python": platform.python_version()}
    imgs = sorted(Path("samples").glob("demo_*.jpg"))
    for name, det in (("yolo", YoloHoleDetector()), ("classic", ClassicHoleDetector())):
        analyze_target(str(imgs[0]), detector=det)                       # warm-up
        ts = [analyze_target(str(p), detector=det).report["timings"]["total_s"] for p in imgs for _ in range(3)]
        res[f"target_pipeline_{name}_ms"] = {"median": round(1000 * stats.median(ts), 1), "max": round(1000 * max(ts), 1)}
    if Path(a.video).exists():
        from shootcoach.pose.keypoints import extract_keypoints
        t0 = time.perf_counter()
        seq = extract_keypoints(a.video)
        dt = time.perf_counter() - t0
        res["pose_ms_per_frame"] = round(1000 * dt / len(seq.xy), 1)
        res["pose_clip"] = {"frames": len(seq.xy), "fps": round(seq.fps, 2), "seconds": round(dt, 2)}
    rep = analyze_target(str(Path("samples/demo_jerking_low_left.jpg")), detector=YoloHoleDetector())
    res["llm"] = {}
    for m in ("qwen3:8b", "qwen2.5vl:3b"):
        if not any(n.startswith(m) for n in installed_models()):
            continue
        vlm_explanation(rep.report, [rep.overlay], model=m)               # warm-up (model load)
        runs = [vlm_explanation(rep.report, [rep.overlay], model=m) for _ in range(3)]
        res["llm"][m] = {"latency_s_median": stats.median(r["latency_s"] for r in runs),
                         "accepted": sum(r["source"] == "vlm" for r in runs), "runs": 3,
                         "sample": runs[0]["text"][:200], "error": runs[0].get("error", "")[:160]}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
