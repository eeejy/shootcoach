"""End-to-end analysis: 사진 → 보정 → 탄공 검출 → 채점·통계 → 1단계 → (자세 영상) 2단계 → 설명."""
from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import TargetSpec, load_target_spec
from shootcoach.diagnosis.stage1 import diagnose_stage1
from shootcoach.target.detect import Hole, get_detector
from shootcoach.target.markers import MarkerError, rectify
from shootcoach.target.scoring import group_stats


@dataclass
class TargetAnalysis:
    report: dict
    rectified: np.ndarray | None
    overlay: np.ndarray | None
    holes: list[Hole]


def draw_overlay(rect: np.ndarray, holes: list[Hole], report: dict, spec: TargetSpec) -> np.ndarray:
    out = rect.copy()
    ppm = spec.px_per_mm
    for h, s in zip(holes, report["group"]["scores"]):
        c = (int(h.x_mm * ppm), int(h.y_mm * ppm))
        cv2.circle(out, c, int(h.r_mm * ppm * 1.15), (0, 200, 255), 3, cv2.LINE_AA)
        cv2.putText(out, str(s), (c[0] + int(h.r_mm * ppm) + 4, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 120, 255), 2, cv2.LINE_AA)
    g = report["group"]
    if g["n"] >= 2:
        gx = spec.center_mm[0] + g["center_mm"][0]
        gy = spec.center_mm[1] - g["center_mm"][1]
        gc = (int(gx * ppm), int(gy * ppm))
        cv2.circle(out, gc, max(4, int(g["mean_radius_mm"] * ppm)), (255, 80, 0), 3, cv2.LINE_AA)
        cv2.drawMarker(out, gc, (255, 80, 0), cv2.MARKER_CROSS, 30, 3)
        tc = (int(spec.center_mm[0] * ppm), int(spec.center_mm[1] * ppm))
        cv2.arrowedLine(out, tc, gc, (60, 200, 60), 3, cv2.LINE_AA, tipLength=0.08)
    return out


def analyze_target(image: np.ndarray | str | Path, spec: TargetSpec | None = None, handedness: str = "right",
                   distance_m: float | None = None, click_mm_per_10m: float | None = None,
                   detector=None, already_rectified: bool = False) -> TargetAnalysis:
    spec = spec or load_target_spec()
    if not isinstance(image, np.ndarray):
        image = cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError("이미지를 읽을 수 없습니다.")
    t0 = time.perf_counter()
    timings = {}
    if already_rectified:
        rect, reproj, markers = cv2.resize(image, spec.canvas_px), 0.0, []
    else:
        r = rectify(image, spec)
        rect, reproj, markers = r.image, r.reprojection_mm, r.marker_ids
    timings["rectify_s"] = time.perf_counter() - t0
    det = detector or get_detector()
    t1 = time.perf_counter()
    holes = det.detect(rect, spec)
    timings["detect_s"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    st = group_stats(holes, spec)
    s1 = diagnose_stage1(st, spec, handedness, distance_m, click_mm_per_10m)
    timings["diagnose_s"] = time.perf_counter() - t2
    timings["total_s"] = time.perf_counter() - t0
    report = {
        "target": spec.name,
        "detector": type(det).__name__,
        "rectification": {"markers": markers, "reprojection_mm": round(reproj, 3)},
        "holes": [h.as_dict() for h in holes],
        "group": st.as_dict(),
        "stage1": s1.as_dict(),
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    return TargetAnalysis(report, rect, draw_overlay(rect, holes, report, spec), holes)


def analyze_posture(report: dict, video_path: str | Path, handedness: str = "right", view: str | None = None,
                    pose_model: str = "yolo11n-pose.pt") -> dict:
    from shootcoach.diagnosis.stage1 import Candidate, Stage1Result, ZeroAdjust
    from shootcoach.diagnosis.stage2 import diagnose_stage2
    from shootcoach.pose.keypoints import extract_keypoints
    from shootcoach.pose.shots import audio_shot_times

    t0 = time.perf_counter()
    seq = extract_keypoints(video_path, pose_model)
    shots = audio_shot_times(video_path)
    d = report["stage1"]
    s1 = Stage1Result(d["shape"], d["shape_ko"], d["sector"], d["sector_ko"], d["handedness"],
                      [Candidate(**c) for c in d["candidates"]],
                      ZeroAdjust(**d["zero_adjust"]) if d["zero_adjust"] else None, d["notes"])
    s2 = diagnose_stage2(s1, seq, shots or None, handedness, view)
    out = s2.as_dict()
    out["shot_source"] = "audio" if shots else "motion"
    out["latency_s"] = round(time.perf_counter() - t0, 2)
    report["stage2"] = out
    return report


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def to_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=_np_default)
