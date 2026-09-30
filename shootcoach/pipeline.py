"""End-to-end analysis: 사진 → 보정 → 탄공 검출 → 채점·통계 → 1단계 → (자세 영상) 2단계 → 설명."""
from __future__ import annotations

import time
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import TargetSpec, load_target_spec
from shootcoach.diagnosis.stage1 import diagnose_stage1
from shootcoach.target.detect import Hole
from shootcoach.target.locate import locate_target, rectified_view
from shootcoach.target.scoring import group_stats

DEFAULT_SPEC = "configs/kcg_circle.yaml"


def default_spec() -> TargetSpec:
    from shootcoach.config import REPO_ROOT

    return load_target_spec(REPO_ROOT / DEFAULT_SPEC)


def get_photo_detector():
    """The photo-trained hole detector (models/hole_detector_photo.pt)."""
    from shootcoach.target.photo import PHOTO_ONNX_WEIGHTS, PHOTO_WEIGHTS, OpenCVDNNHoleDetector, PhotoHoleDetector

    if os.environ.get("VERCEL") or os.environ.get("SHOOTCOACH_ONNX"):
        if not PHOTO_ONNX_WEIGHTS.exists():
            raise FileNotFoundError(f"ONNX 탄공 모델이 없습니다: {PHOTO_ONNX_WEIGHTS}")
        return OpenCVDNNHoleDetector()

    if not PHOTO_WEIGHTS.exists():
        raise FileNotFoundError(f"탄공 모델이 없습니다: {PHOTO_WEIGHTS} (scripts/setup.sh 로 설치)")
    return PhotoHoleDetector()


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


def detect_holes_px(det, image: np.ndarray, frame, spec: TargetSpec, expected_shots: int | None):
    """AI 검출 + 발수 보정 + (모자라면) 검은 원 안 밝은 탄공 보충. 사진 px 좌표로 돌려준다."""
    from shootcoach.target.photo import bright_holes_in_disc

    outer = spec.outer_radius_mm * 1.08

    def inside(x, y):
        return float(np.hypot(*frame.to_target_mm(np.array([[x, y]]))[0])) <= outer

    px, log = det.detect(image, expected_shots, inside)
    if expected_shots and len(px) < expected_shots:
        extra = bright_holes_in_disc(image, frame, spec, px, expected_shots - len(px))
        px = px + extra
        log["bright_in_disc"] = len(extra)
        log["found"] = len(px)
    return px, log


def analyze_target(image: np.ndarray | str | Path, spec: TargetSpec | None = None, handedness: str = "right",
                   distance_m: float | None = None, click_mm_per_10m: float | None = None,
                   detector=None, expected_shots: int | None = 10, holes_override: list[Hole] | None = None
                   ) -> TargetAnalysis:
    """사진 → 검은 원으로 표적 찾기(마커 없음) → 탄공 검출·발수 보정 → 채점·통계 → 1단계 진단.

    holes_override: 교관이 수정한 탄공 목록(표적 mm 좌표). 주면 검출을 건너뛰고 그대로 채점한다.
    """
    spec = spec or default_spec()
    if not isinstance(image, np.ndarray):
        image = cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError("이미지를 읽을 수 없습니다.")
    t0 = time.perf_counter()
    timings = {}
    frame = locate_target(image, spec)
    rect = rectified_view(image, frame, spec)
    timings["locate_s"] = time.perf_counter() - t0
    det = detector if detector is not None else get_photo_detector()
    t1 = time.perf_counter()
    det_log: dict = {}
    if holes_override is not None:
        holes = list(holes_override)
        det_log = {"source": "instructor"}
    else:                                                       # detect on the original photo
        px, det_log = detect_holes_px(det, image, frame, spec, expected_shots)
        pts = frame.paper_mm(np.array([[h.x, h.y] for h in px]).reshape(-1, 2), spec)
        holes = [Hole(float(x), float(y), max(spec.bullet_diameter_mm / 2 * 0.6, h.r * frame.mm_per_px), h.conf)
                 for (x, y), h in zip(pts, px)]
    timings["detect_s"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    st = group_stats(holes, spec)
    s1 = diagnose_stage1(st, spec, handedness, distance_m, click_mm_per_10m)
    timings["diagnose_s"] = time.perf_counter() - t2
    timings["total_s"] = time.perf_counter() - t0
    notes = []
    if expected_shots and holes_override is None and len(holes) != expected_shots:
        notes.append(f"탄공 {len(holes)}개 검출 (기준 {expected_shots}발). 빠졌거나 잘못 잡힌 탄공은 사진을 눌러 고쳐 주세요.")
    report = {
        "target": spec.name,
        "detector": type(det).__name__,
        "frame": {"center_px": [round(v, 1) for v in frame.center_px], "axes_px": [round(v, 1) for v in frame.axes_px],
                  "fill": round(frame.fill, 3), "mm_per_px": round(frame.mm_per_px, 3)},
        "detection": det_log,
        "expected_shots": expected_shots,
        "holes": [h.as_dict() for h in holes],
        "group": st.as_dict(),
        "stage1": s1.as_dict(),
        "notes": notes,
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    return TargetAnalysis(report, rect, draw_overlay(rect, holes, report, spec), holes)


def diagnose_posture(stage1, video_path: str | Path, handedness: str = "right", view: str | None = None,
                     pose_model: str | None = None) -> dict:
    """자세 영상 → 2단계 판정 dict. stage1 후보 목록(표적지 분석 결과 또는 standalone_stage1())에 대해 확정/배제한다.

    표적지 분석과 별개로 쓸 수 있다: 자세 영상만 있을 때는 stage1=standalone_stage1(handedness) 를 넘긴다.
    """
    from shootcoach.diagnosis.stage2 import diagnose_stage2
    from shootcoach.pose.keypoints import extract_keypoints
    from shootcoach.pose.shots import audio_shot_times

    t0 = time.perf_counter()
    seq = extract_keypoints(video_path, pose_model)
    shots = audio_shot_times(video_path)
    s2 = diagnose_stage2(stage1, seq, shots or None, handedness, view)
    out = s2.as_dict()
    out["shot_source"] = "audio" if shots else "motion"
    out["latency_s"] = round(time.perf_counter() - t0, 2)
    return out


def analyze_posture(report: dict, video_path: str | Path, handedness: str = "right", view: str | None = None,
                    pose_model: str | None = None) -> dict:
    """표적지 분석 결과(report)에 자세 영상 판정을 덧붙인다 (CLI용). 앱은 diagnose_posture 를 직접 쓴다."""
    from shootcoach.diagnosis.stage1 import Candidate, Stage1Result, ZeroAdjust

    d = report["stage1"]
    s1 = Stage1Result(d["shape"], d["shape_ko"], d["sector"], d["sector_ko"], d["handedness"],
                      [Candidate(**c) for c in d["candidates"]],
                      ZeroAdjust(**d["zero_adjust"]) if d["zero_adjust"] else None, d["notes"])
    report["stage2"] = diagnose_posture(s1, video_path, handedness, view, pose_model)
    return report


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def to_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=_np_default)


def count_holes(image: np.ndarray, detector=None, expected_shots: int | None = None) -> dict:
    """원형 표적이 아닐 때(속사 하반신 표적 등): 채점 없이 탄공 위치·개수만.

    영역 채점(2·5·4점)은 속사 표적 규정을 받은 뒤 추가한다.
    """
    det = detector if detector is not None else get_photo_detector()
    px, log = det.detect(image, expected_shots)
    vis = image.copy()
    lw = max(2, image.shape[1] // 400)
    for i, h in enumerate(px, 1):
        c = (int(h.x), int(h.y))
        cv2.circle(vis, c, int(max(h.r * 1.4, 6)), (0, 200, 255), lw, cv2.LINE_AA)
        cv2.putText(vis, str(i), (c[0] + int(h.r * 1.4) + 2, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5 + lw * 0.15,
                    (0, 120, 255), lw, cv2.LINE_AA)
    return {"n": len(px), "expected": expected_shots, "detection": log, "overlay": vis,
            "holes_px": [[round(h.x, 1), round(h.y, 1), round(h.r, 1), round(h.conf, 3)] for h in px]}
