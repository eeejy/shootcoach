"""Pose keypoint sequences (COCO-17) from video, via Ultralytics YOLO-pose."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# COCO-17 indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 5, 6, 7, 8, 9, 10, 11, 12


@dataclass
class KeypointSeq:
    fps: float
    xy: np.ndarray              # (T, 17, 2) image px, y down; NaN where missing
    conf: np.ndarray            # (T, 17)
    frame_size: tuple[int, int] = (0, 0)
    meta: dict = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.xy)) / self.fps


def extract_keypoints(video_path: str | Path, model: str = "yolo11n-pose.pt", device: str | None = None,
                      max_frames: int | None = None, stride: int = 1) -> KeypointSeq:
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    net = YOLO(model)
    xs, cs, cuts, diffs = [], [], [], []
    prev_hist = prev_small = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        # Edited footage has scene cuts; a cut is not a recoil, so remember where they are.
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_hist is not None:
            bhatt = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            mad = float(np.abs(small - prev_small).mean())
            base = float(np.median(diffs)) if diffs else mad
            if bhatt > 0.35 or (mad > 30 and mad > 4 * base):
                cuts.append(i / fps)
            diffs.append(mad)
        prev_hist, prev_small = hist, small
        if i % stride == 0:
            r = net.predict(frame, device=device, verbose=False)[0]
            if r.keypoints is not None and len(r.keypoints) and r.keypoints.conf is not None:
                boxes = r.boxes.xywh.cpu().numpy()
                k = int(np.argmax(boxes[:, 2] * boxes[:, 3]))          # largest person = shooter
                xs.append(r.keypoints.xy[k].cpu().numpy())
                cs.append(r.keypoints.conf[k].cpu().numpy())
            else:
                xs.append(np.full((17, 2), np.nan))
                cs.append(np.zeros(17))
        i += 1
    cap.release()
    return KeypointSeq(fps / stride, np.asarray(xs, float), np.asarray(cs, float), (w, h),
                       {"source": str(video_path), "scene_cuts": cuts})
