"""Stick-figure frames from a keypoint sequence (demo visuals, VLM key frames for synthetic input)."""
from __future__ import annotations

import cv2
import numpy as np

from shootcoach.pose.keypoints import KeypointSeq

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 5), (0, 6)]


def render_frame(seq: KeypointSeq, i: int, label: str = "") -> np.ndarray:
    w, h = seq.frame_size if seq.frame_size[0] else (1280, 1280)
    img = np.full((h, w, 3), (34, 31, 31), np.uint8)   # BGR #1f1f22, matches the app theme
    k = seq.xy[i]
    for a, b in EDGES:
        if np.isfinite(k[[a, b]]).all():
            cv2.line(img, tuple(int(v) for v in k[a]), tuple(int(v) for v in k[b]), (214, 206, 201), 6, cv2.LINE_AA)
    for j, p in enumerate(k):
        if np.isfinite(p).all():
            cv2.circle(img, tuple(int(v) for v in p), 9, (48, 58, 207) if j in (9, 10) else (130, 138, 143), -1, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (48, 58, 207), 3, cv2.LINE_AA)
    return img


def key_frames(seq: KeypointSeq, t_shot: float, video_path: str | None = None) -> list[np.ndarray]:
    """Frames just before, at, and after the shot — from the video when available."""
    times = [(t_shot - 0.3, "-0.3s"), (t_shot - 0.05, "-0.05s"), (t_shot + 0.5, "+0.5s")]
    out = []
    cap = cv2.VideoCapture(str(video_path)) if video_path else None
    for t, lab in times:
        i = int(np.clip(round(t * seq.fps), 0, len(seq.xy) - 1))
        frame = None
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            frame = frame if ok else None
        out.append(frame if frame is not None else render_frame(seq, i, lab))
    if cap is not None:
        cap.release()
    return out
