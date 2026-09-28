"""Per-shot posture features around each 격발 moment, normalised by torso length."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shootcoach.pose.keypoints import L_EL, L_HIP, L_SH, L_WR, NOSE, R_EL, R_HIP, R_SH, R_WR, KeypointSeq

MIN_CONF = 0.3


def _pt(seq: KeypointSeq, idx: list[int]) -> np.ndarray:
    """Confidence-weighted mean of several keypoints per frame → (T,2), NaN if none visible."""
    xy = seq.xy[:, idx, :]
    w = np.where(seq.conf[:, idx] >= MIN_CONF, seq.conf[:, idx], 0.0)
    s = w.sum(1, keepdims=True)
    out = (xy * w[..., None]).sum(1) / np.where(s == 0, 1, s)
    out[s[:, 0] == 0] = np.nan
    return out


def detect_view(seq: KeypointSeq) -> str:
    """'side' when the shoulders overlap in the image, else 'rear' (front/back)."""
    sw = np.linalg.norm(seq.xy[:, L_SH] - seq.xy[:, R_SH], axis=1)
    torso = torso_length(seq)
    ratio = np.nanmedian(sw) / torso if torso else 0
    return "side" if ratio < 0.45 else "rear"


def torso_length(seq: KeypointSeq) -> float:
    sh = _pt(seq, [L_SH, R_SH])
    hp = _pt(seq, [L_HIP, R_HIP])
    return float(np.nanmedian(np.linalg.norm(sh - hp, axis=1)))


def arm_series(seq: KeypointSeq, handedness: str = "right"):
    el_idx, sh_idx = ([R_EL], [R_SH]) if handedness == "right" else ([L_EL], [L_SH])
    wrist = _pt(seq, [L_WR, R_WR])           # two-hand grip: both wrists hold the pistol
    elbow = _pt(seq, el_idx)
    elbow = np.where(np.isnan(elbow), _pt(seq, [L_EL, R_EL]), elbow)
    shoulder = _pt(seq, sh_idx)
    shoulder = np.where(np.isnan(shoulder), _pt(seq, [L_SH, R_SH]), shoulder)
    d = wrist - elbow
    pitch = np.degrees(np.arctan2(-d[:, 1], np.abs(d[:, 0])))        # + = muzzle up (image y down)
    v1, v2 = shoulder - elbow, wrist - elbow
    cosang = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    elbow_ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return wrist, pitch, elbow_ang


@dataclass
class ShotFeatures:
    t_shot: float
    values: dict[str, float]


def _win(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return (t >= a) & (t < b)


def _nanmean(x):
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def shot_features(seq: KeypointSeq, t_shot: float, handedness: str = "right") -> ShotFeatures:
    t = seq.t
    L = torso_length(seq) or 1.0
    wrist, pitch, elbow_ang = arm_series(seq, handedness)
    sh = _pt(seq, [L_SH, R_SH])
    nose = _pt(seq, [NOSE])
    base = _win(t, t_shot - 1.0, t_shot - 0.35)
    pre = _win(t, t_shot - 0.30, t_shot - 0.02)
    late_pre = _win(t, t_shot - 0.12, t_shot - 0.02)
    long_pre = _win(t, t_shot - 2.0, t_shot - 0.3)
    post = _win(t, t_shot + 0.35, t_shot + 1.0)

    def lowpass(x, n):
        k = max(1, int(n))
        valid = np.isfinite(x)
        if valid.sum() < 2:
            return x
        xi = np.interp(np.arange(len(x)), np.flatnonzero(valid), x[valid])
        return np.convolve(xi, np.ones(k) / k, mode="same")

    wy_lp = lowpass(wrist[:, 1], 0.3 * seq.fps)
    shy_lp = lowpass(sh[:, 1], 0.3 * seq.fps)
    shx_lp = lowpass(sh[:, 0], 0.3 * seq.fps)
    v = {}
    v["forearm_pitch_delta_deg"] = _nanmean(pitch[late_pre]) - _nanmean(pitch[base])
    v["wrist_drop_rel"] = (_nanmean(wrist[late_pre, 1]) - _nanmean(wrist[base, 1])) / L
    v["shoulder_rise_rel"] = ((_nanmean(sh[base, 1]) - _nanmean(nose[base, 1]))
                              - (_nanmean(sh[late_pre, 1]) - _nanmean(nose[late_pre, 1]))) / L
    v["wrist_lateral_rel"] = abs(_nanmean(wrist[late_pre, 0]) - _nanmean(wrist[base, 0])) / L
    jitter = wrist[base] - np.c_[wy_lp[base] * 0 + np.nanmean(wrist[base, 0]), wy_lp[base]]
    v["wrist_jitter_rel"] = float(np.nanstd(np.linalg.norm(jitter, axis=1))) / L if base.any() else float("nan")
    v["wrist_vertical_ptp_rel"] = float(np.ptp(wy_lp[long_pre])) / L if long_pre.any() else float("nan")
    v["torso_sway_rel"] = float(max(np.ptp(shy_lp[long_pre]), np.ptp(shx_lp[long_pre]))) / L if long_pre.any() else float("nan")
    if base.sum() >= 3:
        tt = t[base]
        slope = np.polyfit(tt - tt[0], np.nan_to_num(wrist[base, 1], nan=np.nanmean(wrist[base, 1])), 1)[0]
        v["wrist_sag_rel"] = float(slope * 1.0) / L       # px/s → rel per second
    else:
        v["wrist_sag_rel"] = float("nan")
    v["elbow_angle_delta_deg"] = _nanmean(elbow_ang[pre]) - _nanmean(elbow_ang[base])
    v["wrist_drop_post_rel"] = (_nanmean(wrist[post, 1]) - _nanmean(wrist[pre, 1])) / L
    v["head_rise_post_rel"] = (_nanmean(nose[pre, 1]) - _nanmean(nose[post, 1])) / L
    return ShotFeatures(t_shot, {k: float(x) for k, x in v.items()})
