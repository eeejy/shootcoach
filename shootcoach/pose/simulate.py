"""Synthetic side-view shooter keypoints with injectable faults (for tests and demos without range video)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shootcoach.pose.keypoints import KeypointSeq


@dataclass
class Faults:
    dip_deg: float = 0.0          # 격발 직전 총구 하강 (저킹/반동 예상)
    heel_deg: float = 0.0         # 격발 직전 총구 들림 (힐링)
    shrug: float = 0.0            # 어깨 움츠림 (몸통 길이 비율)
    early_drop: float = 0.0       # 격발 직후 팔 내림 (몸통 길이 비율)
    breath_amp: float = 0.0       # 조준 중 상하 호흡 흔들림 (몸통 길이 비율, 진폭)
    tremor: float = 0.002         # 손 떨림 (몸통 길이 비율)
    sway: float = 0.0             # 몸통 흔들림


def simulate_side_view(shot_times=(2.5, 5.0, 7.5), duration=9.0, fps=60.0, faults: Faults | None = None,
                       seed: int = 0) -> KeypointSeq:
    f = faults or Faults()
    rng = np.random.default_rng(seed)
    T = int(duration * fps)
    t = np.arange(T) / fps
    L = 300.0                                    # torso length px
    hip = np.array([600.0, 900.0])
    sh = hip + [0, -L]
    upper, fore = 0.55 * L, 0.5 * L
    pitch = np.zeros(T)                          # forearm pitch offset (deg), + = muzzle up
    drop = np.zeros(T)
    shrug = np.zeros(T)
    for ts in shot_times:
        pre = (t >= ts - 0.25) & (t < ts)
        ramp = np.clip((t - (ts - 0.25)) / 0.25, 0, 1)
        pitch[pre] += (f.heel_deg - f.dip_deg) * ramp[pre]
        shrug[pre] += f.shrug * ramp[pre]
        rec = (t >= ts) & (t < ts + 0.3)
        pitch[rec] += 25 * np.exp(-(t[rec] - ts) / 0.06) * np.clip((t[rec] - ts) / 0.015, 0, 1)   # recoil
        post = (t >= ts + 0.3) & (t < ts + 2.0)
        up = np.clip((t[post] - ts - 0.3) / 0.3, 0, 1) * np.clip((ts + 2.0 - t[post]) / 0.7, 0, 1)
        drop[post] += f.early_drop * L * up           # lowered, then raised back slowly
    breath = f.breath_amp * L * np.sin(2 * np.pi * 0.3 * t)
    sway = f.sway * L * np.sin(2 * np.pi * 0.5 * t + 1.0)
    kp = np.zeros((T, 17, 2))
    for i in range(T):
        s = sh + [sway[i], -shrug[i] * L]
        elbow = s + upper * np.array([np.cos(np.radians(-5)), -np.sin(np.radians(-5))])
        a = np.radians(pitch[i])
        wrist = elbow + fore * np.array([np.cos(a), -np.sin(a)]) + [0, breath[i] + drop[i]]
        wrist = wrist + rng.normal(0, f.tremor * L, 2)
        nose = sh + [15, -0.35 * L] + [sway[i], 0]
        kp[i, 0] = nose
        kp[i, 1:5] = nose + [[-5, -8], [5, -8], [-15, 0], [10, 0]]
        kp[i, 5] = s + [-6, 0]; kp[i, 6] = s + [6, 0]
        kp[i, 7] = elbow + [-4, 3]; kp[i, 8] = elbow
        kp[i, 9] = wrist + [-2, 2]; kp[i, 10] = wrist
        kp[i, 11] = hip + [-6, 0] + [sway[i] * 0.3, 0]; kp[i, 12] = hip + [6, 0] + [sway[i] * 0.3, 0]
        kp[i, 13:15] = hip + [[0, 0.5 * L], [8, 0.5 * L]]
        kp[i, 15:17] = hip + [[0, L], [8, L]]
    kp += rng.normal(0, 0.6, kp.shape)
    return KeypointSeq(fps, kp, np.full((T, 17), 0.9), (1280, 1280), {"synthetic": True, "shots": list(shot_times)})
