"""Find shot (격발) moments: gunshot audio onsets, or the recoil spike in the forearm angle."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def _peaks(x: np.ndarray, thr: float, min_gap: int) -> list[int]:
    idx = []
    i = 0
    while i < len(x):
        if x[i] > thr:
            j = min(len(x), i + min_gap)
            k = i + int(np.argmax(x[i:j]))
            idx.append(k)
            i = k + min_gap
        else:
            i += 1
    return idx


def audio_shot_times(video_path: str | Path, sr: int = 16000, min_gap_s: float = 0.25) -> list[float]:
    """Gunshot onsets from the video's audio track (needs ffmpeg). Empty list if no audio."""
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", str(video_path), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
            capture_output=True, check=True, timeout=120).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    a = np.frombuffer(raw, np.int16).astype(np.float32)
    if a.size < sr // 2:
        return []
    hop = int(sr * 0.005)
    e = np.sqrt(np.convolve(a ** 2, np.ones(hop) / hop, mode="valid")[::hop])
    med = np.median(e)
    mad = np.median(np.abs(e - med)) + 1e-6
    thr = max(med + 12 * mad, 0.35 * e.max())
    if e.max() < 8 * (med + 1):
        return []
    onsets = []
    for p in _peaks(e, thr, int(min_gap_s / 0.005)):
        k = p
        while k > 0 and e[k - 1] > 0.3 * e[p]:
            k -= 1
        onsets.append(k * 0.005)
    return onsets


def motion_shot_times(pitch_deg: np.ndarray, fps: float, min_gap_s: float = 0.3) -> list[float]:
    """Recoil = sharp upward jump of the forearm pitch. Returns onset times (start of the jump)."""
    p = np.asarray(pitch_deg, float)
    if np.isnan(p).all() or len(p) < 5:
        return []
    p = np.where(np.isnan(p), np.nanmedian(p), p)
    v = np.gradient(p) * fps                     # deg/s
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-6
    thr = max(med + 10 * mad, 120.0)              # recoil flips the muzzle by hundreds of deg/s
    onsets = []
    back = max(2, int(0.4 * fps))
    for pk in _peaks(v, thr, max(1, int(min_gap_s * fps))):
        k = pk
        while k > 0 and v[k - 1] > 0.25 * v[pk]:
            k -= 1
        # Recoil flips the muzzle up and it comes back down; a posture change (e.g. raising
        # the arm again) is a step that stays. Require ≥50% return within 0.4 s.
        top = int(np.argmax(p[pk:pk + back])) + pk
        rise = p[top] - p[k]
        after = p[top:top + back]
        if rise <= 0 or (p[top] - after.min()) < 0.5 * rise:
            continue
        onsets.append(k / fps)
    return onsets
