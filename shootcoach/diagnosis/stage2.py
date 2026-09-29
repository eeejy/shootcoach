"""2단계: 자세 신호로 1단계 원인 후보를 확정 / 배제 / 관측 불가로 판정."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

from shootcoach.diagnosis.rules import RULES_DIR
from shootcoach.diagnosis.stage1 import Stage1Result
from shootcoach.pose.features import arm_series, detect_view, shot_features
from shootcoach.pose.keypoints import KeypointSeq
from shootcoach.pose.shots import motion_shot_times


@dataclass(frozen=True)
class SignalDef:
    signal_id: str
    name_ko: str
    feature: str
    compare: str
    threshold: float
    views: tuple[str, ...]
    description_ko: str


@lru_cache(maxsize=2)
def load_signals() -> dict[str, SignalDef]:
    with open(RULES_DIR / "posture_signals.csv", encoding="utf-8") as f:
        return {r["signal_id"]: SignalDef(r["signal_id"], r["name_ko"], r["feature"], r["compare"],
                                          float(r["threshold"]), tuple(r["views"].split(";")), r["description_ko"])
                for r in csv.DictReader(f)}


def _strength(sig: SignalDef, x: float) -> float:
    """≥1 means the signal is present; 0.5 ≈ half-way to the threshold."""
    if not np.isfinite(x):
        return float("nan")
    if sig.compare == "gt":
        return x / sig.threshold
    if sig.compare == "lt":
        return x / sig.threshold          # both negative → positive ratio
    if sig.compare == "abs_gt":
        return abs(x) / sig.threshold
    raise ValueError(sig.compare)


@dataclass
class SignalResult:
    signal_id: str
    name_ko: str
    observable: bool
    present: bool
    strength: float           # mean over shots
    shots_present: int
    shots_total: int


@dataclass
class CauseVerdict:
    cause_id: str
    cause_ko: str
    status: str               # confirmed | ruled_out | unobservable
    status_ko: str
    confidence: float
    evidence: list[str]


@dataclass
class Stage2Result:
    view: str
    shot_times: list[float]
    signals: dict[str, SignalResult]
    verdicts: list[CauseVerdict]
    final_cause_id: str | None
    final_ko: str
    extra_findings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def evaluate_signals(seq: KeypointSeq, shot_times: list[float], handedness: str, view: str) -> dict[str, SignalResult]:
    sigs = load_signals()
    feats = [shot_features(seq, ts, handedness) for ts in shot_times]
    out = {}
    for sid, s in sigs.items():
        observable = view in s.views
        st = [_strength(s, f.values.get(s.feature, float("nan"))) for f in feats]
        st = [x for x in st if np.isfinite(x)]
        n_present = sum(1 for x in st if x >= 1.0)
        mean = float(np.mean(st)) if st else float("nan")
        present = observable and bool(st) and n_present >= max(1, int(np.ceil(0.5 * len(st))))
        out[sid] = SignalResult(sid, s.name_ko, observable and bool(st), present, round(mean, 2), n_present, len(st))
    return out


STATUS_KO = {"confirmed": "확정", "ruled_out": "배제", "unobservable": "관측 불가 (후보 유지)"}


def diagnose_stage2(stage1: Stage1Result, seq: KeypointSeq, shot_times: list[float] | None = None,
                    handedness: str = "right", view: str | None = None) -> Stage2Result:
    view = view or detect_view(seq)
    notes = []
    if not shot_times:
        _, pitch, _ = arm_series(seq, handedness)
        shot_times = motion_shot_times(pitch, seq.fps)
        notes.append("격발 시점: 반동 동작으로 추정" if shot_times else "격발 시점을 찾지 못했습니다.")
    cuts = seq.meta.get("scene_cuts", [])
    if cuts and shot_times:
        kept = [t for t in shot_times if min(abs(t - c) for c in cuts) > 0.3]
        if len(kept) < len(shot_times):
            notes.append(f"영상 편집 컷 근처의 격발 후보 {len(shot_times) - len(kept)}개를 제외했습니다 (컷은 반동이 아님).")
        shot_times = kept
    if not shot_times:
        return Stage2Result(view, [], {}, [], None, "보류 — 격발 시점을 찾지 못함", notes=notes)
    signals = evaluate_signals(seq, shot_times, handedness, view)
    verdicts = []
    linked = set()
    for c in stage1.candidates:
        sids = [s for s in c.posture_signals if s in signals]
        linked |= set(sids)
        obs = [signals[s] for s in sids if signals[s].observable]
        if not obs:
            status, conf, ev = "unobservable", c.score * 0.5, [c.observable_note or "이 원인은 자세 영상으로 확인할 신호가 없습니다."]
        elif any(s.present for s in obs):
            best = max((s for s in obs if s.present), key=lambda s: s.strength)
            status = "confirmed"
            n_ok = sum(1 for s in obs if s.present)
            # 뒷받침하는 신호가 많을수록(증거 기반 다중 추론) 신뢰도가 높다
            conf = min(0.99, c.score * 0.35 + 0.5 * min(1.0, best.strength / 2) + 0.12 * (n_ok - 1) + 0.05 * n_ok / len(obs))
            ev = [f"{s.name_ko}: 강도 {s.strength:.1f} ({s.shots_present}/{s.shots_total}발)" for s in obs if s.present]
        else:
            status, conf = "ruled_out", c.score * 0.2
            ev = [f"{s.name_ko} 관찰 안 됨 (강도 {s.strength:.1f})" for s in obs]
        verdicts.append(CauseVerdict(c.cause_id, c.cause_ko, status, STATUS_KO[status], round(conf, 2), ev))
    order = {"confirmed": 0, "unobservable": 1, "ruled_out": 2}
    verdicts.sort(key=lambda v: (order[v.status], -v.confidence))
    confirmed = [v for v in verdicts if v.status == "confirmed"]
    if confirmed:
        final_id, final_ko = confirmed[0].cause_id, f"확정 — {confirmed[0].cause_ko}"
    else:
        final_id = None
        final_ko = "보류 — 자세 영상에서 후보 원인을 뒷받침하는 신호가 없습니다" if verdicts else "보류"
    extra = [f"{s.name_ko} (강도 {s.strength:.1f}) — 표적 후보와 별개로 관찰됨"
             for sid, s in signals.items() if s.present and sid not in linked]
    if view == "side":
        notes.append("측면 영상: 좌우 흔들림 신호는 측정하지 않습니다 (후방 영상 필요).")
    return Stage2Result(view, [round(x, 3) for x in shot_times], signals, verdicts, final_id, final_ko, extra, notes)
