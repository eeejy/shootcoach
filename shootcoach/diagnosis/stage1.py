"""1단계: 표적지 탄착군 → 원인 후보 (교관 지식베이스 · 다중 변수 가중합).

교관 자료 원칙: Pattern → Candidate Causes. '좌하탄 = 방아쇠' 같은 1차원 매핑을 쓰지 않는다.
- 탄착군 중심을 상하 성분과 좌우 성분으로 나눠 각각의 크기만큼 해당 축 원인에 점수를 준다.
- 두 성분이 함께 크면(대각선) 교관 자료의 '복합 편향 벡터'에 해당하는 원인에 가산한다.
- 탄착군 크기(분산)는 독립 축으로 진단한다. 조밀한데 치우쳤으면 영점을 먼저 본다.
- 사수 프로필(손 크기·손가락 길이·피로도·그립 적합성)과 사전 점검(영점)을 반영한다.
오른손잡이 기준이며, 왼손잡이는 좌우를 뒤집는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import yaml

from shootcoach.config import REPO_ROOT, TargetSpec
from shootcoach.target.scoring import CLOCK_LABELS_KO, GroupStats

KB_PATH = REPO_ROOT / "rules" / "instructor_kb.yaml"
SOURCE = "교관 KB"


@dataclass
class Candidate:
    cause_id: str
    cause_ko: str
    score: float
    guidance_ko: str
    drill_ko: str
    posture_signals: list[str]
    observable_note: str
    rule_ids: list[str]
    sources: list[str]
    checklist: list[str] = field(default_factory=list)
    axis: str = ""


@dataclass
class ZeroAdjust:
    move_right_mm: float      # + → 탄착을 오른쪽으로 옮겨야 함
    move_up_mm: float         # + → 탄착을 위로 옮겨야 함
    text_ko: str
    clicks_windage: int | None = None
    clicks_elevation: int | None = None


@dataclass
class Stage1Result:
    shape: str
    shape_ko: str
    sector: float
    sector_ko: str
    handedness: str
    candidates: list[Candidate]
    zero_adjust: ZeroAdjust | None
    notes: list[str] = field(default_factory=list)
    components: dict = field(default_factory=dict)      # 상하·좌우·분산 가중치 (설명용)

    def as_dict(self):
        return asdict(self)


@dataclass
class ShooterProfile:
    """교관 자료 Axis 2 (사수) + Phase 00 (사전 점검)."""
    hand_size: str = "medium"          # small | medium | large
    finger_length: str = "normal"      # short | normal | long
    fatigue: str = "low"               # low | high
    zero_confirmed: str = "unknown"    # yes | no | unknown
    grip_fit: str = "ok"               # ok | too_large | too_small  (총기 그립 크기 대비 손)


@lru_cache(maxsize=2)
def load_kb(path: str | None = None) -> dict:
    return yaml.safe_load(open(path or KB_PATH, encoding="utf-8"))


def _zero_adjust(stats: GroupStats) -> ZeroAdjust:
    cx, cy = stats.center_mm
    right, up = -cx, -cy
    parts = []
    if abs(right) >= 1:
        parts.append(f"{'오른쪽' if right > 0 else '왼쪽'}으로 {abs(right):.0f}mm")
    if abs(up) >= 1:
        parts.append(f"{'위' if up > 0 else '아래'}로 {abs(up):.0f}mm")
    text = ("탄착군을 " + ", ".join(parts) + " 옮기도록 조준기를 조정하세요 (한 발이 아니라 탄착군 기준)."
            if parts else "조정 불필요.")
    return ZeroAdjust(right, up, text)


def _ramp(x: float, lo: float, hi: float) -> float:
    return 0.0 if x <= lo else 1.0 if x >= hi else (x - lo) / (hi - lo)


def components(stats: GroupStats, spec: TargetSpec, handedness: str = "right") -> dict:
    """상하·좌우 편향과 분산의 세기 (0~1). 링 간격 기준이라 표적 크기와 무관."""
    s = spec.ring_step_mm
    cx, cy = stats.center_mm
    if handedness != "right":
        cx = -cx
    return {
        "low": _ramp(-cy / s, 0.35, 1.6), "high": _ramp(cy / s, 0.35, 1.6),
        "left": _ramp(-cx / s, 0.35, 1.6), "right": _ramp(cx / s, 0.35, 1.6),
        "spread": _ramp(stats.mean_radius_mm / s, 1.3, 3.0),
    }


def diagnose_stage1(stats: GroupStats, spec: TargetSpec, handedness: str = "right",
                    distance_m: float | None = None, click_mm_per_10m: float | None = None,
                    top_k: int = 5, profile: ShooterProfile | None = None) -> Stage1Result:
    kb = load_kb()
    profile = profile or ShooterProfile()
    notes: list[str] = []
    if handedness != "right":
        notes.append("왼손잡이: 오른손잡이 기준 좌우 원인을 뒤집어 적용했습니다.")
    if stats.shape == "insufficient":
        return Stage1Result(stats.shape, stats.shape_ko, stats.sector, stats.sector_ko, handedness, [], None,
                            ["탄 수가 부족합니다. 한 발만 보고 원인을 판단하거나 조준기를 조정하지 마세요 (최소 3발, 권장 5발)."])
    comp = components(stats, spec, handedness)
    scores: dict[str, float] = {}
    why: dict[str, list[str]] = {}

    def add(cid: str, v: float, reason: str):
        if v <= 0:
            return
        scores[cid] = scores.get(cid, 0.0) + v
        why.setdefault(cid, []).append(reason)

    for cid, c in kb["causes"].items():
        ax = c["axis"]
        if ax in ("low", "high", "left", "right", "spread"):
            add(cid, c["prior"] * comp[ax], ax)
    # 대각선 = 두 성분의 결합 → 짝지어진 원인 가산
    for key, d in kb["diagonals"].items():
        v_ax, h_ax = ("high" if "high" in key else "low"), ("left" if "left" in key else "right")
        w = min(comp[v_ax], comp[h_ax])
        if w > 0.3:
            for cid in d["boost"]:
                add(cid, 0.35 * w, f"diag:{key}")
    # 조밀한데 한쪽으로 몰림 → 영점 우선 (교관 자료: 조밀한 편향은 총기 영점 오류를 우선 확인)
    zero = None
    if stats.shape == "tight_offset":
        add("Z1", 1.2, "tight_offset")
        zero = _zero_adjust(stats)
        if profile.zero_confirmed == "yes":
            scores["Z1"] *= 0.35
            notes.append("영점이 확인된 총기라고 입력되어, 일관된 동작 오류 가능성을 더 높게 봅니다.")
        else:
            notes.append("탄착군이 촘촘한데 한쪽으로 몰려 있습니다. 자세를 고치기 전에 총기 영점부터 확인하세요.")
    elif stats.shape in ("scattered_offset", "scattered_centered", "vertical_string", "horizontal_string"):
        notes.append("탄착군이 흩어져 있을 때는 조준기를 조정하지 마세요. 먼저 동작을 안정시켜 탄착군을 모읍니다.")
    if stats.shape == "vertical_string":
        add("S1", 0.5, "vertical_string")
    if stats.shape == "horizontal_string":
        add("S3", 0.35, "horizontal_string")
        add("LB3" if handedness == "right" else "RB3", 0.2, "horizontal_string")
    # 사수 프로필 배수
    for cid in list(scores):
        for field_, mult in (kb["causes"][cid].get("profile") or {}).items():
            m = mult.get(getattr(profile, field_, None))
            if m:
                scores[cid] *= m
                why[cid].append(f"profile:{field_}")
    if profile.grip_fit == "too_large":
        for cid in ("LB4", "H1"):
            if cid in scores:
                scores[cid] *= 1.5
                why[cid].append("profile:grip_fit")
    if stats.n_fliers:
        notes.append(f"주 탄착군에서 벗어난 탄 {stats.n_fliers}발은 일시적 변동일 수 있어 중심 계산에서 제외했습니다.")
    if stats.n < 5:
        notes.append("5발 미만이라 판정 신뢰도가 낮습니다.")

    if not scores:
        good = Candidate("G0", "양호한 탄착군", 1.0, "탄착군이 중앙에 촘촘합니다. 현재 자세와 격발을 유지하세요.",
                         "5발 1세트 반복 기록", [], "", ["G0"], [SOURCE], ["다음 세트에서도 같은 크기로 모이는가"], "")
        return Stage1Result(stats.shape, stats.shape_ko, stats.sector, CLOCK_LABELS_KO[stats.sector], handedness,
                            [good], None, notes, comp)
    top = max(scores.values())
    cands = []
    for cid, v in sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]:
        c = kb["causes"][cid]
        sig = list(c.get("signals") or [])
        cands.append(Candidate(
            cause_id=cid, cause_ko=c["name"], score=round(min(0.99, v / top * min(1.0, top)), 3),
            guidance_ko=c["desc"], drill_ko=c.get("drill", ""), posture_signals=sig,
            observable_note="" if sig else "자세 영상으로 확인하기 어려움 (현장 확인 필요)",
            rule_ids=[cid] + sorted({w for w in why[cid] if w.startswith(("diag", "profile"))}),
            sources=[SOURCE], checklist=list(c.get("check") or []), axis=c["axis"]))
    return Stage1Result(stats.shape, stats.shape_ko, stats.sector, CLOCK_LABELS_KO[stats.sector], handedness,
                        cands, zero, notes, {k: round(v, 2) for k, v in comp.items()})
