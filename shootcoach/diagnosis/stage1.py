"""1단계: 표적지 탄착군만으로 원인 후보 + 교정 가이드."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from shootcoach.config import TargetSpec
from shootcoach.diagnosis.rules import load_causes, load_rules
from shootcoach.target.scoring import CLOCK_LABELS_KO, GroupStats

MIRROR = {12: 12, 1.5: 10.5, 3: 9, 4.5: 7.5, 6: 6, 7.5: 4.5, 9: 3, 10.5: 1.5}


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

    def as_dict(self):
        return asdict(self)


def _zero_adjust(stats: GroupStats, distance_m: float | None, click_mm_per_10m: float | None) -> ZeroAdjust:
    cx, cy = stats.center_mm
    right, up = -cx, -cy
    parts = []
    if abs(right) >= 1:
        parts.append(f"{'오른쪽' if right > 0 else '왼쪽'}으로 {abs(right):.0f}mm")
    if abs(up) >= 1:
        parts.append(f"{'위' if up > 0 else '아래'}로 {abs(up):.0f}mm")
    text = "탄착군을 " + ", ".join(parts) + " 옮기도록 조준기를 조정하세요." if parts else "조정 불필요."
    cw = ce = None
    if distance_m and click_mm_per_10m:
        per_click = click_mm_per_10m * distance_m / 10.0
        cw, ce = int(round(right / per_click)), int(round(up / per_click))
        text += (f" (1클릭 {per_click:.1f}mm @ {distance_m:g}m 기준: 좌우 {abs(cw)}클릭 {'R' if cw > 0 else 'L'}, "
                 f"상하 {abs(ce)}클릭 {'U' if ce > 0 else 'D'})")
    else:
        text += " 기종별 1클릭 값을 입력하면 클릭 수를 계산합니다. 고정식 조준기라면 정비관에게 영점 점검을 요청하세요."
    return ZeroAdjust(right, up, text, cw, ce)


def diagnose_stage1(stats: GroupStats, spec: TargetSpec, handedness: str = "right",
                    distance_m: float | None = None, click_mm_per_10m: float | None = None,
                    top_k: int = 4) -> Stage1Result:
    rules, causes = load_rules(), load_causes()
    # Rules are written for right-handers; mirror the sector for left-handers.
    sector = stats.sector if handedness == "right" else MIRROR[stats.sector]
    notes = []
    if handedness != "right":
        notes.append("왼손잡이: 오른손잡이 기준 규칙을 좌우 반전해 적용했습니다.")
    if stats.shape == "insufficient":
        return Stage1Result(stats.shape, stats.shape_ko, stats.sector, stats.sector_ko, handedness, [], None,
                            ["탄 수가 부족합니다. 한 발만 보고 원인을 판단하거나 조준기를 조정하지 마세요 (최소 3발, 권장 5발)."])
    agg: dict[str, Candidate] = {}
    for r in rules:
        if r.shape != stats.shape:
            continue
        if r.sector != "*" and float(r.sector) != float(sector):
            continue
        c = causes[r.cause_id]
        cand = agg.get(r.cause_id)
        if cand is None:
            agg[r.cause_id] = Candidate(c.cause_id, c.cause_ko, r.prior, c.guidance_ko, c.drill_ko,
                                        list(c.posture_signals), c.observable_note, [r.rule_id], r.source.split(";"))
        else:  # same cause from several rules → keep max prior, merge provenance
            cand.score = max(cand.score, r.prior)
            cand.rule_ids.append(r.rule_id)
            cand.sources = sorted(set(cand.sources) | set(r.source.split(";")))
    cands = sorted(agg.values(), key=lambda c: -c.score)[:top_k]
    zero = None
    if stats.shape == "tight_offset":
        zero = _zero_adjust(stats, distance_m, click_mm_per_10m)
    elif stats.shape == "scattered_offset":
        notes.append("탄착군이 흩어져 있을 때는 조준기를 조정하지 마세요. 먼저 동작 오류를 고쳐 탄착군을 모읍니다 (USAMU).")
    if stats.n_fliers:
        notes.append(f"탄착군에서 크게 벗어난 탄 {stats.n_fliers}발은 중심 계산에서 제외했습니다.")
    if stats.n < 5:
        notes.append("5발 미만이라 판정 신뢰도가 낮습니다.")
    return Stage1Result(stats.shape, stats.shape_ko, stats.sector, CLOCK_LABELS_KO[stats.sector], handedness,
                        cands, zero, notes)
