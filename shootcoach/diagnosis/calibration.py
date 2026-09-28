"""사수별 캘리브레이션: 파이 차트를 '그 사수 본인' 기준으로 다시 만든다.

진단 세션에서 사수가 일부러 특정 오류(예: 저킹)를 내며 5발씩 쏘면, 그 오류가 이 사수에게서
어느 방향으로 나타나는지 기록한다. 이후 진단에서 같은 방향으로 탄착군이 형성되면 그 원인의
가중치를 올린다. 프로필은 개인 데이터이므로 이 PC의 profiles/ 에만 저장한다(저장소 제외).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shootcoach.diagnosis.rules import load_causes
from shootcoach.diagnosis.stage1 import Candidate, Stage1Result
from shootcoach.target.scoring import GroupStats


@dataclass
class ShooterProfile:
    shooter_id: str
    handedness: str = "right"
    fault_sectors: dict[str, dict[str, int]] = field(default_factory=dict)   # cause_id → {sector: count}

    def add_session(self, cause_id: str, stats: GroupStats) -> None:
        if stats.shape in ("insufficient", "tight_centered"):
            return                        # 오류가 드러나지 않은 세션은 기록하지 않음
        d = self.fault_sectors.setdefault(cause_id, {})
        key = str(stats.sector)
        d[key] = d.get(key, 0) + 1

    def match(self, cause_id: str, sector: float) -> float:
        d = self.fault_sectors.get(cause_id)
        if not d:
            return 0.0
        return d.get(str(sector), 0) / sum(d.values())

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ShooterProfile":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def apply_profile(res: Stage1Result, stats: GroupStats, profile: ShooterProfile, boost: float = 0.3) -> Stage1Result:
    """Re-rank candidates with the shooter's own fault map; add personal causes the chart missed."""
    causes = load_causes()
    by_id = {c.cause_id: c for c in res.candidates}
    for cid in profile.fault_sectors:
        m = profile.match(cid, stats.sector)
        if m <= 0:
            continue
        if cid in by_id:
            by_id[cid].score = round(min(1.0, by_id[cid].score + boost * m), 3)
            by_id[cid].sources = sorted(set(by_id[cid].sources) | {"CAL"})
        elif cid in causes:
            c = causes[cid]
            by_id[cid] = Candidate(cid, c.cause_ko, round(0.5 * m, 3), c.guidance_ko, c.drill_ko,
                                   list(c.posture_signals), c.observable_note, ["CAL"], ["CAL"])
    res.candidates = sorted(by_id.values(), key=lambda c: -c.score)
    res.notes.append(f"사수 '{profile.shooter_id}' 캘리브레이션 반영 (CAL = 본인 진단 세션 기준)")
    return res
