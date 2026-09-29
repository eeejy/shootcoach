"""Rule files in rules/ (edited by instructors, not in code).

- rules/instructor_kb.yaml : 원인 지식베이스 (교관 자료)
- rules/posture_signals.csv : 자세 영상 신호 정의·임계값
"""
from __future__ import annotations

from dataclasses import dataclass

from shootcoach.config import REPO_ROOT

RULES_DIR = REPO_ROOT / "rules"


@dataclass(frozen=True)
class Cause:
    cause_id: str
    cause_ko: str
    guidance_ko: str
    drill_ko: str
    posture_signals: tuple[str, ...]
    checklist: tuple[str, ...]
    axis: str


def load_causes() -> dict[str, Cause]:
    from shootcoach.diagnosis.stage1 import load_kb

    return {cid: Cause(cid, c["name"], c["desc"], c.get("drill", ""), tuple(c.get("signals") or ()),
                       tuple(c.get("check") or ()), c["axis"])
            for cid, c in load_kb()["causes"].items()}
