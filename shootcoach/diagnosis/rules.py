"""Load the rule tables in rules/*.csv (edited by the domain expert, not in code)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from shootcoach.config import REPO_ROOT

RULES_DIR = REPO_ROOT / "rules"


@dataclass(frozen=True)
class Cause:
    cause_id: str
    cause_ko: str
    guidance_ko: str
    drill_ko: str
    posture_signals: tuple[str, ...]
    observable_note: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    shape: str
    sector: str        # "*" or a clock sector like "7.5"
    cause_id: str
    prior: float
    source: str


def _read(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=4)
def load_causes(rules_dir: str | None = None) -> dict[str, Cause]:
    d = Path(rules_dir) if rules_dir else RULES_DIR
    out = {}
    for r in _read(d / "causes.csv"):
        sig = tuple(s for s in (r.get("posture_signals") or "").split(";") if s)
        out[r["cause_id"]] = Cause(r["cause_id"], r["cause_ko"], r["guidance_ko"], r["drill_ko"], sig,
                                   r.get("observable_note") or "")
    return out


@lru_cache(maxsize=4)
def load_rules(rules_dir: str | None = None) -> tuple[Rule, ...]:
    d = Path(rules_dir) if rules_dir else RULES_DIR
    rules = tuple(Rule(r["rule_id"], r["shape"], r["sector"], r["cause_id"], float(r["prior"]), r["source"])
                  for r in _read(d / "stage1_rules.csv"))
    causes = load_causes(rules_dir)
    missing = {r.cause_id for r in rules} - set(causes)
    if missing:
        raise ValueError(f"causes.csv에 없는 원인 ID: {sorted(missing)}")
    return rules
