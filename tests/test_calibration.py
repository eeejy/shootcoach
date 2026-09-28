import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.diagnosis.calibration import ShooterProfile, apply_profile
from shootcoach.diagnosis.stage1 import diagnose_stage1
from shootcoach.target.detect import Hole
from shootcoach.target.scoring import group_stats

SPEC = load_target_spec()


def stats_at(offset, sigma=(13, 13), n=10):
    k = np.arange(n) + 0.5
    a = k * np.pi * (3 - np.sqrt(5))
    u = np.c_[np.sqrt(k / n) * np.cos(a), np.sqrt(k / n) * np.sin(a)]
    u = (u - u.mean(0)) / u.std(0)
    pts = np.asarray(offset) + u * np.asarray(sigma)
    return group_stats([Hole(SPEC.center_mm[0] + x, SPEC.center_mm[1] - y, 4.5) for x, y in pts], SPEC)


def test_personal_jerk_direction_overrides_chart(tmp_path):
    # This shooter's deliberate-jerk sessions land at 6 o'clock, not the chart's 7:30.
    p = ShooterProfile("사수A")
    for _ in range(3):
        p.add_session("jerking", stats_at((0, -24)))
    p.save(tmp_path / "a.json")
    p = ShooterProfile.load(tmp_path / "a.json")
    st = stats_at((0, -24))
    base = diagnose_stage1(st, SPEC)
    assert "jerking" not in [c.cause_id for c in base.candidates]       # chart alone misses it
    cal = apply_profile(diagnose_stage1(st, SPEC), st, p)
    assert "jerking" in [c.cause_id for c in cal.candidates]
    assert any("CAL" in c.sources for c in cal.candidates)


def test_boost_existing_candidate():
    p = ShooterProfile("사수B")
    p.add_session("canting", stats_at((-22, -22)))
    st = stats_at((-22, -22))
    before = {c.cause_id: c.score for c in diagnose_stage1(st, SPEC).candidates}
    after = {c.cause_id: c.score for c in apply_profile(diagnose_stage1(st, SPEC), st, p).candidates}
    assert after["canting"] > before["canting"]
