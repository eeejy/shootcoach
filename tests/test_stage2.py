import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.diagnosis.stage1 import diagnose_stage1
from shootcoach.diagnosis.stage2 import diagnose_stage2, evaluate_signals
from shootcoach.pose.features import arm_series, detect_view
from shootcoach.pose.shots import motion_shot_times
from shootcoach.pose.simulate import Faults, simulate_side_view
from shootcoach.target.detect import Hole
from shootcoach.target.scoring import group_stats

SPEC = load_target_spec("configs/target_a4.yaml")
SHOTS = (2.5, 5.0, 7.5)


def stage1_for(offset, sigma=(13, 13), n=10):
    k = np.arange(n) + 0.5
    a = k * np.pi * (3 - np.sqrt(5))
    u = np.c_[np.sqrt(k / n) * np.cos(a), np.sqrt(k / n) * np.sin(a)]
    u = (u - u.mean(0)) / u.std(0)
    pts = np.asarray(offset) + u * np.asarray(sigma)
    cx, cy = SPEC.center_mm
    st = group_stats([Hole(cx + x, cy - y, 4.5) for x, y in pts], SPEC)
    return diagnose_stage1(st, SPEC)


def test_view_and_recoil_shot_detection():
    seq = simulate_side_view(SHOTS)
    assert detect_view(seq) == "side"
    _, pitch, _ = arm_series(seq)
    found = motion_shot_times(pitch, seq.fps)
    assert len(found) == 3
    assert np.allclose(found, SHOTS, atol=0.05)


def test_clean_shooter_has_no_fault_signals():
    seq = simulate_side_view(SHOTS, faults=Faults())
    sig = evaluate_signals(seq, list(SHOTS), "right", "side")
    present = [k for k, s in sig.items() if s.present]
    assert present == [], present


def test_dip_confirms_low_cause_for_low_left_group():
    s1 = stage1_for((-22, -22))                      # 좌하 → 반동 예측/급작 격발 + 좌측 횡력
    assert {"L1", "L2"} <= {c.cause_id for c in s1.candidates}
    seq = simulate_side_view(SHOTS, faults=Faults(dip_deg=5))
    s2 = diagnose_stage2(s1, seq, list(SHOTS))
    assert s2.final_cause_id in {"L1", "L2"}
    lb4 = next(v for v in s2.verdicts if v.cause_id == "LB4")
    assert lb4.status == "unobservable"               # 손가락 횡력은 영상으로 못 봄


def test_muzzle_up_confirms_high_cause():
    s1 = stage1_for((22, 22))
    seq = simulate_side_view(SHOTS, faults=Faults(heel_deg=5, shrug=0.03))
    s2 = diagnose_stage2(s1, seq)                   # shot times from recoil
    assert s2.final_cause_id in {"H1", "H2", "H4"}


def test_no_supporting_signal_means_hold():
    s1 = stage1_for((-22, -22))
    seq = simulate_side_view(SHOTS, faults=Faults())
    s2 = diagnose_stage2(s1, seq, list(SHOTS))
    assert s2.final_cause_id is None and s2.final_ko.startswith("보류")
    assert any(v.status == "ruled_out" for v in s2.verdicts)


def test_early_drop_confirms_recoil_management_for_high_group():
    s1 = stage1_for((0, 24))
    assert "H4" in {c.cause_id for c in s1.candidates}
    seq = simulate_side_view(SHOTS, faults=Faults(early_drop=0.15))
    s2 = diagnose_stage2(s1, seq, list(SHOTS))
    assert s2.final_cause_id == "H4"


def test_breathing_sway_for_vertical_string():
    s1 = stage1_for((0, 0), sigma=(2.5, 16))
    assert s1.shape == "vertical_string"
    seq = simulate_side_view(SHOTS, faults=Faults(breath_amp=0.04))
    s2 = diagnose_stage2(s1, seq, list(SHOTS))
    assert s2.final_cause_id == "S1"
