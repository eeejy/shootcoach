"""1단계 진단: 교관 지식베이스(다중 변수 가중합) 동작 확인."""
import numpy as np
import pytest

from shootcoach.diagnosis.rules import load_causes
from shootcoach.diagnosis.stage1 import ShooterProfile, diagnose_stage1, load_kb, standalone_stage1
from shootcoach.pipeline import default_spec
from shootcoach.target.detect import Hole
from shootcoach.target.scoring import group_stats, score_hole

SPEC = default_spec()
S = SPEC.ring_step_mm


def holes_for(offset, sigma, n=10):
    """Deterministic, evenly spread group; offset/sigma in ring widths."""
    k = np.arange(n) + 0.5
    a = k * np.pi * (3 - np.sqrt(5))
    u = np.c_[np.sqrt(k / n) * np.cos(a), np.sqrt(k / n) * np.sin(a)]
    u = (u - u.mean(0)) / u.std(0)
    pts = np.asarray(offset) * S + u * np.asarray(sigma) * S
    cx, cy = SPEC.center_mm
    return [Hole(cx + x, cy - y, SPEC.bullet_diameter_mm / 2) for x, y in pts]


def top(offset, sigma, k=3, **kw):
    res = diagnose_stage1(group_stats(holes_for(offset, sigma), SPEC), SPEC, **kw)
    return [c.cause_id for c in res.candidates[:k]], res


def test_kb_is_consistent():
    kb = load_kb()
    ids = set(kb["causes"])
    for d in kb["diagonals"].values():
        assert set(d["boost"]) <= ids
    assert load_causes().keys() == ids
    for c in kb["causes"].values():
        assert c["check"], "every cause needs an instructor checklist"


def test_score_edge_rule():
    assert score_hole(0.0, SPEC) == 10
    assert score_hole(SPEC.ring10_radius_mm + SPEC.bullet_diameter_mm / 2 - 0.1, SPEC) == 10
    assert score_hole(400.0, SPEC) == 0


def test_low_left_combines_vertical_and_horizontal():
    ids, res = top((-1.6, -1.6), (1.2, 1.2), k=4)
    assert {"L1", "L2"} <= set(ids)                 # 하방: 반동 예측 / 급작 격발
    assert "LB4" in ids                             # 좌측: 구조적 횡력 (복합 벡터)
    assert res.zero_adjust is None


def test_high_right_combination():
    ids, _ = top((1.6, 1.6), (1.2, 1.2))
    assert set(ids[:2]) == {"H1", "RB1"}


def test_pure_low_and_pure_left_stay_on_their_axis():
    ids, _ = top((0, -1.8), (1.2, 1.2), k=4)
    assert all(i.startswith("L") and not i.startswith("LB") for i in ids)
    ids, _ = top((-1.8, 0), (1.2, 1.2), k=4)
    assert all(i.startswith("LB") for i in ids)


def test_tight_offset_checks_zero_first():
    ids, res = top((1.5, 1.2), (0.3, 0.3))
    assert ids[0] == "Z1" and res.zero_adjust is not None
    assert res.zero_adjust.move_right_mm < 0 and res.zero_adjust.move_up_mm < 0


def test_zero_confirmed_demotes_zero():
    ids, _ = top((1.5, 1.2), (0.3, 0.3), profile=ShooterProfile(zero_confirmed="yes"))
    assert ids[0] != "Z1"


def test_scatter_and_good_group():
    ids, _ = top((0, 0), (2.0, 2.0))
    assert set(ids) == {"S1", "S2", "S3"}
    ids, _ = top((0, 0), (0.3, 0.3))
    assert ids == ["G0"]


def test_profile_short_finger_raises_lateral_force():
    ids, _ = top((-1.6, -1.6), (1.2, 1.2), profile=ShooterProfile(finger_length="short"))
    assert ids[0] == "LB4"


def test_left_handed_mirror():
    ids, _ = top((-1.6, -1.6), (1.2, 1.2), k=3, handedness="left")
    assert "RB2" in ids and "L3" in ids              # 왼손잡이의 좌하 = 오른손잡이 우하 조합


def test_insufficient_shots():
    res = diagnose_stage1(group_stats(holes_for((0, 0), (0.3, 0.3), n=2), SPEC), SPEC)
    assert res.shape == "insufficient" and not res.candidates


@pytest.mark.parametrize("cid", ["L1", "H1", "S1", "Z1"])
def test_candidates_carry_checklist(cid):
    offs = {"L1": ((0, -1.8), (1.2, 1.2)), "H1": ((0, 1.8), (1.2, 1.2)), "S1": ((0, 0), (2, 2)), "Z1": ((1.5, 1.2), (0.3, 0.3))}
    _, res = top(*offs[cid], k=5)
    c = next(c for c in res.candidates if c.cause_id == cid)
    assert c.checklist and c.guidance_ko


def test_standalone_stage1_needs_no_target():
    """자세 영상만 있을 때: 표적지 증거 없이도 자세로 확인 가능한 원인 후보를 낸다."""
    res = standalone_stage1()
    assert res.candidates
    assert all(c.posture_signals for c in res.candidates)     # signals 없는 원인(LB1/LB4/RB1/Z1 등)은 제외
    assert not any(c.cause_id == "Z1" for c in res.candidates)  # 영점 오류는 자세로 확인할 수 없다
    assert "표적지" in " ".join(res.notes)


def test_standalone_stage1_mirrors_handedness():
    left = standalone_stage1(handedness="left")
    right = standalone_stage1(handedness="right")
    assert {c.cause_id for c in left.candidates} == {c.cause_id for c in right.candidates}
