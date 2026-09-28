import numpy as np
import pytest

from shootcoach.config import load_target_spec
from shootcoach.diagnosis.rules import load_causes, load_rules
from shootcoach.diagnosis.stage1 import diagnose_stage1
from shootcoach.target.detect import Hole
from shootcoach.target.scoring import group_stats, score_hole

SPEC = load_target_spec()


def holes_for(offset, sigma, n=8, seed=0):
    """Deterministic, evenly spread group (sunflower pattern scaled to per-axis std = sigma)."""
    k = np.arange(n) + 0.5
    r = np.sqrt(k / n)
    a = k * np.pi * (3 - np.sqrt(5)) + seed
    u = np.c_[r * np.cos(a), r * np.sin(a)]
    u = (u - u.mean(0)) / u.std(0)
    pts = np.asarray(offset) + u * np.asarray(sigma)
    cx, cy = SPEC.center_mm
    return [Hole(cx + x, cy - y, SPEC.bullet_diameter_mm / 2) for x, y in pts]


def test_rules_reference_known_causes():
    rules, causes = load_rules(), load_causes()
    assert rules and {r.cause_id for r in rules} <= set(causes)


def test_score_edge_rule():
    # centre shot = 10; hole whose edge touches the 10-ring line still scores 10
    assert score_hole(0.0, SPEC) == 10
    assert score_hole(SPEC.ring10_radius_mm + SPEC.bullet_diameter_mm / 2 - 0.1, SPEC) == 10
    assert score_hole(200.0, SPEC) == 0


@pytest.mark.parametrize("offset,sigma,shape,top", [
    ((0, 0), (3, 3), "tight_centered", "good_group"),
    ((25, 20), (3, 3), "tight_offset", "sight_zero"),
    ((-22, -22), (13, 13), "scattered_offset", "jerking"),        # 7시 반, 오른손잡이
    ((22, 22), (13, 13), "scattered_offset", "heeling"),          # 1시 반
    ((0, 0), (4, 22), "vertical_string", "breathing"),
    ((0, 0), (24, 4), "horizontal_string", "natural_point_of_aim"),
    ((0, 0), (2.5, 11), "vertical_string", "breathing"),          # narrow line, small mean radius
    ((0, 0), (20, 20), "scattered_centered", "fundamentals"),
])
def test_shapes_and_top_cause(offset, sigma, shape, top):
    st = group_stats(holes_for(offset, sigma, n=10, seed=3), SPEC)
    assert st.shape == shape, st
    res = diagnose_stage1(st, SPEC)
    assert res.candidates[0].cause_id == top


def test_left_handed_mirror():
    st = group_stats(holes_for((22, -22), (13, 13), n=10, seed=4), SPEC)   # 4시 반
    assert st.sector == 4.5
    right = diagnose_stage1(st, SPEC, "right")
    left = diagnose_stage1(st, SPEC, "left")
    assert right.candidates[0].cause_id == "snatching"
    assert left.candidates[0].cause_id == "jerking"   # 왼손잡이의 4시 반 = 오른손잡이 7시 반


def test_zero_adjust_clicks():
    st = group_stats(holes_for((25, 20), (3, 3), seed=5), SPEC)
    res = diagnose_stage1(st, SPEC, distance_m=15, click_mm_per_10m=5)
    z = res.zero_adjust
    assert z and z.move_right_mm < 0 and z.move_up_mm < 0
    assert z.clicks_windage < 0 and z.clicks_elevation < 0


def test_insufficient_shots():
    st = group_stats(holes_for((0, 0), (3, 3), n=2), SPEC)
    res = diagnose_stage1(st, SPEC)
    assert st.shape == "insufficient" and not res.candidates
