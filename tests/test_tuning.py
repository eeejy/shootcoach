import numpy as np

from shootcoach.diagnosis.tuning import best_threshold, tune, video_feature_means
from shootcoach.pose.simulate import Faults, simulate_side_view

SHOTS = [2.5, 5.0, 7.5]


def test_best_threshold_directions():
    v = np.array([-5.0, -4.0, -3.0, -0.5, 0.0, 0.5])
    y = np.array([1, 1, 1, 0, 0, 0])
    thr, acc = best_threshold(v, y, "lt")
    assert acc == 1.0 and -3.0 < thr < -0.5
    thr, acc = best_threshold(-v, y, "gt")
    assert acc == 1.0 and 0.5 < thr < 3.0


def test_tune_recovers_separating_threshold_for_dip():
    rows = []
    for i, dip in enumerate([0, 0.3, 0.6, 0.8, 3, 4, 5, 6]):
        seq = simulate_side_view(SHOTS, faults=Faults(dip_deg=dip), seed=i)
        rows.append({"features": video_feature_means(seq, SHOTS), "labels": {"muzzle_dip_preshot": int(dip >= 2)}})
    res = {r.signal_id: r for r in tune(rows)}
    r = res["muzzle_dip_preshot"]
    assert r.suggested is not None and r.acc_suggested == 1.0
    assert -3.0 < r.suggested < -0.8
    assert res["breath_sway"].suggested is None          # no labels → untouched
