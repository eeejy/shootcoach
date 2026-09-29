import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.pipeline import analyze_target, default_spec
from shootcoach.synth import ShotGroupSpec, draw_holes, make_synth_target, sample_group, simulate_photo
from shootcoach.target.detect import ClassicHoleDetector, Hole
from shootcoach.target.locate import TargetNotFound, locate_target
from shootcoach.target.scoring import clock_of, sector_of
from shootcoach.target.sequence import new_holes
from shootcoach.target.template import render_target

SPEC = default_spec()
A4 = load_target_spec("configs/target_a4.yaml")


def test_locate_without_markers_maps_holes_to_mm():
    import cv2

    rng = np.random.default_rng(3)
    t = make_synth_target(SPEC, rng)
    photo, H = simulate_photo(t.rect, rng, max_tilt=0.08)
    f = locate_target(photo, SPEC)
    ppm = SPEC.px_per_mm
    ph = cv2.perspectiveTransform((t.holes_mm * ppm).reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    back = f.paper_mm(ph, SPEC)
    near = np.hypot(*(t.holes_mm - np.asarray(SPEC.center_mm)).T) < SPEC.ring_radius_mm(4)
    err = np.hypot(*(back - t.holes_mm).T)[near]
    assert np.median(err) < 5.0                    # mm (ring width 25 mm): affine approximation of a tilted photo


def test_locate_ignores_shots_on_disc_edge():
    """Dark holes just outside ring 7 used to merge with the black disc and drag the fit towards the group."""
    rng = np.random.default_rng(11)
    c = np.asarray(SPEC.center_mm)
    r7 = SPEC.ring_radius_mm(SPEC.black_from_ring)
    ang = np.deg2rad(np.linspace(20, 80, 8))
    holes = c + np.column_stack([np.cos(ang), -np.sin(ang)]) * (r7 + rng.uniform(-6, 8, 8))[:, None]
    rect = draw_holes(render_target(SPEC), holes * SPEC.px_per_mm, np.full(8, 4.5 * SPEC.px_per_mm), rng, backer="dark")
    f = locate_target(rect, SPEC)
    centre_err_mm = np.hypot(*(np.asarray(f.center_px) / SPEC.px_per_mm - c))
    assert centre_err_mm < 2.0


def test_locate_fails_without_target():
    try:
        locate_target(np.full((800, 600, 3), 255, np.uint8), SPEC)
    except TargetNotFound as e:
        assert "검은 원" in str(e)
    else:
        raise AssertionError("expected TargetNotFound")


def test_pipeline_runs_on_demo_and_respects_override():
    r = analyze_target("samples/demo_low_left.jpg", detector=ClassicHoleDetector())
    assert r.report["frame"]["fill"] > 0.8
    holes = [Hole(SPEC.center_mm[0], SPEC.center_mm[1], 4.5)] * 3
    r2 = analyze_target("samples/demo_low_left.jpg", holes_override=holes)
    assert r2.report["group"]["n"] == 3 and r2.report["group"]["total_score"] == 30


def test_classic_detector_on_clean_rectified_target():
    rng = np.random.default_rng(5)
    holes = sample_group(A4, ShotGroupSpec(6, (15, -10), (12, 12)), rng)
    rect = draw_holes(render_target(A4), holes * A4.px_per_mm, np.full(6, 4.5 * A4.px_per_mm), rng, backer="cardboard")
    found = ClassicHoleDetector().detect(rect, A4)
    P = np.array([[h.x_mm, h.y_mm] for h in found])
    hit = sum(1 for g in holes if np.min(np.hypot(*(P - g).T)) < 4.5)
    assert hit >= 5


def test_clock_and_sector():
    assert sector_of(np.array([0, 10])) == 12
    assert sector_of(np.array([10, 0])) == 3
    assert sector_of(np.array([-10, -10])) == 7.5
    assert abs(clock_of(np.array([0, -10])) - 6) < 1e-6


def test_sequence_new_holes_only():
    prev = [Hole(100, 100, 4.5), Hole(120, 110, 4.5)]
    curr = prev + [Hole(104, 101, 4.5, 0.9)]
    new = new_holes(prev, curr, match_mm=2.5)
    assert len(new) == 1 and new[0].x_mm == 104
