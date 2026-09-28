import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.synth import ShotGroupSpec, draw_holes, make_synth_target, sample_group, simulate_photo
from shootcoach.target.detect import ClassicHoleDetector, Hole
from shootcoach.target.markers import MarkerError, rectify
from shootcoach.target.scoring import clock_of, sector_of
from shootcoach.target.sequence import new_holes
from shootcoach.target.template import render_target

SPEC = load_target_spec()


def test_rectify_roundtrip_submillimetre():
    import cv2

    rng = np.random.default_rng(3)
    t = make_synth_target(SPEC, rng)
    photo, H = simulate_photo(t.rect, rng, max_tilt=0.2)
    r = rectify(photo, SPEC)
    assert len(r.marker_ids) >= 3 and r.reprojection_mm < 0.5
    ppm = SPEC.px_per_mm
    p = cv2.perspectiveTransform((t.holes_mm * ppm).reshape(-1, 1, 2).astype(np.float32), H)
    back = cv2.perspectiveTransform(p, r.H_img_to_rect).reshape(-1, 2) / ppm
    assert np.abs(back - t.holes_mm).max() < 1.0


def test_rectify_fails_without_markers():
    img = np.full((800, 600, 3), 255, np.uint8)
    try:
        rectify(img, SPEC)
    except MarkerError as e:
        assert "마커" in str(e)
    else:
        raise AssertionError("expected MarkerError")


def test_classic_detector_on_clean_rectified_target():
    rng = np.random.default_rng(5)
    holes = sample_group(SPEC, ShotGroupSpec(6, (15, -10), (12, 12)), rng)
    rect = draw_holes(render_target(SPEC), holes * SPEC.px_per_mm, np.full(6, 4.5 * SPEC.px_per_mm), rng, backer="cardboard")
    found = ClassicHoleDetector().detect(rect, SPEC)
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
    curr = prev + [Hole(104, 101, 4.5, 0.9)]          # overlapping new hole 4 mm away
    new = new_holes(prev, curr, match_mm=2.5)
    assert len(new) == 1 and new[0].x_mm == 104
