import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.labeling import dataset_counts, load_sample, prelabel, save_sample, split_for
from shootcoach.synth import make_synth_target, simulate_photo
from shootcoach.target.detect import ClassicHoleDetector, Hole

SPEC = load_target_spec()


def test_split_is_deterministic():
    assert split_for("IMG_0001") == split_for("IMG_0001")
    splits = {split_for(f"img{i}") for i in range(50)}
    assert splits == {"train", "val"}


def test_save_and_reload_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    t = make_synth_target(SPEC, rng)
    holes = [Hole(x, y, r) for (x, y), r in zip(t.holes_mm, t.radii_mm)]
    save_sample(t.rect, holes, SPEC, tmp_path, "sample1", split="train")
    back = load_sample(tmp_path, "sample1", SPEC)
    assert back is not None and len(back) == len(holes)
    err = max(min(np.hypot(b.x_mm - h.x_mm, b.y_mm - h.y_mm) for b in back) for h in holes)
    assert err < 0.3
    assert dataset_counts(tmp_path)["train"] == {"images": 1, "holes": len(holes)}
    assert (tmp_path / "data.yaml").exists()


def test_prelabel_from_photo():
    rng = np.random.default_rng(2)
    t = make_synth_target(SPEC, rng)
    photo, _ = simulate_photo(t.rect, rng, max_tilt=0.1)
    rect, holes = prelabel(photo, SPEC, ClassicHoleDetector())
    assert rect.shape[:2] == (SPEC.canvas_px[1], SPEC.canvas_px[0])
    assert len(holes) > 0
