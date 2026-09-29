"""Demo photos for each diagnosis scenario (synthetic, with ground truth)."""
import json
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.synth import ShotGroupSpec, draw_holes, sample_group, simulate_photo
from shootcoach.target.template import render_target

S_ = 25.0 / 8.0   # scenarios were tuned in A4 ring widths (8 mm); KCG ring width is 25 mm
SCENARIOS = {  # 10발 (완사 기준)
    "good": ShotGroupSpec(10, (1 * S_, -1 * S_), (3.5 * S_, 3.5 * S_)),
    "zero_offset": ShotGroupSpec(10, (24 * S_, 18 * S_), (3.5 * S_, 3.5 * S_)),
    "low_left": ShotGroupSpec(10, (-20 * S_, -22 * S_), (10 * S_, 10 * S_)),
    "high_right": ShotGroupSpec(10, (20 * S_, 22 * S_), (10 * S_, 10 * S_)),
    "vertical": ShotGroupSpec(10, (0, 0), (3 * S_, 20 * S_)),
    "scattered": ShotGroupSpec(10, (0, 0), (20 * S_, 20 * S_)),
}


def main(out="samples"):
    spec = load_target_spec()
    out = Path(out)
    out.mkdir(exist_ok=True)
    truth = {}
    for i, (name, g) in enumerate(SCENARIOS.items()):
        rng = np.random.default_rng(100 + i)
        holes = sample_group(spec, g, rng)
        radii = np.full(len(holes), spec.bullet_diameter_mm / 2)
        rect = draw_holes(render_target(spec), holes * spec.px_per_mm, radii * spec.px_per_mm, rng, backer="cardboard")
        photo, _ = simulate_photo(rect, rng, max_tilt=0.15)
        cv2.imwrite(str(out / f"demo_{name}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        truth[name] = {"holes_mm": np.round(holes, 2).tolist(), "offset_mm": g.offset_mm, "sigma_mm": g.sigma_mm}
    (out / "demo_truth.json").write_text(json.dumps(truth, indent=1))
    print("wrote", len(SCENARIOS), "demo photos")


if __name__ == "__main__":
    main()
