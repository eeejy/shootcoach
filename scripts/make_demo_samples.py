"""Demo photos for each diagnosis scenario (synthetic, with ground truth)."""
import json
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.synth import ShotGroupSpec, draw_holes, sample_group, simulate_photo
from shootcoach.target.template import render_target

SCENARIOS = {
    "good": ShotGroupSpec(8, (1, -1), (3.5, 3.5)),
    "zero_offset": ShotGroupSpec(8, (24, 18), (3.5, 3.5)),
    "jerking_low_left": ShotGroupSpec(8, (-20, -22), (10, 10)),
    "heeling_high_right": ShotGroupSpec(8, (20, 22), (10, 10)),
    "breathing_vertical": ShotGroupSpec(8, (0, 0), (3, 20)),
    "scattered": ShotGroupSpec(10, (0, 0), (20, 20)),
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
