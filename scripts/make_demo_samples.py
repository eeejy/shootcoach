"""Demo photos for each diagnosis scenario (synthetic, with ground truth)."""
import json
from pathlib import Path

import cv2
import numpy as np

from shootcoach.config import load_target_spec
from shootcoach.synth import ShotGroupSpec, draw_holes, sample_group, simulate_photo
from shootcoach.target.template import render_target

# 해경 원형 표적(링 간격 25mm) 기준 mm. 탄공 지름 9mm라 10발이 한 덩어리로 겹치지 않을 만큼은 퍼뜨린다.
SCENARIOS = {  # 10발 (완사 기준)
    "good": ShotGroupSpec(10, (3, -3), (15, 15)),
    "zero_offset": ShotGroupSpec(10, (75, 56), (15, 15)),
    "low_left": ShotGroupSpec(10, (-62, -69), (28, 28)),
    "high_right": ShotGroupSpec(10, (62, 69), (28, 28)),
    "vertical": ShotGroupSpec(10, (0, 0), (9, 60)),
    "scattered": ShotGroupSpec(10, (0, 0), (60, 60)),
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
        photo, _ = simulate_photo(rect, rng, max_tilt=0.04)   # 앱 안내대로 거의 정면 촬영
        cv2.imwrite(str(out / f"demo_{name}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        truth[name] = {"holes_mm": np.round(holes, 2).tolist(), "offset_mm": g.offset_mm, "sigma_mm": g.sigma_mm}
    (out / "demo_truth.json").write_text(json.dumps(truth, indent=1))
    print("wrote", len(SCENARIOS), "demo photos")


if __name__ == "__main__":
    main()
