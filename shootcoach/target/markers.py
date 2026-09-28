"""ArUco-based perspective correction: photo → top-down target in paper mm."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shootcoach.config import TargetSpec


@dataclass
class Rectification:
    image: np.ndarray            # rectified BGR image (spec.canvas_px)
    H_img_to_rect: np.ndarray    # 3x3 homography, photo px → rectified px
    marker_ids: list[int]
    reprojection_mm: float       # mean marker-corner error after fit


class MarkerError(RuntimeError):
    pass


def detect_markers(image: np.ndarray, spec: TargetSpec):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.aruco_dict))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if int(i) in spec.markers:
                found[int(i)] = c.reshape(4, 2)
    return found


def rectify(image: np.ndarray, spec: TargetSpec, min_markers: int = 3) -> Rectification:
    """Warp a photo of the target to a canonical top-down view.

    Uses every corner of every visible marker (4 per marker), so 3 markers are
    enough and a partially covered target still works.
    """
    found = detect_markers(image, spec)
    if len(found) < min_markers:
        raise MarkerError(
            f"마커를 {len(found)}개만 찾았습니다(필요 {min_markers}개). 네 모서리 마커가 모두 보이게 다시 촬영하세요."
        )
    ppm = spec.px_per_mm
    src, dst = [], []
    for mid, img_corners in found.items():
        src.extend(img_corners.tolist())
        dst.extend([(x * ppm, y * ppm) for x, y in spec.marker_corners_mm(mid)])
    src = np.asarray(src, np.float32)
    dst = np.asarray(dst, np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        raise MarkerError("정면 보정(호모그래피) 계산에 실패했습니다.")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err_mm = float(np.linalg.norm(proj - dst, axis=1).mean() / ppm)
    rect = cv2.warpPerspective(image, H, spec.canvas_px, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return Rectification(rect, H, sorted(found), err_mm)
