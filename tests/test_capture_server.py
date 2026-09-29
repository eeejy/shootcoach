from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.capture_server import app

client = TestClient(app)
DEMO = Path(__file__).resolve().parents[1] / "samples" / "demo_low_left.jpg"


def _jpg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_index_serves_page():
    r = client.get("/")
    assert r.status_code == 200 and "표적" in r.text


def test_frame_finds_target_without_markers():
    img = cv2.resize(cv2.imread(str(DEMO)), None, fx=0.5, fy=0.5)     # phone preview size
    m = client.post("/api/frame", files={"frame": ("f.jpg", _jpg(img), "image/jpeg")}).json()
    assert m["found"] and m["ok"] and len(m["ellipse"]) == 5


def test_frame_rejects_blank_view():
    blank = np.full((600, 400, 3), 200, np.uint8)
    m = client.post("/api/frame", files={"frame": ("f.jpg", _jpg(blank), "image/jpeg")}).json()
    assert not m["found"] and not m["ok"]


def test_frame_asks_to_come_closer_when_target_is_tiny():
    img = cv2.imread(str(DEMO))
    small = cv2.resize(img, None, fx=0.2, fy=0.2)
    canvas = np.full((img.shape[0], img.shape[1], 3), 90, np.uint8)
    canvas[: small.shape[0], : small.shape[1]] = small
    m = client.post("/api/frame", files={"frame": ("f.jpg", _jpg(canvas), "image/jpeg")}).json()
    assert m["found"] and not m["ok"] and "가까이" in m["message"]


def test_analyze_returns_diagnosis():
    r = client.post("/api/analyze", files={"photo": ("p.jpg", DEMO.read_bytes(), "image/jpeg")},
                    data={"handedness": "right", "distance_m": "10", "expected": "10"})
    j = r.json()
    assert r.status_code == 200, j
    assert j["group"]["n"] >= 3 and j["stage1"]["candidates"]
    assert j["overlay_jpg"]


def test_analyze_rejects_blank_photo():
    blank = np.full((600, 400, 3), 200, np.uint8)
    r = client.post("/api/analyze", files={"photo": ("p.jpg", _jpg(blank), "image/jpeg")})
    assert r.status_code == 422 and "검은 원" in r.json()["error"]


def test_token_required_when_set(monkeypatch):
    import app.capture_server as cs

    monkeypatch.setattr(cs, "TOKEN", "secret123")
    assert client.get("/").status_code == 403
    assert client.get("/?t=secret123").status_code == 200
    img = cv2.resize(cv2.imread(str(DEMO)), None, fx=0.4, fy=0.4)
    files = {"frame": ("f.jpg", _jpg(img), "image/jpeg")}
    assert client.post("/api/frame", files=files).status_code == 403
    assert client.post("/api/frame", files=files, headers={"X-Token": "secret123"}).status_code == 200


def test_upload_size_limit(monkeypatch):
    import app.capture_server as cs

    monkeypatch.setattr(cs, "MAX_UPLOAD_BYTES", 1000)
    r = client.post("/api/frame", files={"frame": ("f.jpg", b"x" * 5000, "image/jpeg")})
    assert r.status_code == 400 and "너무 큽니다" in r.json()["error"]
