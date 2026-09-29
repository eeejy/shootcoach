from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.capture_server import app

client = TestClient(app)
DEMO = Path("samples/demo_jerking_low_left.jpg")


def _jpg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_index_serves_page():
    r = client.get("/")
    assert r.status_code == 200 and "표적지 촬영" in r.text


def test_markers_all_found_on_demo():
    img = cv2.resize(cv2.imread(str(DEMO)), None, fx=0.4, fy=0.4)     # phone preview size
    m = client.post("/api/markers", files={"frame": ("f.jpg", _jpg(img), "image/jpeg")}).json()
    assert m["ok"] and m["found"] == [0, 1, 2, 3]


def test_markers_reports_missing_corner():
    img = cv2.imread(str(DEMO))
    h, w = img.shape[:2]
    img[int(h * 0.62):, : int(w * 0.5)] = 90                               # cover bottom-left marker
    m = client.post("/api/markers", files={"frame": ("f.jpg", _jpg(img), "image/jpeg")}).json()
    assert not m["ok"] and "왼쪽 아래" in m["missing_ko"]


def test_analyze_returns_diagnosis():
    r = client.post("/api/analyze", files={"photo": ("p.jpg", DEMO.read_bytes(), "image/jpeg")},
                    data={"handedness": "right", "distance_m": "15"})
    j = r.json()
    assert r.status_code == 200, j
    assert j["group"]["n"] >= 6 and j["stage1"]["candidates"][0]["cause_id"] == "jerking"
    assert j["overlay_jpg"]


def test_analyze_rejects_blank_photo():
    blank = np.full((600, 400, 3), 200, np.uint8)
    r = client.post("/api/analyze", files={"photo": ("p.jpg", _jpg(blank), "image/jpeg")})
    assert r.status_code == 422 and "마커" in r.json()["error"]


def test_token_required_when_set(monkeypatch):
    import app.capture_server as cs

    monkeypatch.setattr(cs, "TOKEN", "secret123")
    assert client.get("/").status_code == 403
    assert client.get("/?t=secret123").status_code == 200
    img = cv2.resize(cv2.imread(str(DEMO)), None, fx=0.4, fy=0.4)
    files = {"frame": ("f.jpg", _jpg(img), "image/jpeg")}
    assert client.post("/api/markers", files=files).status_code == 403
    assert client.post("/api/markers", files=files, headers={"X-Token": "secret123"}).status_code == 200


def test_upload_size_limit(monkeypatch):
    import app.capture_server as cs

    monkeypatch.setattr(cs, "MAX_UPLOAD_BYTES", 1000)
    r = client.post("/api/markers", files={"frame": ("f.jpg", b"x" * 5000, "image/jpeg")})
    assert r.status_code == 400 and "너무 큽니다" in r.json()["error"]
