"""모바일 촬영 가이드 서버 — 폰 브라우저에서 마커 4개가 잡힐 때만 셔터가 켜진다.

    python app/capture_server.py                # http://<PC IP>:8600  (사진 모드: 찍고 나서 마커 확인)
    python app/capture_server.py --https        # https://<PC IP>:8600 (실시간 모드: 카메라 화면에 마커 표시)

폰 브라우저는 HTTPS에서만 실시간 카메라를 허용한다. 인증서는 scripts/make_dev_cert.sh 로 만든다.
모든 처리는 이 PC에서만 한다.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import secrets

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shootcoach.explain.vlm import template_explanation  # noqa: E402
from shootcoach.pipeline import analyze_target, default_spec, get_photo_detector  # noqa: E402
from shootcoach.target.locate import TargetNotFound, locate_target  # noqa: E402

SPEC = default_spec()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# Access token: required by default when run from the command line (see main). Tests import the
# module with no token. The page receives it in the URL (?t=...) and sends it back as a header.
TOKEN: str | None = os.environ.get("SHOOTCOACH_TOKEN") or None
app = FastAPI(title="BullsAI 촬영 가이드")
_detector = None


def detector():
    global _detector
    if _detector is None:
        _detector = get_photo_detector()
    return _detector


def _check_token(t: str | None) -> bool:
    return TOKEN is None or (t is not None and secrets.compare_digest(t, TOKEN))


async def _read_limited(f: UploadFile) -> bytes:
    data = await f.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"파일이 너무 큽니다 (최대 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB).")
    return data


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return img


MIN_DISC_FRACTION = 0.07      # 검은 원 반지름이 화면 짧은 변의 7% 이상이어야 탄공이 보인다


def frame_status(img: np.ndarray) -> dict:
    """촬영 가능 여부: 표적의 검은 원이 화면 안에 충분히 크게 보이는가 (마커 없음)."""
    h, w = img.shape[:2]
    try:
        f = locate_target(img, SPEC)
    except TargetNotFound:
        return {"ok": False, "found": False, "message": "표적의 검은 원이 보이게 비춰 주세요"}
    (cx, cy), (a, b), ang = f.center_px, f.axes_px, f.angle_deg
    size = min(a, b) / min(h, w)
    ok, msg = True, "표적 확인 — 촬영하세요"
    if size < MIN_DISC_FRACTION:
        ok, msg = False, "조금 더 가까이 찍어 주세요"
    elif min(a, b) / max(a, b) < 0.8:
        ok, msg = False, "표적 정면에서 찍어 주세요 (너무 비스듬합니다)"
    return {"ok": ok, "found": True, "message": msg,
            "ellipse": [round(cx / w, 4), round(cy / h, 4), round(a / w, 4), round(b / h, 4), round(ang, 1)]}


@app.get("/")
def index(t: str | None = None):
    if not _check_token(t):
        return JSONResponse({"error": "접속 주소에 토큰(?t=...)이 필요합니다. 서버 실행 창에 표시된 주소로 접속하세요."}, status_code=403)
    return FileResponse(ROOT / "app" / "static" / "capture.html")


LOGO = ROOT / "app" / "static" / "bullsai_logo.png"


@app.get("/favicon.ico")
def favicon():
    return FileResponse(LOGO) if LOGO.exists() else Response(status_code=204)


@app.get("/logo.png")
def logo():
    # 앱 로고 원본을 그대로 전송 (가공 없음)
    return FileResponse(LOGO, media_type="image/png") if LOGO.exists() else Response(status_code=404)


@app.post("/api/frame")
async def api_frame(frame: UploadFile = File(...), x_token: str | None = Header(None)):
    if not _check_token(x_token):
        return JSONResponse({"error": "토큰이 없거나 틀립니다."}, status_code=403)
    try:
        img = _decode(await _read_limited(frame))
        if os.environ.get("SHOOTCOACH_DEBUG_FRAMES"):
            cv2.imwrite(os.environ["SHOOTCOACH_DEBUG_FRAMES"], img)
        return frame_status(img)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/analyze")
async def api_analyze(photo: UploadFile = File(...), handedness: str = Form("right"),
                      distance_m: float = Form(10.0), expected: int = Form(10), x_token: str | None = Header(None)):
    if not _check_token(x_token):
        return JSONResponse({"error": "토큰이 없거나 틀립니다."}, status_code=403)
    if handedness not in ("right", "left"):
        return JSONResponse({"error": "handedness는 right 또는 left"}, status_code=400)
    try:
        img = _decode(await _read_limited(photo))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    status = frame_status(img)
    if not status["found"]:
        return JSONResponse({"error": status["message"], "frame": status}, status_code=422)
    try:
        res = analyze_target(img, SPEC, handedness, distance_m, None, detector(), expected_shots=expected)
    except TargetNotFound as e:
        return JSONResponse({"error": str(e), "frame": status}, status_code=422)
    ok, enc = cv2.imencode(".jpg", cv2.resize(res.overlay, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 85])
    rep = res.report
    return {
        "group": {k: rep["group"][k] for k in ("n", "total_score", "offset_mm", "mean_radius_mm")},
        "stage1": {k: rep["stage1"][k] for k in ("shape_ko", "sector_ko", "candidates", "zero_adjust", "notes")},
        "explanation": template_explanation(rep),
        "overlay_jpg": base64.b64encode(enc.tobytes()).decode(),
        "timings": rep["timings"],
        "frame": status,
        "notes": rep["notes"],
    }


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--https", action="store_true")
    ap.add_argument("--cert", default=str(ROOT / "certs" / "dev-cert.pem"))
    ap.add_argument("--key", default=str(ROOT / "certs" / "dev-key.pem"))
    ap.add_argument("--no-token", action="store_true", help="접속 토큰 끄기 (신뢰할 수 있는 네트워크에서만)")
    a = ap.parse_args()
    global TOKEN
    if not a.no_token and TOKEN is None:
        TOKEN = secrets.token_urlsafe(9)
    scheme = "https" if a.https else "http"
    try:
        import socket

        s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s_.connect(("10.255.255.255", 1))           # no packet is sent; just picks the LAN interface
        ip = s_.getsockname()[0]
        s_.close()
    except OSError:
        ip = "127.0.0.1"
    print(f"\n  폰에서 접속: {scheme}://{ip}:{a.port}/" + (f"?t={TOKEN}" if TOKEN else "") + "\n")
    kw = {}
    if a.https:
        if not (Path(a.cert).exists() and Path(a.key).exists()):
            raise SystemExit("인증서가 없습니다. 먼저 bash scripts/make_dev_cert.sh 를 실행하세요.")
        kw = {"ssl_certfile": a.cert, "ssl_keyfile": a.key}
    uvicorn.run(app, host=a.host, port=a.port, **kw)


if __name__ == "__main__":
    main()
