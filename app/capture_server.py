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

from shootcoach.config import load_target_spec  # noqa: E402
from shootcoach.explain.vlm import template_explanation  # noqa: E402
from shootcoach.pipeline import analyze_target  # noqa: E402
from shootcoach.target.detect import get_detector  # noqa: E402
from shootcoach.target.markers import MarkerError, detect_markers  # noqa: E402

SPEC = load_target_spec()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# Access token: required by default when run from the command line (see main). Tests import the
# module with no token. The page receives it in the URL (?t=...) and sends it back as a header.
TOKEN: str | None = os.environ.get("SHOOTCOACH_TOKEN") or None
POSITION_KO = {0: "왼쪽 위", 1: "오른쪽 위", 2: "오른쪽 아래", 3: "왼쪽 아래"}
app = FastAPI(title="BullsAI 촬영 가이드")
_detector = None


def detector():
    global _detector
    if _detector is None:
        _detector = get_detector("auto")
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


def marker_status(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    found = detect_markers(img, SPEC)
    if len(found) < len(SPEC.markers) and max(h, w) < 1400:
        # Small preview frames: markers can be ~15 px. Retry once on a 2x upscale.
        big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        more = {k: v / 2 for k, v in detect_markers(big, SPEC).items()}
        if len(more) > len(found):
            found = more
    missing = [m for m in SPEC.markers if m not in found]
    return {
        "found": sorted(found),
        "missing": missing,
        "missing_ko": [POSITION_KO.get(m, str(m)) for m in missing],
        "ok": not missing,
        "quads": {str(k): (v / [w, h]).round(4).tolist() for k, v in found.items()},   # normalised corners
    }


@app.get("/")
def index(t: str | None = None):
    if not _check_token(t):
        return JSONResponse({"error": "접속 주소에 토큰(?t=...)이 필요합니다. 서버 실행 창에 표시된 주소로 접속하세요."}, status_code=403)
    return FileResponse(ROOT / "app" / "static" / "capture.html")


EMBLEM = ROOT / "app" / "static" / "brand" / "ssat_emblem.png"


@app.get("/favicon.ico")
def favicon():
    return FileResponse(EMBLEM) if EMBLEM.exists() else Response(status_code=204)


@app.get("/brand/emblem.png")
def emblem():
    # 마크 원본 파일을 그대로 전송 (가공 없음). 파일이 없으면 페이지가 텍스트 로고로 대체한다.
    return FileResponse(EMBLEM, media_type="image/png") if EMBLEM.exists() else Response(status_code=404)


@app.post("/api/markers")
async def api_markers(frame: UploadFile = File(...), x_token: str | None = Header(None)):
    if not _check_token(x_token):
        return JSONResponse({"error": "토큰이 없거나 틀립니다."}, status_code=403)
    try:
        img = _decode(await _read_limited(frame))
        if os.environ.get("SHOOTCOACH_DEBUG_FRAMES"):
            cv2.imwrite(os.environ["SHOOTCOACH_DEBUG_FRAMES"], img)
        return marker_status(img)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/analyze")
async def api_analyze(photo: UploadFile = File(...), handedness: str = Form("right"),
                      distance_m: float = Form(15.0), x_token: str | None = Header(None)):
    if not _check_token(x_token):
        return JSONResponse({"error": "토큰이 없거나 틀립니다."}, status_code=403)
    if handedness not in ("right", "left"):
        return JSONResponse({"error": "handedness는 right 또는 left"}, status_code=400)
    try:
        img = _decode(await _read_limited(photo))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    status = marker_status(img)
    if len(status["found"]) < 3:
        return JSONResponse({"error": "마커가 부족합니다: " + ", ".join(status["missing_ko"]) + " 마커가 보이게 다시 찍어 주세요.",
                             "markers": status}, status_code=422)
    try:
        res = analyze_target(img, SPEC, handedness, distance_m, None, detector())
    except MarkerError as e:
        return JSONResponse({"error": str(e), "markers": status}, status_code=422)
    ok, enc = cv2.imencode(".jpg", cv2.resize(res.overlay, None, fx=0.6, fy=0.6), [cv2.IMWRITE_JPEG_QUALITY, 85])
    rep = res.report
    return {
        "group": {k: rep["group"][k] for k in ("n", "total_score", "offset_mm", "mean_radius_mm")},
        "stage1": {k: rep["stage1"][k] for k in ("shape_ko", "sector_ko", "candidates", "zero_adjust", "notes")},
        "explanation": template_explanation(rep),
        "overlay_jpg": base64.b64encode(enc.tobytes()).decode(),
        "timings": rep["timings"],
        "markers": status,
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
