#!/usr/bin/env bash
# 촬영 가이드용 로컬 HTTPS 인증서.
#  - mkcert 가 있으면 사용 (권장: 폰에 루트 인증서를 설치하면 경고 없이 카메라가 켜짐)
#  - 없으면 openssl 자체 서명 인증서 (폰에서 '안전하지 않음' 경고를 넘겨야 함, 기기에 따라 카메라가 막힐 수 있음 → 사진 모드로 자동 전환)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p certs
IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo 127.0.0.1)
if command -v mkcert >/dev/null; then
  mkcert -cert-file certs/dev-cert.pem -key-file certs/dev-key.pem "$IP" localhost 127.0.0.1
  echo "루트 인증서 위치: $(mkcert -CAROOT)/rootCA.pem  → 폰에 설치 후 신뢰 설정"
else
  openssl req -x509 -newkey rsa:2048 -nodes -days 30 -keyout certs/dev-key.pem -out certs/dev-cert.pem \
    -subj "/CN=shootcoach-local" -addext "subjectAltName=IP:$IP,DNS:localhost,IP:127.0.0.1" 2>/dev/null
fi
echo "완료: https://$IP:8600  (python app/capture_server.py --https)"
