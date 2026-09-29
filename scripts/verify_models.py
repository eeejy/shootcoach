"""모델 파일이 저장소에 올린 원본과 같은지 확인 (바꿔치기된 .pt는 불러오는 순간 코드가 실행될 수 있음).

    python scripts/verify_models.py
"""
import hashlib
import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
ok = True
for line in (MODELS / "SHA256SUMS").read_text().split("\n"):
    if not line.strip():
        continue
    digest, name = line.split(maxsplit=1)
    p = MODELS / name.strip()
    if not p.exists():
        print(f"❌ 없음: {name}")
        ok = False
        continue
    real = hashlib.sha256(p.read_bytes()).hexdigest()
    good = real == digest
    ok &= good
    print(f"{'✅' if good else '❌ 체크섬 불일치'} {name}")
sys.exit(0 if ok else 1)
