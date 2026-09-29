#!/usr/bin/env bash
# 새 컴퓨터(macOS / Linux)에서 한 번 실행: 가상환경 + 패키지 + 모델 확인 + 테스트
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null; then
  echo "uv 설치 중… (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra ml --extra app --extra dev
uv run python scripts/verify_models.py
uv run pytest -q
uv run python scripts/analyze.py samples/demo_jerking_low_left.jpg --out out
command -v ollama >/dev/null && echo "Ollama 있음 → 설명 문장용: ollama pull qwen3:8b" || echo "(선택) 설명 문장용 Ollama: https://ollama.com → ollama pull qwen3:8b"
echo "✅ 준비 완료: uv run streamlit run app/streamlit_app.py"
