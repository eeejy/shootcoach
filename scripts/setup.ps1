# 새 컴퓨터(Windows)에서 한 번 실행:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "uv 설치 중… (https://docs.astral.sh/uv/)"
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv sync --extra ml --extra app --extra dev
uv run python scripts/verify_models.py
uv run pytest -q
uv run python scripts/analyze.py samples/demo_jerking_low_left.jpg --out out
Write-Host "✅ 준비 완료: uv run streamlit run app/streamlit_app.py"
Write-Host "NVIDIA GPU가 있으면 학습이 자동으로 CUDA를 씁니다. (CUDA용 PyTorch가 필요하면: https://pytorch.org/get-started/locally/)"
