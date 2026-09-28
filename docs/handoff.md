# 인수인계 — 야간 작업 결과와 팀이 이어서 할 일

## 3일 스프린트 대비 진행 상황

| 계획 항목 | 상태 | 비고 |
|---|---|---|
| Day 0 · Python 3.12 환경, Ollama 모델 | ✅ | `uv` + `.venv`, `qwen2.5vl:3b` 받음, `qwen3:8b`는 기존 설치분 사용 |
| Day 0 · ArUco 마커 표적지 템플릿 | ✅ | `samples/a4_practice_target.pdf` (100% 인쇄) |
| Day 0 · 교범 기반 규칙표 초안 | ✅ **(팀장 검토 필요)** | `rules/*.csv`, 근거 `docs/rule_sources.md` |
| Day 0 · Kaggle/Roboflow 가입 | ⏳ | 계정·API 키는 사람이 해야 함. 합성 데이터로 먼저 진행 |
| Day 0 · 해경 표적지 규격 확인 | ⏳ | 규격이 나오면 `configs/`에 새 YAML 추가 |
| Day 1 · 마커 → 호모그래피 → mm | ✅ | 재투영 오차 약 0.03mm (합성) |
| Day 1 · YOLO 학습 | ✅ (합성) | 실제 데이터 파인튜닝은 ⏳ |
| Day 1 · 탄착 통계 + 모양 분류 | ✅ | 7종 모양, 플라이어 제외 |
| Day 1 · 회차 간 IoU 매칭 | ✅ | `target/sequence.py`, CLI `--prev` |
| Day 1 · 1단계 진단 엔진 + 결과 화면 | ✅ | 웹앱 ① 탭 |
| Day 2 · 관절 + 격발 검출 | ✅ | 반동 스파이크 + 총성 + 편집 컷 제외 |
| Day 2 · 자세 특징 + 2단계 엔진 | ✅ (합성 검증) | 실제 영상 임계값 보정 ⏳ |
| Day 2 · VLM 설명 | ✅ | 검증 가드, qwen3:8b 우선 |
| Day 2 · 캘리브레이션 세션 | ✅ | `diagnosis/calibration.py`, CLI `--profile` `--calibrate` (웹앱 연결은 ⏳) |
| Day 3 · 웹앱 | ✅ | Streamlit, 폰 업로드 |
| Day 3 · 모바일 촬영 가이드(마커 인식 시 셔터) | ⏳ | HTTPS 필요 (`mkcert`) |
| Day 3 · 교관 판정 대비 검증 10건 | ⏳ | **실제 데이터가 있어야 가능** |
| Day 3 · 발표 자료 | ⏳ | `docs/img/` 이미지와 결과 표 활용 |

## 내일 가장 먼저 할 일 (우선순위)

1. **팀장: 규칙표 검토** — `rules/causes.csv`의 원인·교정 문구, `rules/stage1_rules.csv`의 `prior`. K5 권총 기준으로 맞지 않는 항목 표시.
2. **팀장: 실제 데이터 촬영** — 마커 스티커 붙인 표적지 사진 50장, 측면 삼각대 **무편집** 자세 영상 5명 × 5발 (가능하면 240fps, 총성 녹음).
3. **팀원2: 해경 표적지 라벨링 → 맥에서 파인튜닝** (`scripts/train_detector.py --model models/hole_detector.pt`). 외부 클라우드에 올리지 않는다.
4. **팀원4: 실제 영상으로 2단계 임계값 보정** — `rules/posture_signals.csv`의 `threshold`. 교관이 "저킹 있음/없음"을 표시한 영상과 비교.
5. **팀원3: 웹앱 다듬기** — 폰 화면 레이아웃, 촬영 가이드(HTTPS).

## 실행 명령 모음

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py                      # 웹앱
python scripts/analyze.py samples/demo_jerking_low_left.jpg --vlm
python scripts/evaluate.py --n 200                      # 합성 평가 → docs/results.json
python scripts/benchmark.py                             # 이 PC 속도 → docs/benchmark.json
pytest -q
```
