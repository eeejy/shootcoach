# 🎯 취향저격 — 표적지·자세 융합 AI 사격 교정 (MVP)

> 2026 해양경찰청 AI 해커톤 · 팀 **취향저격**
> **표적지 사진 한 장**으로 채점하고 원인 후보와 교정 가이드를 준다(1단계). **자세 영상**을 더하면 후보 중 하나를 확정한다(2단계).
> 모든 처리는 **이 PC 안에서만** 돌아간다. 외장 GPU가 없는 MacBook Air(M4)에서 전체 파이프라인을 확인했다.

```
표적지 사진 ─▶ 마커 정면 보정 ─▶ 탄공 검출(YOLO) ─▶ 채점·탄착군 ─▶ 1단계: 원인 후보 + 교정 가이드
                                                                        │
자세 영상 ──▶ 관절 17개 ──▶ 격발 시점 ──▶ 격발 전후 움직임 ─────────────▶ 2단계: 확정 / 배제 / 관측 불가
                                                                        │
                                                        로컬 LLM(검증 가드) ─▶ 사수용 설명 문장
```

## 왜 이렇게 만들었나

- **탄착점 인식은 이미 풀린 문제다.** 합참도 2021년에 발주했다. 우리는 그다음 단계인 **"왜 빗나갔나"와 "어떻게 고치나"**를 다룬다.
- **표적지만으로는 원인이 여러 개로 남는다.** 예를 들어 오른손잡이가 왼쪽 아래로 몰리면 저킹·총기 기울임·방아쇠 밀기가 모두 후보다. 교관이 사수를 옆에서 지켜보듯, 카메라로 **격발 직전 총구가 꺼지는지** 확인해 하나로 좁힌다.
- **판정은 규칙 엔진이 한다.** 규칙은 교범 3종(FM 3-23.35, USAMU, TargetShooting Canada)이 근거이고, 교관이 CSV로 직접 고칠 수 있다. 언어 모델은 결과를 사람 말로 옮기기만 하고, 검증을 통과하지 못하면 템플릿 문장으로 대체된다.
- **모르면 모른다고 한다.** 측면 영상으로 볼 수 없는 원인(총기 기울임, 시선 등)은 "관측 불가", 근거가 없으면 "보류"로 표시한다.

## 결과 (합성 검증 데이터 · 이 맥북에서 측정)

### 1단계 — 표적지 (학습에 안 쓴 합성 사진 200장, 원근 왜곡·조명·그림자 포함)

| 항목 | **YOLO11n (최종)** | 대체 검출기 (학습 불필요) |
|---|---|---|
| 정면 보정 실패 | **0 / 200** (재투영 오차 평균 0.033mm) | 동일 |
| 탄공 정밀도 / 재현율 | **0.988 / 0.883** | 0.745 / 0.725 |
| 탄공 위치 오차 | **0.51mm** | 2.08mm |
| 발별 점수 일치 | **98.2%** | 88.8% |
| 탄착군 모양 분류 일치 | **95.0%** | 71.0% |

> 재현율이 0.88에 머무는 주된 원인은 **겹친 탄공**이다. 사진 한 장으로는 한계가 있어서, 5발마다 찍어 새 탄공만 골라내는 방식(`--prev`)을 넣었다.

### 2단계 — 자세 (합성 자세 데이터, 시나리오 7종 × 20회)

| 시나리오 | 기대 판정 | 정확도 |
|---|---|---|
| 왼쪽 아래 탄착 + 격발 직전 총구 하강 | 저킹 확정 | 100% |
| 왼쪽 아래 탄착 + 정상 자세 | **보류** (근거 없음) | 100% |
| 오른쪽 위 탄착 + 총구 들림·어깨 긴장 | 힐링/반동 예상 확정 | 100% |
| 아래 탄착 + 격발 직후 팔 내림 | 팔로스루 부족 확정 | 100% |
| 아래 탄착 + 격발 직전 총구 하강 | 반동 예상(누름) 확정 | 100% |
| 세로 줄 탄착 + 호흡 흔들림 | 호흡 확정 | 100% |
| 세로 줄 탄착 + 정상 자세 | **보류** | 100% |

### 속도 — MacBook Air (Apple M4, 24GB, 외장 GPU 없음)

| 항목 | 측정 | 목표 |
|---|---|---|
| 표적지 1장 전체 (보정 + 검출 + 진단) | **48ms** (최대 57ms) | 3초 ✅ |
| 자세 관절 추출 | **프레임당 21ms** | — |
| 설명 문장 · qwen3:8b | **3.6초**, 검증 3/3 통과 | 15초 ✅ |
| 설명 문장 · qwen2.5vl:3b | 2.1초, 검증 2/3 통과 | — |

원본 수치: [docs/results.json](docs/results.json), [docs/benchmark.json](docs/benchmark.json)

### 실제 영상·사진에서 확인한 것 (미 해병대·육군 퍼블릭 도메인 영상)

- ✅ 실제 사수의 관절 추출은 된다 ([이미지](docs/img/real_pose_army_pd.jpg)).
- ✅ 편집 컷을 격발로 오인하던 문제를 찾아 고쳤다 → 편집된 영상에서는 **보류**로 답한다.
- ⚠️ 합성으로만 학습한 검출기는 실제 실루엣 표적에서 **선 끝 오검출**이 많았다. 선 그림 음성 샘플을 넣은 v2에서 오검출이 1/3 이하로 줄었다(15·16·45개 → 2·3·15개). **하지만 실제 탄공도 대부분 놓친다** ([비교](docs/img/real_target_v1_vs_v2_pd.jpg)). → **해경 실제 표적지 파인튜닝이 필수다.**


> ⚠️ 수치는 **학습에 쓰지 않은 합성 데이터** 기준이다. 실제 해경 표적지·자세 영상에서의 성능은 아직 측정하지 않았다. 실제 영상에서 확인한 한계는 [리서치 기록](docs/research_log.md) §3에 정리했다.

## 빠른 시작 — 다른 컴퓨터에서 클론해서 쓰기

```bash
git clone https://github.com/eeejy/ai-shooting-coach.git
cd ai-shooting-coach
bash scripts/setup.sh                                     # macOS / Linux
# powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # Windows
```

설치 스크립트가 하는 일: [uv](https://docs.astral.sh/uv/) 설치(없으면) → Python 3.12 가상환경 + 고정 버전 패키지(`uv.lock`) → 모델 체크섬 확인 → 테스트 → 데모 분석 1건.
모델 4개(`models/`)와 데모 사진·실제 영상 샘플이 저장소에 들어 있어 **인터넷이 없는 내부망에서도** 바로 돌아간다. 학습 장치는 자동 선택된다(NVIDIA → CUDA, 맥 → MPS, 그 외 CPU).

```bash
uv run streamlit run app/streamlit_app.py          # 웹앱 (기본: 이 PC에서만 접속)
uv run python scripts/analyze.py samples/demo_jerking_low_left.jpg --distance 15 --click-mm-per-10m 5
uv run python scripts/analyze.py samples/demo_jerking_low_left.jpg --video samples/real_video/clip_army_5_9.mp4
uv run pytest -q
```

학습 데이터는 저장소에 없다(용량). 필요하면 다시 만든다:
```bash
uv run python scripts/make_synthetic_dataset.py --out data/synth          # 링 표적 1,900장 (~1분)
uv run python scripts/make_synthetic_dataset_v2.py --base data/synth --out data/synth_v2
uv run python scripts/train_detector.py --data data/synth_v2/data.yaml --model models/hole_detector.pt --epochs 5
```

**폰 촬영 가이드 (마커 4개가 보일 때만 셔터가 켜짐):**
```bash
bash scripts/make_dev_cert.sh                # 로컬 HTTPS 인증서 (mkcert 있으면 사용, 없으면 자체 서명)
python app/capture_server.py --https         # 폰에서 https://<PC IP>:8600
python app/capture_server.py                 # HTTP면 사진 모드 (찍은 뒤 빠진 마커를 알려 줌)
```
폰 브라우저는 HTTPS에서만 실시간 카메라를 허용한다. 자체 서명 인증서는 경고를 넘겨야 하고 기기에 따라 카메라가 막힐 수 있다. 그 경우 사진 모드로 자동 전환된다. mkcert 루트 인증서를 폰에 설치하면 경고 없이 동작한다.

**설명 문장(선택):** [Ollama](https://ollama.com) 설치 후 `ollama pull qwen3:8b`(권장) 또는 `ollama pull qwen2.5vl:3b`. 없으면 템플릿 문장을 쓴다.

### 보안 기본값

| 항목 | 기본 동작 |
|---|---|
| 웹앱(Streamlit) | **이 PC에서만 접속** (`.streamlit/config.toml`). 폰에서 쓰려면 `--server.address 0.0.0.0` — 같은 네트워크 누구나 접속 가능하므로 신뢰할 수 있는 네트워크에서만. Streamlit 사용 통계 전송은 꺼 둠 |
| 라벨링 페이지 | `data/` 폴더 아래만 읽고 쓸 수 있음 |
| 촬영 가이드 서버 | 실행할 때마다 **무작위 접속 토큰**을 만들어 주소(`?t=...`)에 붙임. 토큰 없이는 접속·분석 불가. 업로드 최대 20MB |
| 모델 파일 | `.pt`는 불러오는 순간 코드가 실행될 수 있음 → `scripts/verify_models.py`로 체크섬 확인, 출처 모르는 `.pt` 사용 금지 |
| 관절 파일(`.npz`) | pickle 비활성화로 읽음 |

## 실제 사격장에서 쓰는 법

1. **표적지:** `samples/a4_practice_target.pdf`를 **100% 배율**로 인쇄한다. 기존 해경 표적지를 쓸 때는 같은 ArUco 마커 4장을 스티커로 네 모서리에 붙이고, 마커 중심 위치(mm)를 재서 `configs/target_a4.yaml`을 복사해 새 설정 파일을 만든다.
2. **촬영:** 네 모서리 마커가 모두 보이게 찍는다. 기울어져도 된다. 겹친 탄공이 걱정되면 5발마다 찍는다(`shootcoach/target/sequence.py`).
3. **자세 영상:** 사수 **측면 2m, 높이 1.2m**에 삼각대를 두고 **편집 없이 연속 촬영**한다. 전신이 보여야 하고, 가능하면 240fps 슬로모션으로 찍는다. 총성이 녹음되면 격발 시점이 더 정확하다. 좌우 흔들림까지 보려면 후방 영상을 추가한다.
4. **규칙 조정:** `rules/stage1_rules.csv`(위치·모양 → 원인 가중치), `rules/causes.csv`(교정 문구·훈련), `rules/posture_signals.csv`(자세 신호 임계값). 코드는 고칠 필요 없다.

## 실제 데이터로 넘어가기 — 내일 쓸 도구

| 할 일 | 도구 | 결과물 |
|---|---|---|
| 해경 표적지 라벨링 (**PC 안에서만**) | 웹앱 왼쪽 메뉴 **🏷️ 라벨링** — 모델이 먼저 찍고, 클릭으로 추가·번호로 삭제 | `data/haegyeong/dataset/` (YOLO 형식) |
| 파인튜닝 | `python scripts/train_detector.py --data data/haegyeong/dataset/data.yaml --model models/hole_detector.pt --epochs 20` | 새 모델 |
| 자세 임계값 보정 | 교관이 영상별 신호 유무(1/0)를 적은 CSV → `python scripts/tune_posture_thresholds.py data/posture_labels.csv` | `rules/posture_signals.suggested.csv` + 리포트 (기존 규칙은 그대로) |
| 교관 판정 대비 검증 | 표적지·영상·교관 판정 CSV → `python scripts/validate_vs_instructor.py data/validation.csv` | `docs/validation_report.md` (1순위·상위 3·2단계 적중률, 보류율) |

각 스크립트 맨 위 주석에 CSV 형식 예시가 있다.

### (선택) 공개 데이터 추가

```bash
# Roboflow 공개 탄공 데이터 (API 키 필요: https://app.roboflow.com → Settings → API)
pip install roboflow
python - <<'PY'
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
rf.workspace("project-bat-bullet-hole-detection").project("bullet-hole-object-detection").version(1).download("yolov8", location="data/roboflow_bh")  # 버전 번호는 데이터셋 페이지에서 확인
PY
# 공개 데이터는 클라우드(Kaggle)에서 학습해도 된다. 해경 데이터는 반드시 이 PC에서만:
python scripts/train_detector.py --data data/haegyeong/data.yaml --model models/hole_detector.pt --epochs 20
```

## 폴더

| 경로 | 내용 |
|---|---|
| `shootcoach/` | 파이프라인 코드 (구조: [docs/architecture.md](docs/architecture.md)) |
| `rules/` | **교관이 수정하는 규칙표** · 근거: [docs/rule_sources.md](docs/rule_sources.md) |
| `app/streamlit_app.py` | 로컬 웹앱 (+ `app/pages/1_라벨링.py`) |
| `app/capture_server.py` | 폰 촬영 가이드 서버 (+ `app/static/capture.html`) |
| `scripts/` | 표적지 생성, 합성 데이터, 학습, 평가, 벤치마크, CLI |
| `models/` | 탄공 검출 모델(최종·v1), 자세 모델, 학습 시작점 + 체크섬 — [models/README.md](models/README.md) |
| `samples/` | 인쇄용 표적지, 시나리오별 데모 사진(합성), 실제 사격 영상 샘플(퍼블릭 도메인) |
| `docs/` | 설계, 리서치 기록, 평가 결과, 벤치마크 |
| `tests/` | 단위·시나리오 테스트 |

## 한계와 다음 단계

| 한계 | 다음 단계 |
|---|---|
| 탄공 검출기가 **합성 데이터로만** 학습됨. 실제 실루엣 표적에서 선 끝 오검출 확인 (v2에서 완화) | 해경 표적지 50~100장 라벨링 → 맥 로컬 파인튜닝 |
| 2단계 신호 임계값은 **합성 자세 데이터**로만 검증 | 교관 판정이 붙은 실제 영상 10~20건으로 임계값 재조정 |
| 측면 영상은 좌우 흔들림·총기 기울임을 못 봄 | 후방 카메라 추가 (신호 정의는 이미 있음: `views=rear`) |
| 규칙 가중치(`prior`)는 자료 기반 초기값 | 교관 판정 데이터로 보정, 사수별 캘리브레이션 |
| K5 등 조준기 1클릭 값이 미정 | 기종별 값 입력 시 클릭 수 자동 계산 (구현됨) |

## 라이선스·데이터 고지

- 탄공 검출·관절 추출에 **Ultralytics YOLO (AGPL-3.0)**를 쓴다. 외부에 서비스로 배포하려면 AGPL 조건을 따르거나 상용 라이선스를 받아야 한다. 해커톤 이후 정식 도입 단계에서 검토한다.
- `samples/real_video/`와 `docs/img/*_pd.jpg`는 미 해병대·미 육군 영상(퍼블릭 도메인, Wikimedia Commons)에서 잘라낸 것이다. 출처: [samples/real_video/README.md](samples/real_video/README.md).
- 교범 PDF 원문은 저장소에 넣지 않았다. 링크는 [docs/rule_sources.md](docs/rule_sources.md)에 있다.
- 데모 사진은 모두 합성이다. 실제 사람이나 해경 자료는 들어 있지 않다.
