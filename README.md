# 🎯 BullsAI — 표적지·자세 융합 AI 사격 진단

> 2026 해양경찰청 AI 해커톤 · 팀 **취향저격**
> 표적지 사진 한 장 → **표적 찾기(마커 없음) → 탄공 검출 → 채점 → 교관 지식베이스 기반 원인 후보**.
> AI가 틀리면 **교관이 사진을 눌러 바로 고친다.** 자세 영상을 더하면 후보 중 원인을 확정한다.
> 모든 처리는 이 PC 안에서만 한다.

```
표적지 사진 ─▶ 검은 원(7점 경계)으로 표적 찾기 ─▶ 탄공 AI 검출 + 발수 보정 ─▶ 채점·탄착군
                                                                               │
                         교관 클릭 수정 (추가/삭제) ◀──────────────────────────┤
                                                                               ▼
                               교관 지식베이스: 상하·좌우 성분 가중합 + 분산 + 사수 프로필 + 사전 점검
                                                                               │
자세 영상 ─▶ 관절 17개 ─▶ 격발 시점 ─▶ 격발 전후 움직임 ─────────────────────▶ 원인 확정 / 배제 / 현장 확인
```

## 설계 원칙 (교관 자료 반영)

아이디어 제시 교관의 정리 자료(「사격 표적지 탄착군 원인 분석」, 「다중 변수 기반 AI 사격 진단 알고리즘」)를 진단의 기준으로 삼는다.

- **1차원 매핑 배제:** "좌하탄 = 무조건 방아쇠" 같은 표 조회를 쓰지 않는다. 탄착군 중심을 **상하 성분과 좌우 성분으로 나눠** 각각의 크기만큼 해당 원인(하탄 L1~L4, 상탄 H1~H4, 좌 LB1~4, 우 RB1~3)에 점수를 주고, 대각선은 교관 자료의 **복합 편향 벡터**대로 짝지어진 원인에 가산한다.
- **분산은 독립 축:** 조밀하게 치우치면 **영점 먼저**(A), 넓게 흩어지면 흔들림·피로·그립 압력(B), 이탈탄은 단정하지 않는다(C).
- **사전 점검·사수 프로필:** 영점 확인 여부, 그립 크기, 손 크기, 손가락 길이, 피로도가 가중치를 바꾼다.
- **교관 검증:** AI는 후보만 낸다. 원인마다 **현장 확인 체크리스트**가 붙고, 교관 판정을 기록해 정확도를 집계한다.
- 지식베이스는 `rules/instructor_kb.yaml` 한 파일이다. 교관이 문구·가중치를 직접 고칠 수 있다.

## 결과

| 평가 | 데이터 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| 탄공 검출 · 공개 시험셋 | Roboflow 시험 206장 / 탄공 1,380개 | 0.923 | 0.933 | **0.928** |
| 탄공 검출 · 해경 실사진 (AI만) | 원형 표적 9장 / 탄공 104개 | 0.964 | 0.769 | 0.856 |
| 탄공 검출 · 해경 실사진 (+ 발수 보정) | 같음 | 0.908 | 0.856 | **0.881** |

- **해경 실사진은 학습에 한 장도 쓰지 않았다.** 정답은 사진을 보고 사람이 표시했다.
- 표적 찾기(마커 없음): 원형 표적 **11/11장** 성공. 하반신·영점 표적은 원형이 아니라고 정상 판별.
- 맞게 찾은 탄공의 **점수 일치 95.5%**.
- 선명한 원본 사진 3장은 **37개 중 36개**를 찾았다. 놓친 탄공은 대부분 흐리거나 압축된 사진, 여러 발이 겹친 곳이다.
- 자세 2단계(합성 영상 6개 상황): 모두 기대한 판정.
- 속도(Apple M4): 표적 사진 1장 **약 0.5초**, 자세 영상 관절 추출 프레임당 21ms(빠름) · 77ms(정밀).

| 버전 | 해경 실사진 F1 (발수 보정) |
|---|---|
| 합성 표적지로만 학습한 1차 모델 | 실사진 탄공 대부분 놓침 |
| 공개 실사진 8에포크 (중간) | 0.813 |
| **공개 실사진 30에포크 + 좌우 뒤집기·배율 TTA (최종)** | **0.881** |

세부 수치: `docs/results.json`, 학습 곡선: `docs/train_real_v1_results.csv`, 속도: `docs/benchmark.json`

## 빠른 시작

```bash
git clone https://github.com/eeejy/shootcoach.git
cd shootcoach
bash scripts/setup.sh                                     # macOS / Linux
# powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # Windows

uv run streamlit run app/streamlit_app.py                 # 웹앱 (기본: 이 PC에서만 접속)
uv run python scripts/analyze.py samples/demo_low_left.jpg --shots 10
uv run pytest -q
```

폰으로 찍어 바로 분석하려면(촬영 가이드: 표적의 검은 원이 충분히 크게 잡히면 셔터가 켜진다):
```bash
bash scripts/make_dev_cert.sh
uv run python app/capture_server.py --https               # 표시되는 주소(토큰 포함)로 폰에서 접속
```

## 사용법

1. **촬영:** 표적 전체(최소한 검은 원 전체)가 나오게 **정면에서** 찍는다. 마커는 필요 없다. 카톡으로 보내면 사진이 압축되니 원본을 쓴다.
2. **입력:** 사이드바에 주로 쓰는 손, 거리, **발수**(완사 10발), 사전 점검(영점·그립), 사수 프로필을 넣는다. 자세 영상은 **자세 분석 모델**을 고른다: 정밀(기본, 관절 위치 떨림이 빠름의 절반 수준) 또는 빠름(분석 시간 약 1/3).
3. **수정:** 결과 사진에서 **빈 곳을 누르면 탄공 추가, 탄공을 누르면 삭제**. 점수와 진단이 바로 다시 계산된다.
4. **검증:** 원인 후보의 현장 확인 항목을 체크하고, **교관 판정 기록**에 실제 원인을 저장한다(`profiles/instructor_feedback.jsonl`, 이 PC에만).
5. **속사(하반신) 표적:** 원형이 아니면 탄공 **개수만** 센다. 영역 채점(2·5·4점)은 규정을 받은 뒤 추가한다.

## 탄공 검출 모델

| 항목 | 내용 |
|---|---|
| 모델 | YOLO11s, 입력 960px, `models/hole_detector_photo.pt` |
| 학습 데이터 | Roboflow 공개 데이터 4종을 '탄공' 한 클래스로 합침 — 학습 3,699장 / 탄공 18,441개 (아래 표) |
| 발수 보정 | 검출 수 ≠ 발수이면 신뢰도 기준을 올리고 내려 다시 고르고, 그래도 모자라면 뭉친 탄공을 나눈다 (서울청 STARS 과제의 요령) |
| 표적 찾기 | 검은 원 타원 맞춤 → 원으로 펴기. 링 간격은 25m 정밀 권총 표적 구조(10점 반지름 25mm, 링마다 +25mm)로 가정 — `configs/kcg_circle.yaml` |

| 데이터셋 (Roboflow Universe) | 이미지 | 라이선스 |
|---|---|---|
| [bullet-hole-object-detection v30](https://universe.roboflow.com/project-bat-bullet-hole-detection/bullet-hole-object-detection) (Butt et al.) | 3,099 | CC BY 4.0 |
| [knsa-shooting-target v10](https://universe.roboflow.com/knsashootingtargets/knsa-shooting-target) | 176 | CC BY 4.0 |
| [carbine-target-bullet-hole-detection v1](https://universe.roboflow.com/project-bat-bullet-hole-detection/carbine-target-bullet-hole-detection) | 102 | CC BY 4.0 |
| [kanat2.0-yolov11 v5](https://universe.roboflow.com/kanat-hefkh/kanat2.0-yolov11-bullets-targets-and-boards) | 846 | Public Domain |

데이터를 다시 받으려면 Roboflow 무료 계정의 API 키를 `.env`에 `ROBOFLOW_API_KEY=...`로 넣고:
```bash
uv run python scripts/download_public_datasets.py
uv run python scripts/build_public_dataset.py
uv run python scripts/train_detector.py --data data/real_v1/data.yaml --model models/yolo11s.pt --imgsz 960 --batch 8 --epochs 30
```

## 폴더

| 경로 | 내용 |
|---|---|
| `shootcoach/target/locate.py` | 마커 없이 표적 찾기 (검은 원) |
| `shootcoach/target/photo.py` | 원본 사진 탄공 검출 + 발수 보정 + 뭉친 탄공 나누기 |
| `shootcoach/diagnosis/stage1.py` | 교관 지식베이스 가중합 진단 |
| `shootcoach/diagnosis/stage2.py` | 자세 영상 판정 |
| `rules/instructor_kb.yaml` | **교관 지식베이스** (원인·체크리스트·가중치·대각선 조합) |
| `rules/posture_signals.csv` | 자세 신호 임계값 |
| `app/streamlit_app.py` | 웹앱 · `app/capture_server.py` 폰 촬영 가이드 |
| `scripts/` | 분석 CLI, 데이터 받기·합치기, 학습, 평가, 벤치마크, 교관 검증 리포트, 자세 임계값 보정 |
| `models/` | 탄공 모델, 자세 모델, 학습 시작점 + 체크섬 ([models/README.md](models/README.md)) |
| `samples/` | 데모 사진(합성), 실제 사격 영상 샘플(퍼블릭 도메인) |

## 한계

- **해경 표적 실사진 정답 데이터가 적다.** 공개 데이터는 비슷한 표적이지만 같은 표적은 아니다. 교관 수정 기능으로 모인 결과가 가장 좋은 추가 학습 데이터가 된다.
- **표적 찾기는 타원→원 근사**라 많이 비스듬한 사진에서는 수 mm 오차가 생긴다(링 간격 25mm 대비). 정면 촬영을 권장한다.
- **링 간격은 가정값**이다. 실물 표적을 자로 재서 다르면 `configs/kcg_circle.yaml`만 고친다.
- **측면 자세 영상으로는** 하방 원인(반동 예측·급작 격발·동반 수축)을 서로 구분하기 어렵다. 교관 현장 확인이 필요하다.
- **속사(하반신) 표적은 개수만** 센다.

## 라이선스·데이터 고지

- Ultralytics YOLO(AGPL-3.0) 사용. 외부 서비스로 배포하려면 AGPL 조건을 따르거나 상용 라이선스가 필요하다.
- 학습 데이터 출처는 위 표. CC BY 4.0 데이터는 출처 표기 조건을 따른다.
- 교관 자료·경찰청 과제 자료·해경 표적 사진 원본은 저장소에 넣지 않았다.
- `samples/real_video/`는 미 해병대·육군 퍼블릭 도메인 영상에서 잘라낸 것이다.
