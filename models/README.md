# 모델

| 파일 | 용도 | 출처 | 학습 데이터 |
|---|---|---|---|
| `hole_detector.pt` | **탄공 검출 (최종, v2)** — 기본으로 쓰는 모델 | 이 프로젝트에서 학습 (YOLO11n) | 합성 표적지 3,000장 (링 1,600 + 실루엣 선 그림 1,400), v1에서 이어 25에포크 |
| `hole_detector_v1.pt` | 탄공 검출 v1 (비교용) | 이 프로젝트에서 학습 (YOLO11n) | 합성 링 표적 1,600장, 15에포크 |
| `yolo11n-pose.pt` | 자세 관절 17개 추출 | Ultralytics 공식 배포 모델 (v8.4.0 assets) | COCO keypoints |
| `yolo11n.pt` | 새로 학습할 때 시작점 | Ultralytics 공식 배포 모델 | COCO |

- 성능: [../docs/results.json](../docs/results.json), 학습 곡선: `../docs/train_v*_results.csv`
- **합성 데이터로만 학습**했다. 실제 해경 표적지로 파인튜닝해야 한다 (README "실제 데이터로 넘어가기").
- 라이선스: Ultralytics YOLO 및 파생 가중치는 **AGPL-3.0**.

## 무결성

`.pt` 파일은 파이썬 pickle이라 **불러오는 순간 안의 코드가 실행될 수 있다.** 출처를 모르는 `.pt`는 쓰지 말고, 저장소 파일은 체크섬으로 확인한다.

```bash
python scripts/verify_models.py          # SHA256SUMS 와 비교
```

모델을 새로 학습해 교체했다면 `cd models && shasum -a 256 *.pt > SHA256SUMS` 로 갱신해 함께 커밋한다.
