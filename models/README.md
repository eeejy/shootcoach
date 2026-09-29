# 모델

| 파일 | 용도 | 출처 | 학습 데이터 |
|---|---|---|---|
| `hole_detector_photo.pt` | **탄공 검출 (기본)** — 원본 사진을 그대로 넣는다 | 이 프로젝트에서 학습 (YOLO11s, 입력 960px) | Roboflow 공개 실사진 4종 3,699장 / 탄공 18,441개 (README 표) |
| `yolo11n-pose.pt` | 자세 관절 17개 추출 | Ultralytics 공식 배포 모델 | COCO keypoints |
| `yolo11s.pt` | 탄공 모델을 새로 학습할 때 시작점 | Ultralytics 공식 배포 모델 | COCO |
| `yolo11n.pt` | 가벼운 시작점 (예비) | Ultralytics 공식 배포 모델 | COCO |

| 성능 | 값 |
|---|---|
| 공개 검증셋 318장 mAP50 / mAP50-95 | 0.79 / 0.41 (30에포크, M4 GPU로 6.7시간) |
| 공개 시험셋 206장 F1 | 0.928 |
| 해경 원형 표적 실사진 9장 F1 (발수 보정, 학습 미사용) | 0.881 |

추론은 기본으로 TTA(좌우 뒤집기·배율)를 켠다. 끄려면 `PhotoHoleDetector(augment=False)`.

- 해경 표적 실사진으로는 학습하지 않았다. 앱의 교관 수정 결과가 모이면 그 데이터로 추가 학습한다.
- 라이선스: Ultralytics YOLO 및 파생 가중치는 **AGPL-3.0**. 학습 데이터는 CC BY 4.0 / Public Domain.

## 무결성

`.pt` 파일은 파이썬 pickle이라 **불러오는 순간 안의 코드가 실행될 수 있다.** 출처를 모르는 `.pt`는 쓰지 말고, 저장소 파일은 체크섬으로 확인한다.

```bash
uv run python scripts/verify_models.py          # SHA256SUMS 와 비교
```

모델을 새로 학습해 교체했다면 `cd models && shasum -a 256 *.pt > SHA256SUMS` 로 갱신해 함께 커밋한다.
