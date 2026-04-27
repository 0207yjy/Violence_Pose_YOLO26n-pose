# YOLOv26n-pose Violence Detection

CCTV 영상에서 폭력 상황을 탐지하는 YOLOv26n-pose 기반 포즈 추정 모델

## 모델 정보

- **모델**: YOLOv26n-pose (Ultralytics)
- **파라미터**: 3,679,464
- **입력 크기**: 640x640
- **키포인트**: 17개 (COCO 포맷)

## 성능

### Best 모델 (Epoch 88)
| 메트릭 | 값 |
|--------|-----|
| Box mAP50-95 | **0.943** |
| Box mAP50 | **0.995** |
| Pose mAP50-95 | **0.903** |
| Pose mAP50 | **0.995** |
| Precision | 1.000 |
| Recall | 0.999 |

### 추론 속도
- GPU: ~10ms
- CPU: ~30ms

## 데이터셋

[Simuletic CCTV Aggressive Poses Dataset](https://www.kaggle.com/datasets/simuletic/cctv-aggressive-poses-and-fight-detection-dataset)
- 총 103장 이미지 (Train: 82, Val: 21)
- 공격적 자세 포즈 레이블

## 사용 방법

### 모델 다운로드

모델 파일은 GitHub Releases에서 다운로드 가능합니다:
- [best.pt](../../releases) - PyTorch 모델 (7.8 MB)
- [best.onnx](../../releases) - ONNX 모델 (11.6 MB, 라즈베리파이용)

### Python (PyTorch)

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('best.pt')

# 이미지 추론
results = model.predict('image.jpg', conf=0.5)

# 실시간 웹캠
results = model.predict(source=0, show=True)
```

### 라즈베리파이 (ONNX)

```python
import onnxruntime as ort
import cv2
import numpy as np

# ONNX 모델 로드
session = ort.InferenceSession('best.onnx')

# 추론 실행
inputs = {session.get_inputs()[0].name: preprocessed_image}
outputs = session.run(None, inputs)
```

## 학습

```bash
# 데이터셋 준비
python prepare_simuletic_dataset.py

# 학습 (100 epochs)
python train_simuletic_100ep_full.py
```

### 학습 설정
- Epochs: 100
- Batch size: 8
- Optimizer: AdamW
- Learning rate: 0.01
- Early stopping: 비활성화

## ONNX 변환

```bash
python export_onnx.py
```

## 테스트

```bash
# 모델 테스트
python test_final_model.py

# 라벨 비교 검증
python validate_with_labels.py
```

## 프로젝트 구조

```
YOLOv26n_violence/
├── train_simuletic_100ep_full.py    # 학습 스크립트
├── export_onnx.py                    # ONNX 변환
├── test_final_model.py               # 모델 테스트
├── validate_with_labels.py           # 검증 스크립트
├── prepare_simuletic_dataset.py     # 데이터셋 준비
├── TRAINING_SUMMARY.md              # 학습 결과 상세
└── README.md
```

## 폭력 탐지 로직

- 팔 확장 정도 분석 (어깨-손목 거리)
- 공격적 자세 패턴 매칭
- 실시간 추론 가능 (10ms 미만)

## 라이선스

MIT License

## 참고

- [Ultralytics YOLO Docs](https://docs.ultralytics.com)
- [YOLOv26n Paper](https://docs.ultralytics.com/models/yolo26/)
