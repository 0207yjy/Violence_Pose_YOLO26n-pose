# YOLOv26n-pose 폭력 탐지 모델 훈련 결과

## 모델 정보
- **모델**: YOLOv26n-pose (Ultralytics)
- **파라미터**: 3,679,464
- **입력 크기**: 640x640
- **키포인트**: 17개 (COCO 포맷)

## 데이터셋
- **출처**: Simuletic CCTV Aggressive Poses Dataset
- **이미지**: 103장 (Train: 82, Val: 21)
- **라벨**: 17개 COCO 키포인트 + 바운딩 박스

## 훈련 결과

### Best 모델 (에폭 88)
| 메트릭 | 값 |
|--------|-----|
| Box mAP50-95 | **0.943** |
| Box mAP50 | **0.995** |
| Pose mAP50-95 | **0.903** |
| Pose mAP50 | **0.995** |
| Precision | 1.000 |
| Recall | 0.999 |

### 최종 모델 (에폭 100)
| 메트릭 | 값 |
|--------|-----|
| Box mAP50-95 | **0.938** |
| Box mAP50 | **0.995** |
| Pose mAP50-95 | **0.917** |
| Pose mAP50 | **0.995** |
| Precision | 1.000 |
| Recall | 0.999 |

## 테스트 결과
- **테스트 이미지**: 5장
- **폭력 탐지 성공**: 2/5장에서 공격적 자세 감지
- **추론 속도**: 8-11ms (GPU), 2-3ms (CPU 전처리)

## 파일 정보

### PyTorch 모델
- **best.pt**: 7.8 MB (에폭 88)
- **last.pt**: 7.8 MB (에폭 100)

### ONNX 모델
- **best.onnx**: 11.6 MB (라즈베리파이 배포용)
- **설정**: FP32, opset=12, simplified

## 사용 방법

### Python (PyTorch)
```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model.predict('image.jpg', conf=0.5)
```

### Python (ONNX)
```python
import onnxruntime as ort
session = ort.InferenceSession('best.onnx')
# 라즈베리파이에서 실행
```

## 폭력 탐지 로직
- 팔 확장 정도 분석 (어깨-손목 거리)
- 공격적 자세 패턴 매칭
- 실시간 추론 가능 (10ms 미만)

## 훈련 설정
- 에폭: 100
- 배치 크기: 8
- 학습률: 0.01 → 0.01 (AdamW)
- 최적화: patience=0 (조기 종료 없음)
- 데이터 증강: Mosaic=0.5

## 결과 저장 위치
- 훈련 결과: `runs/yolov26n-pose-simuletic-100ep-full/`
- 테스트 결과: `test_results_final/predictions/`
