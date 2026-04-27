"""
Train YOLOv26n-pose for full 100 epochs without early stopping
"""
from ultralytics import YOLO
import os

def train_full_100_epochs():
    print("YOLOv26n-pose 100 에폭 전체 훈련 - Simuletic 데이터셋")
    print("조기 종료 없음")
    print("=" * 70)

    output_dir = "/home/yjy/YOLOv26n_violence/runs"
    dataset_yaml = "/home/yjy/YOLOv26n_violence/simuletic_pose_dataset/data.yaml"

    # Load pretrained YOLOv26n-pose model
    print("\n1. 사전 학습된 YOLOv26n-pose 모델 로드 중...")
    model = YOLO('yolo26n-pose.pt')
    print("   ✓ 모델 로드 완료")

    # Train for full 100 epochs without early stopping
    print(f"\n2. Simuletic 데이터셋으로 100 에폭 훈련 시작...")
    print(f"   - 데이터셋: Simuletic Aggressive Poses")
    print(f"   - Train set: 82 images")
    print(f"   - Val set: 21 images")
    print(f"   - 에폭: 100 (조기 종료 없음)")
    print(f"   - Batch size: 8")
    print(f"   - Device: GPU 1")

    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        name='yolov26n-pose-simuletic-100ep-full',
        project=output_dir,
        device=1,
        plots=True,
        save=True,
        verbose=True,
        patience=0,  # Disable early stopping
        val=True,
        save_period=10,
        lr0=0.01,
        lrf=0.01,
        mosaic=0.5,
        pose=12.0,
        kobj=1.0,
    )

    print(f"\n✓ 100 에폭 훈련 완료!")

    # Validate
    print("\n3. 최종 모델 검증 중...")
    metrics = model.val()
    print(f"   Box mAP50-95: {metrics.box.map:.4f}")
    print(f"   Box mAP50: {metrics.box.map50:.4f}")
    print(f"   Pose mAP50-95: {metrics.pose.map:.4f}")
    print(f"   Pose mAP50: {metrics.pose.map50:.4f}")

    return model

if __name__ == "__main__":
    model = train_full_100_epochs()
