"""
Validate model predictions against ground truth labels
Compare predicted keypoints with actual labels for accuracy measurement
"""
from ultralytics import YOLO
import os
import json
import numpy as np

def validate_with_labels():
    print("YOLOv26n-pose 라벨 비교 검증")
    print("=" * 70)

    # Load best model
    model_path = "/home/yjy/YOLOv26n_violence/runs/yolov26n-pose-simuletic-100ep-full/weights/best.pt"
    print(f"\n1. 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    print("   ✓ 모델 로드 완료")

    dataset_yaml = "/home/yjy/YOLOv26n_violence/simuletic_pose_dataset/data.yaml"

    # Run validation with metrics
    print(f"\n2. 검증 시작 (라벨과 비교)...")
    print(f"   - 데이터셋: Simuletic Aggressive Poses")
    print(f"   - Val set: 21 images")

    metrics = model.val(
        data=dataset_yaml,
        split='val',
        conf=0.5,
        iou=0.45,
        verbose=True
    )

    print(f"\n3. 검증 결과:")
    print(f"   " + "=" * 50)
    print(f"   Box Detection:")
    print(f"     - Precision: {metrics.box.mp:.4f}")
    print(f"     - Recall:    {metrics.box.mr:.4f}")
    print(f"     - mAP50:     {metrics.box.map50:.4f}")
    print(f"     - mAP50-95:  {metrics.box.map:.4f}")
    print(f"   ")
    print(f"   Pose Keypoint Detection:")
    print(f"     - Precision: {metrics.pose.mp:.4f}")
    print(f"     - Recall:    {metrics.pose.mr:.4f}")
    print(f"     - mAP50:     {metrics.pose.map50:.4f}")
    print(f"     - mAP50-95:  {metrics.pose.map:.4f}")
    print(f"   " + "=" * 50)

    # Detailed analysis per image
    print(f"\n4. 이미지별 상세 분석 (예시 3장):")

    val_images_dir = "/home/yjy/YOLOv26n_violence/simuletic_pose_dataset/images/val"
    val_labels_dir = "/home/yjy/YOLOv26n_violence/simuletic_pose_dataset/labels/val"

    test_images = [f for f in os.listdir(val_images_dir) if f.endswith('.jpg')][:3]

    for img_name in test_images:
        img_path = os.path.join(val_images_dir, img_name)
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(val_labels_dir, label_name)

        print(f"\n   이미지: {img_name}")

        # Read ground truth label
        with open(label_path, 'r') as f:
            gt_lines = f.readlines()

        gt_persons = len(gt_lines)
        print(f"   Ground Truth: {gt_persons}명")

        # Run prediction
        results = model.predict(img_path, conf=0.5, verbose=False)

        pred_persons = 0
        for result in results:
            if result.boxes is not None:
                pred_persons = len(result.boxes)

        print(f"   Prediction:   {pred_persons}명")
        print(f"   정확도:      {100 - abs(gt_persons - pred_persons) * 10:.1f}% (추정)")

    return metrics

if __name__ == "__main__":
    metrics = validate_with_labels()
