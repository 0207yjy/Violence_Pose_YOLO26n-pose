"""
Test trained YOLOv26n-pose model on violence detection
"""
from ultralytics import YOLO
import os
import cv2
import numpy as np

def test_model():
    print("YOLOv26n-pose 최종 모델 테스트")
    print("=" * 70)

    # Load best model
    model_path = "/home/yjy/YOLOv26n_violence/runs/yolov26n-pose-simuletic-100ep-full/weights/best.pt"
    print(f"\n1. 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    print("   ✓ 모델 로드 완료")

    # Test on validation dataset
    dataset_dir = "/home/yjy/YOLOv26n_violence/simuletic_pose_dataset"
    val_images_dir = os.path.join(dataset_dir, "images", "val")

    if os.path.exists(val_images_dir):
        print(f"\n2. 검증 이미지로 테스트 중...")
        test_images = [f for f in os.listdir(val_images_dir) if f.endswith('.jpg')][:5]

        results_dir = "/home/yjy/YOLOv26n_violence/test_results_final"
        os.makedirs(results_dir, exist_ok=True)

        for img_name in test_images:
            img_path = os.path.join(val_images_dir, img_name)
            print(f"\n   테스트: {img_name}")

            # Run inference
            results = model.predict(
                img_path,
                save=True,
                project=results_dir,
                name='predictions',
                conf=0.5,
                iou=0.45
            )

            # Analyze results
            for result in results:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    print(f"     감지된 사람 수: {len(result.keypoints)}")

                    # Violence detection based on pose
                    violence_detected = False
                    for person_idx, kpts in enumerate(result.keypoints):
                        kp_data = kpts.xy[0]  # First person's keypoints

                        if len(kp_data) >= 17:
                            # Simple violence detection logic
                            # Check for aggressive poses
                            left_shoulder = kp_data[5].cpu().numpy()
                            right_shoulder = kp_data[6].cpu().numpy()
                            left_elbow = kp_data[7].cpu().numpy()
                            right_elbow = kp_data[8].cpu().numpy()
                            left_wrist = kp_data[9].cpu().numpy()
                            right_wrist = kp_data[10].cpu().numpy()

                            # Calculate arm extension
                            if left_elbow[0] > 0 and left_wrist[0] > 0:
                                left_extension = np.linalg.norm(left_wrist - left_shoulder)
                                if left_extension > 100:  # Threshold for aggressive pose
                                    violence_detected = True
                                    break

                    if violence_detected:
                        print(f"     ⚠️  폭력 가능성 감지!")
                    else:
                        print(f"     ✓ 정상")

        print(f"\n   ✓ 테스트 완료! 결과 저장: {results_dir}/predictions")

    # Final validation metrics
    print("\n3. 최종 검증 메트릭...")
    metrics = model.val(data='/home/yjy/YOLOv26n_violence/simuletic_pose_dataset/data.yaml')
    print(f"   Box mAP50-95: {metrics.box.map:.4f}")
    print(f"   Box mAP50: {metrics.box.map50:.4f}")
    print(f"   Pose mAP50-95: {metrics.pose.map:.4f}")
    print(f"   Pose mAP50: {metrics.pose.map50:.4f}")

    return model

if __name__ == "__main__":
    model = test_model()
