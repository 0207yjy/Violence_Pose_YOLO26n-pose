"""
Export YOLOv26n-pose to ONNX for Raspberry Pi deployment
"""
from ultralytics import YOLO

def export_to_onnx():
    print("YOLOv26n-pose ONNX 변환 중...")
    print("=" * 70)

    # Load best model
    model_path = "/home/yjy/YOLOv26n_violence/runs/yolov26n-pose-simuletic-100ep-full/weights/best.pt"
    print(f"\n1. 모델 로드 중: {model_path}")
    model = YOLO(model_path)
    print("   ✓ 모델 로드 완료")

    # Export to ONNX
    print(f"\n2. ONNX 변환 시작...")
    print(f"   - half: False (라즈베리파이 호환성)")
    print(f"   - simplify: True (최적화)")
    print(f"   - opset: 12")

    model.export(
        format='onnx',
        half=False,  # Disable FP16 for Raspberry Pi compatibility
        simplify=True,  # Simplify model
        opset=12,  # ONNX opset version
    )

    print(f"\n✓ ONNX 변환 완료!")
    print(f"   저장 위치: /home/yjy/YOLOv26n_violence/yolov26n-pose-simuletic-100ep-full/weights/best.onnx")

    # Check file size
    import os
    onnx_path = '/home/yjy/YOLOv26n_violence/yolov26n-pose-simuletic-100ep-full/weights/best.onnx'
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"   파일 크기: {size_mb:.2f} MB")

if __name__ == "__main__":
    export_to_onnx()
