"""
Prepare Simuletic Aggressive Poses Dataset for YOLOv26n-pose training
"""
import os
import shutil
import random

def prepare_simuletic_dataset():
    """
    Split Simuletic dataset into train/val and create data.yaml
    """
    print("Simuletic Aggressive Poses 데이터셋 준비 중...")
    print("=" * 70)

    source_dir = "/home/yjy/YOLOv26n_violence/kaggle_datasets/Aggressive_Poses_Dataset"
    output_dir = "/home/yjy/YOLOv26n_violence/simuletic_pose_dataset"

    # Get all image files
    image_dir = os.path.join(source_dir, "images")
    label_dir = os.path.join(source_dir, "labels")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    print(f"\n총 이미지 수: {len(image_files)}")

    # Split into train/val (80/20)
    random.seed(42)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"Train: {len(train_files)} images")
    print(f"Val: {len(val_files)} images")

    # Copy files to output directories
    print("\n파일 복사 중...")

    for split, files in [('train', train_files), ('val', val_files)]:
        for img_file in files:
            # Copy image
            shutil.copy(
                os.path.join(image_dir, img_file),
                os.path.join(output_dir, "images", split, img_file)
            )
            # Copy label
            label_file = img_file.replace('.jpg', '.txt')
            shutil.copy(
                os.path.join(label_dir, label_file),
                os.path.join(output_dir, "labels", split, label_file)
            )

        print(f"  {split.capitalize()}: {len(files)} files copied")

    # Create data.yaml
    data_yaml_content = f"""path: {output_dir}
train: images/train
val: images/val

# Classes
nc: 1
names:
  0: person

# Keypoints (COCO format)
kpt_shape: [17, 3]
"""

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)

    print(f"\n✓ data.yaml 생성 완료: {yaml_path}")

    # Verify a sample label
    print("\n라벨 포맷 확인:")
    sample_label = os.path.join(output_dir, "labels", "val", os.path.basename(val_files[0]).replace('.jpg', '.txt'))
    if os.path.exists(sample_label):
        with open(sample_label, 'r') as f:
            first_line = f.readline().strip()
            values = first_line.split()
            print(f"  총 값 개수: {len(values)} (예상: 1 + 4 + 17*3 = 56)")
            print(f"  클래스: {values[0]}")
            print(f"  바운딩 박스: {' '.join(values[1:5])}")
            print(f"  키포인트 (처음 3개): {' '.join(values[5:8])} ...")

    return output_dir

if __name__ == "__main__":
    prepare_simuletic_dataset()
