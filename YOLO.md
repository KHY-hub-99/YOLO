```python
from pathlib import Path
import shutil
import random
import os
from sklearn.model_selection import train_test_split
```

```python
Path('./VOC/labels').mkdir(parents=True, exist_ok=True)
```

```python
!unzip -qq "/content/drive/MyDrive/KDT/분석과제/data/Signpost.zip"
```

```python
# 어노테이션을 YOLO 형식으로 변환
!git clone https://github.com/ssaru/convert2Yolo.git
```

```python
!pip install -r /content/convert2Yolo/requirements.txt
```

```python
!python /content/convert2Yolo/example.py \
    --datasets VOC \
    --img_path /content/images \
    --label /content/annotations \
    --convert_output_path /content//VOC/labels \
    --img_type '.png' \
    --manifest_path /content \
    --cls_list_file /content/voc_s.names
```

    VOC Parsing:   |████████████████████████████████████████| 100.0% (877/877)  Complete


    YOLO Generating:|████████████████████████████████████████| 100.0% (877/877)  Complete


    YOLO Saving:   |████████████████████████████████████████| 100.0% (877/877)  Complete

```python
src = Path("images")          # 원본 images 폴더
dst = Path("VOC") / "images"  # 목적지 VOC/images 폴더
```

```python
Path('VOC/images').mkdir(parents=True, exist_ok=True)
for p in src.iterdir():
    if p.is_file():
        shutil.copy2(p, dst / p.name)
```

```python
image_dir = Path('VOC/images')
label_dir = Path('VOC/labels')

image_files = sorted([p for p in image_dir.iterdir() if p.suffix == '.png'])
label_files = sorted([p for p in label_dir.iterdir() if p.suffix == '.txt'])

print(f"Found {len(image_files)} image files.")
print(f"Found {len(label_files)} label files.")
```

    Found 877 image files.
    Found 877 label files.

```python
mapped_files = []
for img_path in image_files:
    img_name = img_path.stem  # Get filename without extension
    label_path = label_dir / f"{img_name}.txt"
    if label_path.exists():
        mapped_files.append((img_path, label_path))

print(f"Found {len(mapped_files)} image-label pairs.")

```

    Found 877 image-label pairs.

```python
train_files, test_files = train_test_split(mapped_files, test_size=0.2, random_state=42)

print(f"Number of training files: {len(train_files)}")
print(f"Number of testing files: {len(test_files)}")
```

    Number of training files: 701
    Number of testing files: 176

```python
Path('VOC/train/images').mkdir(parents=True, exist_ok=True)
Path('VOC/train/labels').mkdir(parents=True, exist_ok=True)
Path('VOC/test/images').mkdir(parents=True, exist_ok=True)
Path('VOC/test/labels').mkdir(parents=True, exist_ok=True)

print("Created directories: VOC/train/images, VOC/train/labels, VOC/test/images, VOC/test/labels")
```

    Created directories: VOC/train/images, VOC/train/labels, VOC/test/images, VOC/test/labels

```python
for img_path, label_path in train_files:
    shutil.copy2(img_path, Path('VOC/train/images') / img_path.name)
    shutil.copy2(label_path, Path('VOC/train/labels') / label_path.name)

print(f"Copied {len(train_files)} training image-label pairs to VOC/train.")
```

    Copied 701 training image-label pairs to VOC/train.

```python
for img_path, label_path in test_files:
    shutil.copy2(img_path, Path('VOC/test/images') / img_path.name)
    shutil.copy2(label_path, Path('VOC/test/labels') / label_path.name)

print(f"Copied {len(test_files)} testing image-label pairs to VOC/test.")
```

    Copied 176 testing image-label pairs to VOC/test.

# YOLO 학습

```python
!pip install ultralytics
```

```python
from ultralytics import YOLO
```

```python
model = YOLO('yolov8s.pt')
```

```python
import yaml

# Read class names from voc.names
with open('voc_s.names', 'r') as f:
    class_names = f.read().splitlines()

num_classes = len(class_names)

# Define the YAML content
data_yaml = {
    'path': '/content/VOC', # Adjust if your VOC directory is located elsewhere relative to this file
    'train': 'train/images',
    'val': 'test/images',
    'names': class_names,
    'nc': num_classes
}

# Write the YAML content to a file
with open('custom_s_voc.yaml', 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("custom_s_voc.yaml file created successfully.")
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")
```

    custom_s_voc.yaml file created successfully.
    Number of classes: 4
    Class names: ['trafficlight', 'stop', 'speedlimit', 'crosswalk']

```python
results = model.train(
    data='custom_s_voc.yaml',
    epochs=20,
    batch=32,
    imgsz=640,
    device=0,
    workers=2,
    name='custom_s'
)
print(results)
```

```python
# # 학습된 모델 로드 후 validation
# model = YOLO("/content/runs/detect/custom_s/weights/best.pt")

# # 검증 수행
# results = model.val(
#     data="custom_s_voc.yaml",  # 데이터셋 설정
#     imgsz=640,               # 이미지 크기
#     iou=0.5,                 # IoU 임계값
#     batch=32,                # 배치 크기
#     device=0,                # GPU 사용
#     workers=2,               # 데이터 로드 시 병렬 처리할 워커 수
#     half=True,               # FP16 연산 활성화 (속도 향상)
#     split="val"             # 테스트 데이터셋을 사용
# )
# print(results)
```

# YOLO 테스트

```python
# VOC/custom에 test에 있는 이미지 랜덤하게 5장 뽑기
Path('VOC/custom').mkdir(parents=True, exist_ok=True)

test_images_dir = Path('VOC/test/images')
custom_dir = Path('VOC/custom')

all_test_image_files = sorted([p for p in test_images_dir.iterdir() if p.suffix == '.png'])

num_files_to_copy = min(5, len(all_test_image_files)) # Ensure we don't try to copy more files than exist
selected_files = random.sample(all_test_image_files, num_files_to_copy)

copied_file_names = []
for img_path in selected_files:
    shutil.copy2(img_path, custom_dir / img_path.name)
    copied_file_names.append(img_path.name)

print(f"Successfully copied {len(copied_file_names)} image files to VOC/custom.")
print("Copied files:")
for name in copied_file_names:
    print(f"- {name}")
```

    Successfully copied 5 image files to VOC/custom.

```python
# 사진 주고 분석 검증
model = YOLO("/content/runs/detect/custom_s/weights/best.pt")

# 객체 탐지 수행
results = model.predict(
    source="/content/VOC/custom",  # 테스트 이미지 폴더
    imgsz=640,           # 입력 이미지 크기
    conf=0.25,           # 신뢰도(Confidence) 임계값
    device=0,            # GPU 사용 (CPU 사용 시 "cpu")
    save=True,           # 탐지 결과 저장
    save_txt=True,       # 탐지 결과를 txt 형식으로 저장 (YOLO 포맷)
    save_conf=True       # 탐지된 객체의 신뢰도 점수도 저장
)
print(results)
```

```python
!jupyter nbconvert --to markdown "/content/파일이름.ipynb"
```
