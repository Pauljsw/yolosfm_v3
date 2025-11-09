# 프로젝트 파일 구조 가이드

## 📁 전체 디렉토리 구조

```
yolosfm_v3/
├── data/                           # 데이터 디렉토리
│   ├── rgb/                       # ✅ RGB 이미지 (필수!)
│   │   ├── camera_RGB_0_0.png
│   │   ├── camera_RGB_0_1.png
│   │   └── ...                    # 모든 RGB 이미지
│   │
│   ├── depth/                     # ✅ Depth 이미지 (필수!)
│   │   ├── camera_DPT_0_0.png
│   │   ├── camera_DPT_0_1.png
│   │   └── ...                    # RGB와 매칭되는 Depth 이미지
│   │
│   ├── yolo_masks/                # YOLO 추론 결과 (자동 생성)
│   │   ├── camera_RGB_0_0.json
│   │   ├── camera_RGB_0_1.json
│   │   └── ...
│   │
│   └── sfm/                       # SFM 재구성 결과 (자동 생성)
│       ├── database.db
│       ├── poses.json
│       └── sparse/
│           └── 0/
│               ├── cameras.bin
│               ├── images.bin
│               ├── points3D.bin
│               └── project.ini
│
├── calib/                         # ✅ 카메라 캘리브레이션 (필수!)
│   ├── rgb_camera_info.json      # RGB 카메라 내부 파라미터
│   └── depth_camera_info.json    # Depth 카메라 내부 파라미터
│
├── models/                        # ✅ YOLO 모델 (필수!)
│   └── best.pt                    # 학습된 YOLOv11 모델
│
├── configs/                       # 설정 파일
│   └── simple.yaml                # Simple 파이프라인 설정
│
├── src/                           # 소스 코드
│   ├── pipeline.py                # 메인 파이프라인
│   ├── colmap_sfm.py             # COLMAP wrapper
│   ├── colmap_io.py              # COLMAP binary parser
│   ├── pixel_calibration.py      # Pixel-to-mm calibration
│   ├── point_cloud_overlay.py    # Point cloud overlay
│   ├── measure_cracks_simple.py  # Simple measurement
│   ├── calib_io.py               # Calibration I/O
│   └── utils.py                  # Utilities
│
├── calibration/                   # Pixel calibration 결과 (자동 생성)
│   └── pixel_scales.json
│
├── outputs/                       # 최종 결과 (자동 생성)
│   ├── sfm_masked_cloud.ply      # 마스크 오버레이된 Point Cloud
│   └── measurements.csv          # 측정 결과
│
├── SIMPLE_PIPELINE.md            # 사용 가이드
├── DENSE_SFM_SETUP.md            # Dense SFM 설정 가이드
└── PROJECT_STRUCTURE.md          # 이 파일
```

---

## ✅ 준비해야 할 파일들 (필수!)

### 1. RGB 이미지 (`data/rgb/`)

**위치:** `data/rgb/`

**파일명 형식:**
```
camera_RGB_X_Y.png
```
- `X_Y`: 이미지 인덱스 (예: 0_0, 0_1, 0_2, ...)
- 반드시 `.png` 형식

**예시:**
```
data/rgb/
├── camera_RGB_0_0.png
├── camera_RGB_0_1.png
├── camera_RGB_0_2.png
└── ...
```

**요구사항:**
- 해상도: 3840×2160 (또는 다른 고해상도)
- 포맷: PNG
- 색상: RGB (컬러)
- 개수: 최소 10장, 권장 50장 이상

---

### 2. Depth 이미지 (`data/depth/`)

**위치:** `data/depth/`

**파일명 형식:**
```
camera_DPT_X_Y.png
```
- `X_Y`는 **RGB와 동일한 인덱스**
- `camera_RGB_0_0.png` ↔ `camera_DPT_0_0.png` 매칭!

**예시:**
```
data/depth/
├── camera_DPT_0_0.png    # ↔ camera_RGB_0_0.png
├── camera_DPT_0_1.png    # ↔ camera_RGB_0_1.png
├── camera_DPT_0_2.png    # ↔ camera_RGB_0_2.png
└── ...
```

**요구사항:**
- 해상도: 512×512 (Depth 센서 원본)
- 포맷: PNG (16-bit unsigned)
- 단위: **mm** (밀리미터!)
- 개수: RGB와 동일 (1:1 매칭)

**중요:**
```python
# Depth 값 예시:
pixel_value = 1500  # PNG에 저장된 값
actual_depth = 1500 mm = 1.5 m
```

---

### 3. RGB 카메라 캘리브레이션 (`calib/rgb_camera_info.json`)

**위치:** `calib/rgb_camera_info.json`

**포맷:**
```json
{
  "width": 3840,
  "height": 2160,
  "K": [
    [2000.0, 0.0, 1920.0],
    [0.0, 2000.0, 1080.0],
    [0.0, 0.0, 1.0]
  ],
  "D": [0.1, -0.05, 0.001, 0.002, 0.01],
  "distortion_model": "rational_polynomial"
}
```

**필드 설명:**
- `width`, `height`: 이미지 크기
- `K`: 내부 파라미터 행렬
  - `K[0][0]`: fx (focal length x)
  - `K[1][1]`: fy (focal length y)
  - `K[0][2]`: cx (principal point x)
  - `K[1][2]`: cy (principal point y)
- `D`: 왜곡 계수 (distortion coefficients)
- `distortion_model`: 왜곡 모델 (보통 "rational_polynomial")

**캘리브레이션 방법:**
```bash
# OpenCV calibration tool 사용
# 또는 카메라 제조사 SDK에서 제공
```

---

### 4. Depth 카메라 캘리브레이션 (`calib/depth_camera_info.json`)

**위치:** `calib/depth_camera_info.json`

**포맷:**
```json
{
  "width": 512,
  "height": 512,
  "K": [
    [365.0, 0.0, 256.0],
    [0.0, 365.0, 256.0],
    [0.0, 0.0, 1.0]
  ],
  "D": [0.0, 0.0, 0.0, 0.0, 0.0],
  "distortion_model": "plumb_bob"
}
```

**필드 설명:**
- RGB 캘리브레이션과 동일
- Depth 카메라 해상도 (512×512)
- 보통 왜곡이 적음 (D = [0, 0, ...])

---

### 5. YOLO 모델 (`models/best.pt`)

**위치:** `models/best.pt`

**포맷:** PyTorch 모델 (`.pt`)

**생성 방법:**
```bash
# YOLOv11 학습
yolo train model=yolo11n-seg.pt data=crack_data.yaml epochs=100

# 학습 완료 후:
# runs/segment/train/weights/best.pt → models/best.pt로 복사
```

**요구사항:**
- YOLOv11 Segmentation 모델
- 크랙 클래스 학습 완료
- 파일 크기: 보통 20~100 MB

**클래스:**
```yaml
# data.yaml
names:
  0: crack
```

---

## 🔄 자동 생성되는 파일들

### 1. YOLO 마스크 (`data/yolo_masks/`)

**생성 명령:**
```bash
python -m src.pipeline infer --config configs/simple.yaml
```

**출력:**
```
data/yolo_masks/
├── camera_RGB_0_0.json
├── camera_RGB_0_1.json
└── ...
```

**JSON 포맷:**
```json
{
  "image_id": "camera_RGB_0_0",
  "masks": [
    {
      "class": "crack",
      "score": 0.87,
      "polygon": [[100, 200], [150, 250], ...],
      "bbox": [100, 200, 300, 400]
    }
  ]
}
```

---

### 2. SFM 재구성 (`data/sfm/`)

**생성 명령:**
```bash
python -m src.pipeline sfm --config configs/simple.yaml
```

**출력:**
```
data/sfm/
├── database.db           # COLMAP feature database
├── poses.json            # 파싱된 카메라 포즈
└── sparse/
    └── 0/
        ├── cameras.bin   # 카메라 파라미터
        ├── images.bin    # 이미지 포즈 + track
        ├── points3D.bin  # 3D points
        └── project.ini
```

---

### 3. Pixel Calibration (`calibration/`)

**생성 명령:**
```bash
python -m src.pixel_calibration \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --calib calib/rgb_camera_info.json \
  --output calibration/pixel_scales.json
```

**출력:**
```
calibration/
└── pixel_scales.json
```

**JSON 포맷:**
```json
{
  "camera_RGB_0_0": {
    "valid_samples": 9,
    "min_scale_mm": 0.72,
    "max_scale_mm": 0.78,
    "mean_scale_mm": 0.75,
    "median_scale_mm": 0.75,
    "std_scale_mm": 0.02
  }
}
```

---

### 4. Point Cloud Overlay (`outputs/`)

**생성 명령:**
```bash
python -m src.point_cloud_overlay \
  --sparse-dir data/sfm/sparse/0 \
  --masks-dir data/yolo_masks \
  --output outputs/sfm_masked_cloud.ply
```

**출력:**
```
outputs/
└── sfm_masked_cloud.ply  # 빨간 크랙이 표시된 Point Cloud
```

---

### 5. Measurements (`outputs/`)

**생성 명령:**
```bash
python -m src.measure_cracks_simple \
  --masks-dir data/yolo_masks \
  --pixel-scales calibration/pixel_scales.json \
  --output outputs/measurements.csv
```

**출력:**
```
outputs/
└── measurements.csv
```

**CSV 포맷:**
```csv
image_id,mask_idx,length_mm,width_mm,length_px,width_px,confidence,pixel_scale_mm
camera_RGB_0_0,0,1234.5,2.3,1647,3,0.87,0.75
```

---

## 📝 파일명 규칙

### RGB-Depth 매칭

**규칙:**
```
RGB:   camera_RGB_X_Y.png
Depth: camera_DPT_X_Y.png

X_Y가 동일해야 매칭!
```

**올바른 예:**
```
✅ camera_RGB_0_0.png ↔ camera_DPT_0_0.png
✅ camera_RGB_0_1.png ↔ camera_DPT_0_1.png
✅ camera_RGB_1_5.png ↔ camera_DPT_1_5.png
```

**잘못된 예:**
```
❌ camera_RGB_0_0.png ↔ camera_DPT_0_1.png  (인덱스 불일치)
❌ image_001.png ↔ depth_001.png            (파일명 형식 다름)
❌ RGB_0.jpg ↔ DPT_0.png                    (확장자 다름)
```

---

## 🔍 체크리스트

파이프라인 실행 전 확인:

### ✅ 필수 파일

- [ ] `data/rgb/*.png` - RGB 이미지 (최소 10장)
- [ ] `data/depth/*.png` - Depth 이미지 (RGB와 동일 개수)
- [ ] `calib/rgb_camera_info.json` - RGB 캘리브레이션
- [ ] `calib/depth_camera_info.json` - Depth 캘리브레이션
- [ ] `models/best.pt` - YOLO 모델

### ✅ 파일명 확인

- [ ] RGB와 Depth 파일명 매칭 확인
- [ ] 파일명 형식: `camera_RGB_X_Y.png`, `camera_DPT_X_Y.png`
- [ ] 확장자: 모두 `.png`

### ✅ Depth 단위

- [ ] Depth 이미지 단위: mm (밀리미터)
- [ ] 값 범위: 100~10000 (0.1m ~ 10m)

### ✅ 디렉토리 생성

```bash
# 필수 디렉토리 생성
mkdir -p data/rgb data/depth calib models

# 자동 생성 디렉토리 (미리 만들 필요 없음)
# data/yolo_masks, data/sfm, calibration, outputs
```

---

## 🛠️ 파일 준비 도구

### RGB-Depth 매칭 확인

```bash
# 매칭 확인 스크립트
python -c "
from pathlib import Path

rgb_dir = Path('data/rgb')
depth_dir = Path('data/depth')

rgb_files = sorted(rgb_dir.glob('camera_RGB_*.png'))
depth_files = sorted(depth_dir.glob('camera_DPT_*.png'))

print(f'RGB files: {len(rgb_files)}')
print(f'Depth files: {len(depth_files)}')

# 매칭 확인
for rgb in rgb_files:
    idx = rgb.stem.replace('camera_RGB_', '')
    depth = depth_dir / f'camera_DPT_{idx}.png'
    if not depth.exists():
        print(f'❌ Missing: {depth.name}')
    else:
        print(f'✅ Matched: {rgb.name} ↔ {depth.name}')
"
```

### Depth 단위 확인

```bash
# Depth 값 범위 확인
python -c "
import cv2
import numpy as np
from pathlib import Path

depth_file = Path('data/depth').glob('*.png').__next__()
depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)

print(f'File: {depth_file.name}')
print(f'Min: {depth.min()} (mm)')
print(f'Max: {depth.max()} (mm)')
print(f'Mean: {depth.mean():.1f} (mm)')
print(f'Unit: mm (millimeters)')
"
```

---

## 🚨 자주 하는 실수

### 1. RGB-Depth 불일치
```
❌ data/rgb/img001.png
❌ data/depth/depth001.png

✅ data/rgb/camera_RGB_0_0.png
✅ data/depth/camera_DPT_0_0.png
```

### 2. Depth 단위 혼동
```
❌ Depth = 1.5 (meters로 저장)
✅ Depth = 1500 (millimeters로 저장)
```

### 3. 캘리브레이션 누락
```
❌ calib/ 디렉토리가 비어있음
✅ rgb_camera_info.json, depth_camera_info.json 존재
```

### 4. YOLO 모델 누락
```
❌ models/ 디렉토리가 비어있음
✅ models/best.pt 존재
```

---

## 📚 참고 문서

- **SIMPLE_PIPELINE.md**: 전체 파이프라인 사용 가이드
- **DENSE_SFM_SETUP.md**: Dense SFM 설정 (선택)
- **configs/simple.yaml**: 설정 파일 예시

---

**준비 완료되면 파이프라인 실행!** 🚀

```bash
python -m src.pipeline sfm --config configs/simple.yaml
```
