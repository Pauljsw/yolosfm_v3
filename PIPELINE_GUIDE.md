# YOLO+SfM 3D Defect Detection Pipeline - Complete Guide

## 📋 Pipeline Overview

8-Phase 파이프라인으로 RGB-D 이미지에서 3D 균열 측정까지 완전 자동화

```
Phase 0: SfM (COLMAP)          → Camera poses in arbitrary scale
Phase 1: Depth GT (TSDF)       → Absolute scale ground truth from depth
Phase 2: Scale Alignment       → Align SfM to depth scale
Phase 3: Depth-RGB Alignment   → Align depth maps to RGB resolution
Phase 4: YOLO Inference        → 2D defect segmentation
Phase 5: 3D Fusion             → Fuse 2D masks into 3D voxel grid
Phase 6: Instance Merge        → Merge fragmented instances
Phase 7: Measurement           → Crack length/width, area, orientation
Phase 8: Export                → PLY, CSV, GeoJSON, Markdown report
```

---

## 🚀 Quick Start: Full Pipeline Execution

### 최소 필요 데이터 (Phase 0 시작 시)

```
yolosfm_v3/
├── data/
│   ├── rgb/                     # RGB images (3840×2160, PNG/JPG)
│   │   ├── camera_RGB_<timestamp>.png
│   │   └── ...
│   └── depth/                   # Depth images (512×512, PNG, mm 단위)
│       ├── camera_DPT_<timestamp>.png
│       └── ...
├── calib/
│   ├── rgb_camera_info.json     # RGB 카메라 내부 파라미터
│   ├── depth_camera_info.json   # Depth 카메라 내부 파라미터
│   └── extrinsic_depth_to_color.json  # Depth→RGB 외부 파라미터
├── models/
│   └── best.pt                  # YOLO 학습된 모델
└── configs/
    └── default.yaml             # 파이프라인 설정
```

**파일명 규칙**:
- RGB: `camera_RGB_<timestamp>.png` (예: `camera_RGB_1758853283_533442048.png`)
- Depth: `camera_DPT_<timestamp>.png` (예: `camera_DPT_1758853283_533442048.png`)
- **Timestamp 부분이 정확히 일치**해야 RGB-Depth 페어링 가능

### 전체 실행 (한 번에)

```bash
# 모든 Phase 자동 실행
python -m src.pipeline all --config configs/default.yaml
```

### 단계별 실행 (권장 - 디버깅 용이)

```bash
# Phase 0: SfM
python -m src.pipeline sfm --config configs/default.yaml

# Phase 1: Depth Ground Truth
python -m src.pipeline depth_gt --config configs/default.yaml

# Phase 2: Scale Alignment
python -m src.pipeline scale_align --config configs/default.yaml

# Phase 3: Depth-RGB Alignment
python -m src.pipeline align --config configs/default.yaml

# Phase 4: YOLO Inference
python -m src.pipeline detect --config configs/default.yaml

# Phase 5-8: Fusion, Merge, Measurement, Export
python -m src.pipeline fusion --config configs/default.yaml
```

---

## 📊 Phase별 상세 설명

### Phase 0: Structure from Motion (SfM)

**목적**: RGB 이미지에서 카메라 포즈 추정 (임의 스케일)

**입력**:
- `data/rgb/camera_RGB_*.png` - RGB 이미지들
- `calib/rgb_camera_info.json` - RGB 카메라 내부 파라미터

**출력**:
- `data/sfm/poses.json` - 카메라 포즈 (SfM 스케일, 임의)
- `data/sfm/sparse/` - COLMAP sparse reconstruction

**실행**:
```bash
python -m src.pipeline sfm --config configs/default.yaml
```

**핵심 파라미터** (`configs/default.yaml`):
```yaml
sfm:
  camera_model: OPENCV          # PINHOLE, OPENCV, RADIAL
  quality: high                 # low, medium, high, extreme
  use_gpu: true
  dense: false
```

**파일명 규칙**:
- 입력: `camera_RGB_<timestamp>.png`
- 출력: `poses.json` - Key는 `camera_RGB_<timestamp>.png`

---

### Phase 1: Depth Ground Truth (TSDF Reconstruction)

**목적**: Depth 이미지만으로 절대 스케일 3D 모델 생성

**입력**:
- `data/depth/camera_DPT_*.png` - Depth 이미지들 (mm 단위)
- `calib/depth_camera_info.json` - Depth 카메라 내부 파라미터

**출력**:
- `output_depth_tsdf/depth_gt.ply` - TSDF 복원된 3D 포인트 클라우드 (절대 스케일)
- `output_depth_tsdf/odometry.json` - ICP odometry 결과

**실행**:
```bash
python -m src.pipeline depth_gt --config configs/default.yaml
```

**핵심 파라미터**:
```yaml
depth_reconstruction:
  tsdf_voxel_size: 0.01         # TSDF 복셀 크기 (m) - 작을수록 상세
  depth_trunc: 10.0             # 최대 유효 깊이 (m)
  use_icp: true                 # ICP odometry 사용
  depth_unit: "auto"            # mm/m/auto
```

**파일명 규칙**:
- 입력: `camera_DPT_<timestamp>.png`
- 출력: `depth_gt.ply` (단일 파일)

---

### Phase 2: SfM Scale Alignment

**목적**: SfM 포즈를 Depth GT 스케일로 정렬

**입력**:
- `data/sfm/poses.json` - SfM 포즈 (임의 스케일)
- `output_depth_tsdf/depth_gt.ply` - Depth GT (절대 스케일)
- `data/rgb/camera_RGB_*.png` - RGB 이미지들
- `calib/rgb_camera_info.json`

**출력**:
- `data/sfm/poses_aligned.json` - 스케일 정렬된 포즈 (절대 스케일)
- `data/sfm/alignment_info.json` - 정렬 메타데이터 (scale factor, RMSE 등)

**실행**:
```bash
python -m src.pipeline scale_align --config configs/default.yaml
```

**핵심 파라미터**:
```yaml
scale_alignment:
  use_camera_trajectory: true   # 카메라 궤적 정렬 (가장 robust)
  use_feature_matching: true    # FPFH feature 백업
  max_points: 10000             # 서브샘플링 포인트 수
```

**파일명 규칙**:
- 입력: `poses.json`
- 출력: `poses_aligned.json` (이후 모든 Phase에서 사용)

---

### Phase 3: Depth-to-RGB Alignment

**목적**: Depth 이미지(512×512)를 RGB 해상도(3840×2160)로 정렬

**입력**:
- `data/depth/camera_DPT_*.png` - Depth 이미지들
- `data/rgb/camera_RGB_*.png` - RGB 이미지들 (Dense 모드용)
- `calib/depth_camera_info.json`
- `calib/rgb_camera_info.json`
- `calib/extrinsic_depth_to_color.json` - Depth→RGB 변환

**출력**:
- `outputs/aligned_depth/camera_DPT_<timestamp>.png` - 정렬된 depth (3840×2160, mm)

**실행**:
```bash
python -m src.pipeline align --config configs/default.yaml
```

**핵심 파라미터** (Sparse vs Dense):

**Sparse Mode (측정값만, 30-50% coverage)**:
```yaml
align:
  use_simple_resize: false      # 기하학적 정렬
  splat_mode: "bilinear"        # 서브픽셀 스플랫
  undistort_depth: false        # SDK가 이미 보정했으면 false
  hole_fill: false              # 구멍 안 채움 (순수 측정값만)
  do_dense: false               # Dense completion OFF
  plane_fill: false
```

**Dense Mode (100% coverage, confidence map 포함)**:
```yaml
align:
  use_simple_resize: false
  splat_mode: "bilinear"
  undistort_depth: false
  hole_fill: false              # Dense가 알아서 채움
  do_dense: true                # ✅ Dense completion ON
  plane_fill: false             # 선택 (평면 씬에서만)
  bilateral_d: 9
  bilateral_sigma_color: 75
  bilateral_sigma_space: 75
```

**파일명 규칙**:
- 입력: `camera_DPT_<timestamp>.png`
- 출력: `camera_DPT_<timestamp>.png` (동일 파일명, 다른 디렉토리)

---

### Phase 4: YOLO Inference

**목적**: RGB 이미지에서 2D 균열 마스크 추출

**입력**:
- `data/rgb/camera_RGB_*.png` - RGB 이미지들
- `models/best.pt` - YOLO 모델
- `data/sfm/poses_aligned.json` - 포즈 (어떤 이미지 처리할지 결정)

**출력**:
- `data/yolo_masks/camera_RGB_<timestamp>.json` - 마스크 JSON
  ```json
  {
    "image_path": "...",
    "detections": [
      {
        "class_name": "crack",
        "confidence": 0.85,
        "mask": [[u1,v1], [u2,v2], ...],  // polygon
        "bbox": [x, y, w, h]
      }
    ]
  }
  ```

**실행**:
```bash
python -m src.pipeline detect --config configs/default.yaml
```

**핵심 파라미터**:
```yaml
yolo:
  weights: models/best.pt
  conf: 0.20                    # Confidence threshold
  iou: 0.45                     # NMS IoU threshold
  img_size: 1280
  max_det: 300
```

**파일명 규칙**:
- 입력: `camera_RGB_<timestamp>.png`
- 출력: `camera_RGB_<timestamp>.json`

---

### Phase 5: 3D Fusion

**목적**: 2D 마스크를 3D 복셀 그리드로 융합 (Bayesian log-odds)

**입력**:
- `data/yolo_masks/camera_RGB_*.json` - 마스크들
- `outputs/aligned_depth/camera_DPT_*.png` - 정렬된 depth
- `data/sfm/poses_aligned.json` - 카메라 포즈
- `calib/rgb_camera_info.json`

**출력**:
- `outputs/fused/voxel_grid.npz` - 3D 복셀 그리드
- `outputs/fused/A_cloud_labeled.ply` - 라벨링된 3D 포인트 클라우드

**실행**:
```bash
python -m src.pipeline fusion --config configs/default.yaml
```

**핵심 파라미터** (균열 측정 최적화):
```yaml
fusion:
  voxel_size_cm: 0.1            # Coarse voxel 크기 (cm)
  voxel_size_mm_crack: 3.0      # ✅ Crack ROI fine voxel (mm) - 작을수록 정밀
  prob_thresh: 0.05             # 확률 임계값
  weight:
    angle_cos_min: 0.3          # 최소 viewing angle
    conf_min: 0.2               # 최소 YOLO confidence
    distance_decay_sigma: 2.0   # 거리 가중치 감쇠 (m)
```

**파일명 규칙**:
- 입력: `camera_RGB_<timestamp>.json` + `camera_DPT_<timestamp>.png` (timestamp 매칭)
- 출력: 통합 파일들 (파일명에 timestamp 없음)

---

### Phase 6: Instance Merge

**목적**: 여러 뷰에서 분할된 균열 인스턴스를 병합

**입력**:
- `outputs/fused/A_cloud_labeled.ply` - 라벨링된 포인트 클라우드

**출력**:
- `outputs/fused/instances_3d.ply` - 병합된 인스턴스
- `outputs/fused/instances.csv` - 인스턴스 메타데이터

**실행**:
```bash
# fusion 명령에 포함됨
python -m src.pipeline fusion --config configs/default.yaml
```

**핵심 파라미터** (균열 중복 제거 최적화):
```yaml
merge:
  dbscan_eps_voxel_mul: 3.0     # ✅ DBSCAN epsilon (복셀 크기의 배수)
                                 # 작을수록: 균열을 더 세밀하게 분리
                                 # 클수록: 더 적극적으로 병합
  dbscan_min_pts: 10            # ✅ 최소 포인트 수 (노이즈 제거)
                                 # 클수록: 작은 noise 제거, 하지만 짧은 균열도 제거될 수 있음
  iou_merge_thresh: 0.3         # ✅ IoU 임계값 (같은 균열 판정)
                                 # 높을수록: 보수적 병합 (중복 많이 남음)
                                 # 낮을수록: 적극적 병합 (과병합 위험)
  skeleton_gap_thresh_cm: 2.0   # Skeleton gap 임계값 (균열 연결 판정)
```

**균열 측정용 권장 설정**:
```yaml
merge:
  dbscan_eps_voxel_mul: 2.0     # 좀 더 보수적 병합 (균열 분리)
  dbscan_min_pts: 15            # 노이즈 제거 강화
  iou_merge_thresh: 0.25        # 적극적 병합 (중복 최소화)
  skeleton_gap_thresh_cm: 1.5   # 근처 균열만 연결
```

---

### Phase 7: Measurement

**목적**: 균열 길이/폭, 면적 등 정량적 측정

**입력**:
- `outputs/fused/instances_3d.ply` - 병합된 인스턴스
- `outputs/fused/instances.csv`

**출력**:
- `outputs/fused/measurements.csv` - 측정값들
  ```csv
  instance_id,class,length_m,width_m,area_m2,orientation_deg,...
  crack_001,crack,1.234,0.003,0.00245,45.6,...
  ```

**실행**:
```bash
# fusion 명령에 포함됨
python -m src.pipeline fusion --config configs/default.yaml
```

**핵심 파라미터** (균열 측정 품질):
```yaml
measure:
  crack_skeleton_smooth: true   # ✅ Skeleton 스무딩 (노이즈 제거)
  crack_min_length_cm: 5.0      # ✅ 최소 균열 길이 (cm)
                                 # 이보다 짧은 균열 무시 (노이즈 제거)
  area_min_m2: 0.001            # 최소 면적 (m²)
  export_formats: ["ply", "geojson", "csv"]
```

**균열 폭 측정용 권장 설정**:
```yaml
measure:
  crack_skeleton_smooth: true   # 스무딩 ON (깔끔한 skeleton)
  crack_min_length_cm: 3.0      # 3cm 이상만 (작은 노이즈 제외)
  area_min_m2: 0.0005           # 0.5cm² 이상
```

---

### Phase 8: Export

**목적**: 결과를 여러 포맷으로 출력

**출력**:
- `outputs/fused/A_cloud_labeled.ply` - 전체 라벨링된 포인트 클라우드
- `outputs/fused/instances_3d.ply` - 인스턴스별 포인트 클라우드
- `outputs/fused/instances.csv` - 인스턴스 메타데이터
- `outputs/fused/instances_3d.geojson` - GeoJSON (GIS용)
- `outputs/fused/report.md` - Markdown 리포트

**실행**:
```bash
# fusion 명령에 포함됨
python -m src.pipeline fusion --config configs/default.yaml
```

---

## 🎯 Dense Mode + 균열 측정 최적화 설정

### `configs/default.yaml` 전체 (Dense + 균열 측정용)

```yaml
paths:
  rgb_dir: data/rgb
  depth_dir: data/depth
  masks_dir: data/yolo_masks
  sfm_dir: data/sfm
  depth_gt_dir: output_depth_tsdf
  calib_rgb: calib/rgb_camera_info.json
  calib_depth: calib/depth_camera_info.json
  out_dir: outputs

yolo:
  weights: models/best.pt
  data_config: yolo_data.yaml
  conf: 0.20                    # ✅ Crack detection threshold
  iou: 0.45
  img_size: 1280
  device: null
  max_det: 300

sfm:
  camera_model: OPENCV
  quality: high                 # ✅ High quality for crack detection
  use_gpu: true
  dense: false

# Phase 1: Depth Ground Truth
depth_reconstruction:
  tsdf_voxel_size: 0.01         # ✅ 1cm voxel (균열 디테일 유지)
  tsdf_trunc_factor: 4.0
  depth_trunc: 10.0
  depth_unit: "auto"
  use_icp: true
  icp_voxel_size: 0.02
  icp_max_corr_dist: 0.05
  icp_max_iterations: 50
  use_undistortion: false

# Phase 2: Scale Alignment
scale_alignment:
  use_camera_trajectory: true
  use_feature_matching: true
  max_points: 10000

# Phase 3: Depth-RGB Alignment (DENSE MODE)
align:
  in_depth_unit: "auto"
  use_simple_resize: false
  splat_mode: "bilinear"        # ✅ Sub-pixel accuracy
  undistort_depth: false
  hole_fill: false
  joint_bilateral: false
  do_dense: true                # ✅✅✅ DENSE MODE ON
  plane_fill: false
  bilateral_d: 9
  bilateral_sigma_color: 75
  bilateral_sigma_space: 75

# Phase 5: 3D Fusion (균열 측정 최적화)
fusion:
  voxel_size_cm: 0.1            # Coarse grid
  voxel_size_mm_crack: 3.0      # ✅ 3mm fine voxel for cracks (균열 디테일)
  prob_thresh: 0.05
  weight:
    angle_cos_min: 0.3
    conf_min: 0.2
    distance_decay_sigma: 2.0

# Phase 6: Instance Merge (중복 최소화)
merge:
  dbscan_eps_voxel_mul: 2.0     # ✅ 보수적 병합 (균열 분리)
  dbscan_min_pts: 15            # ✅ 노이즈 강력 제거
  iou_merge_thresh: 0.25        # ✅ 적극적 병합 (중복 제거)
  skeleton_gap_thresh_cm: 1.5   # ✅ 근처 균열만 연결

# Reinference (선택)
reinfer:
  mode: auto                    # off | on | auto
  tile_px: 1280
  tile_overlap: 0.1
  uv_resolution_mm_px: 2.0
  triggers:
    gap_cm: 8.0
    conflict_rate: 0.15
    mean_conf: 0.45
    scale_std_ratio: 0.2

# Phase 7: Measurement (균열 길이/폭 측정)
measure:
  crack_skeleton_smooth: true   # ✅ 스무딩 ON
  crack_min_length_cm: 3.0      # ✅ 3cm 이상만 측정
  area_min_m2: 0.0005           # ✅ 0.5cm² 이상
  export_formats: ["ply", "geojson", "csv"]

# Class definitions
classes:
  crack: 0
  efflorescence: 1
  detachment: 2
  leak: 3
  spalling: 4
  material separation: 5
  rebar: 6
  damage: 7
  exhilaration: 8

colors:
  crack: [255, 0, 0]
  efflorescence: [0, 255, 0]
  detachment: [0, 0, 255]
  leak: [255, 255, 0]
  spalling: [255, 0, 255]
  material separation: [0, 255, 255]
  rebar: [255, 128, 0]
  damage: [128, 0, 255]
  exhilaration: [0, 128, 255]
```

---

## 🔍 파일명 대응 구조 정리

### RGB-Depth 페어링

**필수**: Timestamp가 정확히 일치해야 함

```
data/rgb/camera_RGB_1758853283_533442048.png
data/depth/camera_DPT_1758853283_533442048.png
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           이 부분이 동일해야 페어링됨
```

### Phase별 파일명 변환

```
Phase 0 (SfM):
  Input:  camera_RGB_<timestamp>.png
  Output: poses.json (key: "camera_RGB_<timestamp>.png")

Phase 1 (Depth GT):
  Input:  camera_DPT_<timestamp>.png
  Output: depth_gt.ply (단일 파일)

Phase 2 (Scale Align):
  Input:  poses.json
  Output: poses_aligned.json

Phase 3 (Alignment):
  Input:  camera_DPT_<timestamp>.png
  Output: outputs/aligned_depth/camera_DPT_<timestamp>.png
          (동일 파일명, 다른 디렉토리)

Phase 4 (YOLO):
  Input:  camera_RGB_<timestamp>.png
  Output: data/yolo_masks/camera_RGB_<timestamp>.json
          (동일 파일명, 확장자만 .json)

Phase 5-8 (Fusion):
  Input:  camera_RGB_<timestamp>.json + camera_DPT_<timestamp>.png
          (timestamp 매칭으로 페어링)
  Output: instances_3d.ply, measurements.csv (통합 파일)
```

---

## ⚠️ 주의사항

### 1. 파일명 규칙 엄수

```bash
# ✅ 올바른 예
camera_RGB_1758853283_533442048.png
camera_DPT_1758853283_533442048.png

# ❌ 잘못된 예 (timestamp 불일치)
camera_RGB_1758853283_533442048.png
camera_DPT_1758853283_533442049.png  # ← 마지막 숫자 다름!

# ❌ 잘못된 예 (prefix 틀림)
rgb_1758853283_533442048.png         # camera_RGB_ 필수
depth_1758853283_533442048.png       # camera_DPT_ 필수
```

### 2. Depth 단위

- **입력**: mm 단위 (PNG 16-bit)
- **내부 처리**: m 단위로 자동 변환
- **출력**: mm 단위 (PNG 16-bit)

### 3. Dense Mode 성능

- **메모리**: RGB 이미지 로딩으로 메모리 사용량 2배 증가
- **속도**: JBU completion으로 약 1.5배 느림
- **품질**: 100% coverage, confidence map 포함

### 4. 균열 측정 품질 체크리스트

- [ ] `voxel_size_mm_crack`: 3mm (균열 디테일)
- [ ] `dbscan_min_pts`: 15+ (노이즈 제거)
- [ ] `iou_merge_thresh`: 0.2-0.3 (중복 제거)
- [ ] `crack_min_length_cm`: 3-5cm (작은 노이즈 무시)
- [ ] `crack_skeleton_smooth`: true (깔끔한 측정)

---

## 🛠️ Troubleshooting

### Q1: "No RGB-Depth pairs found"

**원인**: Timestamp 불일치

**해결**:
```bash
# 파일명 확인
ls data/rgb/ | head -3
ls data/depth/ | head -3

# Timestamp 일치 확인
python -c "
import os
rgb_files = {f.replace('camera_RGB_', '').replace('.png', '') for f in os.listdir('data/rgb')}
dpt_files = {f.replace('camera_DPT_', '').replace('.png', '') for f in os.listdir('data/depth')}
print('Matched:', len(rgb_files & dpt_files))
print('RGB only:', len(rgb_files - dpt_files))
print('Depth only:', len(dpt_files - rgb_files))
"
```

### Q2: "Coverage too low (< 30%)"

**원인**: Extrinsics가 틀렸거나 depth 단위 문제

**해결**:
1. `undistort_depth: true` 시도
2. Validation 실행:
   ```bash
   python -m src.validate_alignment --calib-dir calib --num-samples 5
   ```
3. Edge overlap 60% 이상인지 확인

### Q3: "Too many small crack instances"

**원인**: `dbscan_min_pts` 너무 작음

**해결**:
```yaml
merge:
  dbscan_min_pts: 20  # 15 → 20으로 증가
  crack_min_length_cm: 5.0  # 3 → 5로 증가
```

---

## 📚 추가 자료

- **Validation Tool**: `src/validate_alignment.py` - Alignment 품질 검증
- **Calibration Extraction**: `calib_extraction/` - Orbbec SDK 캘리브레이션 추출
- **README**: 전체 프로젝트 개요 및 설치 가이드
