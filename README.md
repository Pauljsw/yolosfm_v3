# YOLO + SFM 3D Defect Fusion Pipeline

## Overview

This project implements a **complete end-to-end pipeline** for 3D defect detection and measurement from RGB-D images. The pipeline fuses 2D YOLO segmentation masks into a 3D global coordinate system using Structure from Motion (SFM) with **absolute scale alignment** via depth ground truth. It handles multiple RGB-D images, merges fragmented defect instances across views, and provides quantitative measurements (length, area, orientation).

### 🆕 Key Features (Updated)

- **Phase 1-2: Absolute Scale Reconstruction** ⭐ **NEW**
  - Depth-only TSDF reconstruction for ground truth
  - Umeyama algorithm for SFM scale alignment
  - Eliminates scale ambiguity in pure visual SFM

- **Robust RGB-D Processing**
  - Auto-detection of depth units (mm/m)
  - Hardware-aligned depth support (Orbbec Femto Bolt)
  - Filename-based RGB-Depth pairing

## Key Features

- **Depth-to-RGB Alignment**: Precise alignment of depth maps (512×512) to RGB resolution (3840×2160)
- **3D Label Fusion**: Probabilistic fusion of YOLO masks into 3D voxel grid using log-odds accumulation
- **Instance Merging**: DBSCAN clustering and IoU-based merging to unify fragmented defects
- **Automatic Measurement**: 
  - Cracks: Length, width, skeleton topology, orientation
  - Area defects: Surface area, depth, plane fitting
- **Multiple Export Formats**: PLY point clouds, CSV tables, GeoJSON, Markdown reports
- **Reinference Support**: Optional tile-based YOLO reinference for quality improvement

## Project Structure

```
yolo_sfm_3d_fusion/
├── configs/
│   └── default.yaml           # Configuration file
├── calib/
│   ├── rgb_camera_info.json   # RGB camera calibration
│   └── depth_camera_info.json # Depth camera calibration
├── data/
│   ├── rgb/                   # RGB images (3840×2160)
│   ├── depth/                 # Depth images (512×512)
│   ├── yolo_masks/            # YOLO mask JSONs
│   └── sfm/
│       └── poses.json         # Camera poses from SFM
├── outputs/
│   ├── aligned_depth/         # Aligned depth maps
│   ├── fused/                 # Fusion results
│   │   ├── A_cloud_labeled.ply
│   │   ├── instances_3d.ply
│   │   ├── instances.csv
│   │   ├── instances_3d.geojson
│   │   └── report.md
│   └── report/
└── src/
    ├── calib_io.py            # Camera calibration I/O
    ├── align_depth_to_rgb.py # Depth alignment
    ├── project_mask_to_A.py  # Mask projection to 3D
    ├── fusion_3d.py           # 3D label fusion
    ├── instance_merge.py      # Instance merging
    ├── measurement.py         # Geometric measurements
    ├── export_results.py      # Result export
    ├── utils.py               # Utilities
    └── pipeline.py            # Main pipeline CLI
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd yolo_sfm_3d_fusion

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- numpy >= 1.21.0
- scipy >= 1.7.0
- opencv-python >= 4.5.0
- scikit-learn >= 0.24.0
- scikit-image >= 0.18.0
- PyYAML >= 5.4.0
- matplotlib >= 3.3.0 (optional, for visualization)

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt

# Verify Open3D (required for Phase 1-2)
python -c "import open3d; print(open3d.__version__)"
```

### 2. Prepare Data

Place your RGB-D data with matched filenames:

```
data/
├── rgb/
│   ├── camera_RGB_0_0.png
│   ├── camera_RGB_0_1.png
│   └── ...
├── depth/
│   ├── camera_DPT_0_0.png  # Matches camera_RGB_0_0.png
│   ├── camera_DPT_0_1.png
│   └── ...
└── (sfm/ and yolo_masks/ will be auto-generated)
```

**Naming convention**: `camera_RGB_X_Y.png` ↔ `camera_DPT_X_Y.png` (X_Y must match)

### 2. Configure Calibration

Create calibration files:

```bash
python -m src.calib_io create-samples
```

Edit `calib/rgb_camera_info.json` and `calib/depth_camera_info.json` with your camera parameters.

### 3. Configure YOLO Model

Place your trained YOLO segmentation model:
```
models/best.pt
```

### 4. Run Full Pipeline

**Automatic (Recommended)**:
```bash
python -m src.pipeline full --config configs/default.yaml
```

This runs all phases automatically:
- Phase 0: SFM pose estimation (COLMAP)
- Phase 1: Depth ground truth reconstruction
- Phase 2: SFM scale alignment ⭐
- Phase 3: Depth-RGB alignment
- Phase 4: YOLO segmentation
- Phase 5-7: 3D fusion, merging, measurement

**Manual step-by-step**:
```bash
# Phase 0: SFM
python -m src.pipeline sfm --config configs/default.yaml

# Phase 1: Depth ground truth
python -m src.pipeline depth_gt --config configs/default.yaml

# Phase 2: Scale alignment ⭐
python -m src.pipeline scale_align --config configs/default.yaml

# Phase 3: Depth-RGB alignment
python -m src.pipeline align --config configs/default.yaml

# Phase 4: YOLO inference
python -m src.pipeline infer --config configs/default.yaml

# Phase 5-7: Fusion + measurement
python -m src.pipeline fuse3d --config configs/default.yaml
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Paths**: Input/output directories
- **Alignment**: Depth unit, hole filling, bilateral filtering
- **Fusion**: Voxel size, probability threshold, weighting
- **Merging**: DBSCAN parameters, IoU threshold
- **Measurement**: Crack smoothing, minimum dimensions
- **Reinference**: Mode (off/on/auto), tile size, quality triggers

Example configuration:

```yaml
paths:
  rgb_dir: data/rgb
  depth_dir: data/depth
  masks_dir: data/yolo_masks
  sfm_dir: data/sfm
  out_dir: outputs

fusion:
  voxel_size_cm: 1.0
  prob_thresh: 0.55
  weight:
    angle_cos_min: 0.3
    conf_min: 0.2

merge:
  dbscan_eps_voxel_mul: 3.0
  dbscan_min_pts: 10
  iou_merge_thresh: 0.3

reinfer:
  mode: auto  # off | on | auto
  triggers:
    gap_cm: 8.0
    conflict_rate: 0.15
    mean_conf: 0.45
```

## Data Formats

### Camera Calibration JSON

```json
{
  "width": 3840,
  "height": 2160,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "D": [k1, k2, p1, p2, k3, k4, k5, k6],
  "distortion_model": "rational_polynomial"
}
```

### Poses JSON (SFM Output)

```json
{
  "000123.png": {
    "filename": "000123.png",
    "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
    "t": [tx, ty, tz]
  }
}
```

### YOLO Masks JSON

```json
{
  "image_id": "000123",
  "masks": [
    {
      "class": "crack",
      "score": 0.78,
      "polygon": [[x1, y1], [x2, y2], ...],
      "instance_id": "i_0001"
    }
  ]
}
```

## Output Files

- **`A_cloud_labeled.ply`**: Labeled voxel point cloud with per-voxel class probabilities
- **`instances_3d.ply`**: Instance-colored point cloud
- **`instances.csv`**: Measurement table with columns:
  - instance_id, class_id, class_name
  - length_m, width_m, area_m2, depth_m
  - mean_confidence, view_count, entropy
  - bbox_min, bbox_max, centroid
- **`instances_3d.geojson`**: GeoJSON format for GIS tools
- **`report.md`**: Summary report with statistics and quality metrics

## Pipeline Stages

### Stage 1: Depth-to-RGB Alignment

Aligns depth images to RGB resolution using camera calibrations:

1. Undistort depth coordinates (optional)
2. Backproject to 3D (depth camera frame)
3. Transform to RGB camera frame (if extrinsics provided)
4. Project to RGB image coordinates
5. Z-buffer to handle overlaps
6. Hole filling and bilateral filtering

**Quality Check**: Valid pixel ratio, plane RMSE

### Stage 2: 3D Label Fusion

Projects YOLO masks to global (A) coordinate system:

1. Rasterize polygon masks
2. Backproject using aligned depth
3. Transform to world frame using SFM poses
4. Accumulate into 3D voxel grid using log-odds fusion
5. Weight by viewing angle, confidence, distance

**Quality Metrics**: Class conflict rate, mean entropy

### Stage 3: Instance Merging

Merges fragmented instances across views:

1. Cluster each class using DBSCAN (3D spatial clustering)
2. Merge nearby clusters based on IoU or minimum distance
3. Compute per-instance statistics

**Parameters**: DBSCAN eps, min_samples, IoU threshold

### Stage 4: Measurement

Computes geometric measurements:

- **Cracks**: 
  - 3D skeletonization
  - Length (total skeleton path)
  - Width (distance to skeleton)
  - Topology (branches, endpoints)
  - Principal orientation (PCA)

- **Area defects** (spalling, efflorescence, exposed rebar):
  - Surface area (convex hull or voxel projection)
  - Depth (distance to fitted plane)
  - Plane normal and offset

### Stage 5: Export

Exports results to multiple formats for analysis and visualization.

## Reinference Mode

**Purpose**: Improve fusion quality by running YOLO again on global orthomaps or tiles.

**Modes:**
- `off`: No reinference
- `on`: Force reinference on all tiles
- `auto`: Reinfer only tiles with quality issues

**Auto Triggers** (configurable):
- Continuity gap > 8 cm
- Class conflict rate > 15%
- Mean confidence < 0.45
- Scale variance > 20%

## Validation

Quality assurance checks at each stage:

1. **Alignment**: Plane residual RMSE < 5mm
2. **Fusion**: Class conflict rate < 15%
3. **Measurement**: Reproducibility error < 1%

## Troubleshooting

### No valid depth values
- Check depth unit (m vs mm) in config
- Verify depth camera calibration

### Low fusion quality
- Increase voxel size for coarse features
- Adjust probability threshold
- Check YOLO mask quality

### Poor instance merging
- Tune DBSCAN eps (typically 2-5× voxel size)
- Adjust IoU threshold
- Check for scale inconsistencies in SFM

### Missing measurements
- Verify class names match config
- Check minimum dimension thresholds
- Ensure sufficient voxel density

## Citation

If you use this pipeline in your research, please cite:

```
@software{yolo_sfm_3d_fusion,
  title = {YOLO + SFM 3D Defect Fusion Pipeline},
  year = {2025},
  author = {Your Name},
  url = {https://github.com/...}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please contact: your.email@example.com
