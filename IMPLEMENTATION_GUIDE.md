# YOLO + SFM 3D Fusion Pipeline - Complete Implementation Guide

## 📋 Overview

This pipeline implements a complete workflow for 3D defect detection and measurement from RGB-D images:

**Phase 0**: Coordinate system and unit standardization
**Phase 1**: Depth-only ground truth reconstruction (absolute scale)
**Phase 2**: SFM scale alignment (Umeyama algorithm)
**Phase 3**: RGB-Depth alignment
**Phase 4**: YOLO segmentation inference
**Phase 5**: 3D label fusion (log-odds voting)
**Phase 6**: Instance merging (DBSCAN + IoU)
**Phase 7**: Geometric measurement and reporting

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify Open3D installation (required for Phase 1-2)
python -c "import open3d as o3d; print(o3d.__version__)"

# Verify COLMAP installation
colmap --version
```

### 2. Data Preparation

Place your data following this structure:

```
data/
├── rgb/
│   ├── camera_RGB_0_0.png
│   ├── camera_RGB_0_1.png
│   └── ...
├── depth/
│   ├── camera_DPT_0_0.png  (matching RGB files)
│   ├── camera_DPT_0_1.png
│   └── ...
└── sfm/  (will be created automatically)
```

**File naming convention**: `camera_RGB_X_Y.png` and `camera_DPT_X_Y.png` where X_Y matches.

### 3. Camera Calibration

Update calibration files with your camera parameters:

- `calib/rgb_camera_info.json`
- `calib/depth_camera_info.json`

**For Orbbec Femto Bolt**:
- RGB: 3840×2160
- Depth: 512×512
- Depth unit: Usually mm (auto-detected)

### 4. YOLO Model

Place your trained YOLO segmentation model:
```
models/best.pt
```

### 5. Run Full Pipeline

```bash
python -m src.pipeline full --config configs/default.yaml --log-level INFO
```

This will execute all phases automatically!

---

## 📊 Pipeline Phases Explained

### Phase 0: Coordinate System Setup

**Purpose**: Establish consistent coordinate frames and units across all stages.

**Key Decisions**:
- Pose representation: `T_world_cam` (camera-to-world)
- Distance unit: meters (m)
- Domain: RGB camera domain

**Automatic**:
- RGB-Depth pair matching via filename parsing
- Depth unit auto-detection (mm or m)

---

### Phase 1: Depth-only Ground Truth Reconstruction

**Purpose**: Generate absolute-scale 3D model from depth images to serve as ground truth for SFM scale alignment.

**Method**: TSDF fusion using Open3D

**Why needed?**
```
Problem: SFM (COLMAP) has scale ambiguity
- RGB-only reconstruction cannot determine absolute scale
- 10m structure might be reconstructed as 1m

Solution: Depth provides absolute metric scale
- Use depth images to build ground truth 3D model
- Align SFM results to this ground truth (Phase 2)
```

**Run separately**:
```bash
python -m src.pipeline depth_gt --config configs/default.yaml
```

**Output**:
- `output_depth_tsdf/fused_pointcloud.ply` - Ground truth point cloud
- `output_depth_tsdf/fused_mesh.ply` - Mesh representation
- `output_depth_tsdf/trajectory.json` - Camera trajectory

**Configuration** (`configs/default.yaml`):
```yaml
depth_reconstruction:
  voxel_size_m: 0.01      # 1cm voxels
  depth_unit: "auto"      # Auto-detect mm or m
  use_icp: false          # Enable for better pose estimation
```

---

### Phase 2: SFM Scale Alignment ⭐ **CRITICAL!**

**Purpose**: Align SFM poses to absolute scale using depth ground truth.

**Method**: Umeyama algorithm (similarity transformation estimation)

**Algorithm**:
```
1. Extract sparse 3D points from COLMAP
2. Load depth ground truth point cloud
3. Match point clouds using FPFH features
4. Estimate similarity transformation: target = s·R·source + t
5. Apply (s, R, t) to all SFM camera poses
```

**Mathematical transformation**:
```
For each SFM pose (R_i, t_i):
  R'_i = R_align @ R_i
  t'_i = s × (R_align @ t_i) + t_align
```

**Run separately**:
```bash
python -m src.pipeline scale_align --config configs/default.yaml
```

**Output**:
- `data/sfm/poses_aligned.json` - Scale-corrected poses
- `data/sfm/alignment_metadata.json` - Alignment statistics

**Quality metrics**:
- Scale factor (should be reasonable, e.g., 0.8-1.2)
- RMSE (should be < 0.1m for good alignment)

**Configuration**:
```yaml
scale_alignment:
  use_feature_matching: true  # FPFH-based correspondence
  max_points: 10000           # Subsample for efficiency
```

---

### Phase 3: RGB-Depth Alignment

**Purpose**: Align depth images (512×512) to RGB resolution (3840×2160).

**Two modes**:

**Mode A: Hardware-aligned** (Recommended for Orbbec)
```yaml
align:
  use_simple_resize: true
```
Fast resize assuming factory-aligned depth.

**Mode B: Full calibration-based**
```yaml
align:
  use_simple_resize: false
```
Proper backprojection → transformation → reprojection.

**Pipeline**:
1. Auto-detect depth unit (mm or m)
2. Backproject depth pixels to 3D
3. (Optional) Transform to RGB frame using extrinsics
4. Project to RGB image coordinates
5. Z-buffer for overlap handling
6. Hole filling + bilateral filtering

---

### Phase 4: YOLO Segmentation

**Purpose**: Detect defects in each RGB image.

**Output format** (`data/yolo_masks/*.json`):
```json
{
  "image_id": "camera_RGB_0_0",
  "masks": [
    {
      "class": "crack",
      "score": 0.85,
      "polygon": [[x1,y1], [x2,y2], ...],
      "instance_id": "camera_RGB_0_0_0000"
    }
  ]
}
```

**Run separately**:
```bash
python -m src.pipeline infer --config configs/default.yaml --reinfer auto
```

---

### Phase 5: 3D Label Fusion

**Purpose**: Project 2D masks to 3D and fuse across multiple views.

**Method**: Log-odds Bayesian fusion

**Pipeline**:
1. Rasterize polygon masks
2. Backproject using aligned depth
3. Transform to world frame using (aligned) poses
4. Accumulate into 3D voxel grid with weighted voting
5. Weights: viewing angle × confidence × distance

**Key feature**: Handles occlusion via Z-buffer and multi-view voting.

---

### Phase 6: Instance Merging

**Purpose**: Merge fragmented defects across views into unified instances.

**Method**:
1. DBSCAN spatial clustering (3D, metric units)
2. IoU-based merging of nearby clusters
3. Connected component analysis

**Parameters**:
```yaml
merge:
  dbscan_eps_voxel_mul: 3.0   # eps = 3 × voxel_size
  dbscan_min_pts: 10
  iou_merge_thresh: 0.3
```

---

### Phase 7: Geometric Measurement

**Purpose**: Compute quantitative measurements for each instance.

**Crack measurements**:
- Length: 3D skeletonization + MST path
- Width: Distance transform to skeleton
- Orientation: PCA of skeleton points

**Area defect measurements**:
- Surface area: Convex hull or voxel count
- Depth: Distance to fitted plane
- Plane normal and offset

**Outputs**:
- `outputs/fused/instances.csv` - Measurement table
- `outputs/fused/A_cloud_labeled.ply` - Labeled point cloud
- `outputs/fused/instances_3d.ply` - Instance-colored cloud
- `outputs/fused/instances_3d.geojson` - GeoJSON format
- `outputs/fused/report.md` - Summary report

---

## 🔧 Configuration Guide

### For Orbbec Femto Bolt

```yaml
# Optimized for Orbbec
depth_reconstruction:
  voxel_size_m: 0.01
  depth_unit: "auto"        # Will detect mm
  use_icp: false

align:
  in_depth_unit: "auto"
  use_simple_resize: true   # Hardware-aligned
  hole_fill: true
  joint_bilateral: true
```

### For Fine Cracks

```yaml
fusion:
  voxel_size_cm: 0.5        # 5mm voxels
  prob_thresh: 0.50

measure:
  crack_min_length_cm: 3.0
```

### For Large Defects

```yaml
fusion:
  voxel_size_cm: 2.0        # 2cm voxels
  prob_thresh: 0.60

merge:
  dbscan_eps_voxel_mul: 5.0  # Larger merging radius
```

---

## 📂 Directory Structure After Full Run

```
yolosfm_v2/
├── data/
│   ├── rgb/
│   ├── depth/
│   ├── yolo_masks/        (Phase 4 output)
│   └── sfm/
│       ├── database.db
│       ├── sparse/0/
│       ├── poses.json          (Phase 0 - SFM)
│       └── poses_aligned.json  (Phase 2 - ⭐ Scale aligned)
├── output_depth_tsdf/     (Phase 1 output)
│   ├── fused_pointcloud.ply   (Ground truth)
│   ├── fused_mesh.ply
│   └── trajectory.json
├── outputs/
│   ├── aligned_depth/     (Phase 3 output)
│   └── fused/             (Phase 5-7 output)
│       ├── A_cloud_labeled.ply
│       ├── instances_3d.ply
│       ├── instances.csv
│       ├── instances_3d.geojson
│       └── report.md
└── ...
```

---

## 🎯 Step-by-Step Manual Execution

For debugging or customization:

```bash
# Phase 0: SFM (if no poses exist)
python -m src.pipeline sfm --config configs/default.yaml

# Phase 1: Depth ground truth
python -m src.pipeline depth_gt --config configs/default.yaml

# Phase 2: Scale alignment ⭐
python -m src.pipeline scale_align --config configs/default.yaml

# Phase 3: Depth-RGB alignment
python -m src.pipeline align --config configs/default.yaml

# Phase 4: YOLO inference
python -m src.pipeline infer --config configs/default.yaml

# Phase 5-7: Fusion, merging, measurement
python -m src.pipeline fuse3d --config configs/default.yaml

# Generate report
python -m src.pipeline report --config configs/default.yaml
```

---

## 🔍 Troubleshooting

### "No RGB-Depth pairs found"
→ Check filename pattern: `camera_RGB_X_Y.png` and `camera_DPT_X_Y.png`

### "SFM scale factor is too large/small"
→ Check depth unit (mm vs m)
→ Verify depth camera calibration
→ Try `use_feature_matching: false` for simpler alignment

### "Low alignment RMSE but wrong scale"
→ Insufficient correspondences (< 100 points)
→ Increase `max_points` or improve overlap

### "Depth alignment produces empty image"
→ Depth unit mismatch
→ Set `in_depth_unit: "mm"` explicitly
→ Check depth value range

### "No voxels after fusion"
→ Check aligned poses are used (poses_aligned.json)
→ Verify YOLO masks exist
→ Lower `prob_thresh`

---

## 📈 Performance Tips

1. **Reduce TSDF voxel size** for faster Phase 1:
   ```yaml
   depth_reconstruction:
     voxel_size_m: 0.02  # 2cm instead of 1cm
   ```

2. **Skip scale alignment** if absolute scale not critical:
   ```bash
   python -m src.pipeline full --skip-scale-align
   ```

3. **Parallel processing** (future enhancement):
   - Process images in batches
   - Use GPU for YOLO inference

---

## 🎓 Key Concepts

### Coordinate Frames

- **Depth camera frame**: Origin at depth sensor
- **RGB camera frame**: Origin at RGB sensor
- **World frame (A)**: Global 3D space from SFM

### Transformations

- `T_d2r`: Depth → RGB (extrinsics)
- `T_cam_world`: Camera → World (SFM poses)
- `T_world_cam`: World → Camera (inverse, stored in poses.json)

### Scale Ambiguity

SFM reconstructs **relative** structure:
```
Scale 1: Camera moves 1m, object is 1m away
Scale 10: Camera moves 10m, object is 10m away
→ Same image sequence, different interpretations!
```

Depth provides **absolute** scale:
```
Depth says: Object is 2.5m away
→ Fixes SFM scale unambiguously
```

---

## 📚 References

- **Umeyama Algorithm**: "Least-squares estimation of transformation parameters"
- **TSDF Fusion**: "A volumetric method for building complex models from range images"
- **FPFH Features**: "Fast Point Feature Histograms (FPFH) for 3D registration"
- **COLMAP**: https://colmap.github.io/

---

## 🎉 Success Criteria

✅ Phase 1 complete: Point cloud has > 100k points
✅ Phase 2 complete: Scale factor 0.8-1.2, RMSE < 0.1m
✅ Phase 3 complete: > 80% valid pixels in aligned depth
✅ Phase 4 complete: YOLO masks for all images
✅ Phase 5 complete: > 1000 labeled voxels
✅ Phase 6 complete: Instances > 0
✅ Phase 7 complete: Measurements in `instances.csv`

---

## 📧 Support

For issues:
1. Check logs with `--log-level DEBUG`
2. Verify each phase outputs exist
3. Review configuration parameters
4. Contact: [Your contact info]

---

**Happy 3D defect detection!** 🚀
