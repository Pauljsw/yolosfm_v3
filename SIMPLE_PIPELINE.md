# Simple Crack Measurement Pipeline

A streamlined pipeline for crack detection and measurement using:
- **SFM (Sparse)**: Camera poses + Point cloud visualization
- **YOLO**: Crack segmentation
- **Pixel-to-MM**: Depth-based calibration
- **2D Measurement**: Length and width in each image

No voxels, no TSDF, no scale alignment needed!

---

## 🚀 Quick Start

### 1. Run Complete Pipeline

```bash
# Activate environment
conda activate yolov11

# Phase 0: SFM
python -m src.pipeline sfm --config configs/simple.yaml

# Phase 1: YOLO
python -m src.pipeline infer --config configs/simple.yaml

# Phase 2: Pixel Calibration
python -m src.pixel_calibration \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --calib calib/rgb_camera_info.json \
  --output calibration/pixel_scales.json

# Phase 3: Point Cloud Overlay
python -m src.point_cloud_overlay \
  --sparse-dir data/sfm/sparse/0 \
  --masks-dir data/yolo_masks \
  --output outputs/sfm_masked_cloud.ply

# Phase 4: Measurement
python -m src.measure_cracks_simple \
  --masks-dir data/yolo_masks \
  --pixel-scales calibration/pixel_scales.json \
  --output outputs/measurements.csv
```

### 2. Visualize

```bash
# View colored point cloud (crack = red)
cloudcompare.CloudCompare outputs/sfm_masked_cloud.ply

# View measurements
cat outputs/measurements.csv
```

---

## 📊 Pipeline Flow

```
RGB Images → SFM → Sparse Point Cloud
     ↓
  YOLO → Crack Masks
     ↓
RGB + Depth → Pixel-to-MM Calibration
     ↓
Point Cloud + Masks → Colored Visualization (Red cracks)
     ↓
Masks + Pixel-MM → 2D Measurements → CSV
```

---

## 📁 Output Files

```
outputs/
├── sfm_masked_cloud.ply    # Point cloud with red cracks
└── measurements.csv         # Crack measurements

calibration/
└── pixel_scales.json        # Pixel-to-mm scales per image

data/sfm/
├── sparse/0/
│   ├── cameras.bin         # Camera parameters
│   ├── images.bin          # Image poses
│   └── points3D.bin        # 3D points with tracks
└── poses.json              # Parsed poses
```

---

## 📋 measurements.csv Format

```csv
image_id,mask_idx,length_mm,width_mm,length_px,width_px,confidence,pixel_scale_mm
camera_RGB_0_0,0,1234.5,2.3,1647,3,0.87,0.75
camera_RGB_0_1,0,567.8,1.2,756,2,0.82,0.75
```

**Columns:**
- `image_id`: Image identifier
- `mask_idx`: Mask index in that image
- `length_mm`: Crack length in millimeters
- `width_mm`: Crack width in millimeters
- `length_px`: Length in pixels
- `width_px`: Width in pixels
- `confidence`: YOLO confidence score
- `pixel_scale_mm`: Pixel size in mm/pixel

---

## ⚙️ Configuration

Edit `configs/simple.yaml`:

```yaml
sfm:
  dense: false  # Sparse is enough! (See DENSE_SFM_SETUP.md for Dense)
  quality: high

yolo:
  conf: 0.20    # Low threshold to catch all cracks

measurement:
  min_length_mm: 10.0  # Filter cracks shorter than 1cm
```

---

## 🔍 How It Works

### Phase 0: SFM
- Extracts features from RGB images
- Matches features across images
- Estimates camera poses
- Triangulates 3D points
- **Output**: Sparse point cloud + poses

### Phase 1: YOLO
- Segments cracks in each RGB image
- **Output**: Polygon masks JSON

### Phase 2: Pixel Calibration
- Samples depth at multiple locations
- Calculates `pixel_mm = (depth / fx) * 1000`
- Interpolates to full image (considers distortion)
- **Output**: mm/pixel scale for each image

### Phase 3: Point Cloud Overlay
- Reads COLMAP `points3D.bin` (tracks)
- For each 3D point:
  - Checks which images observe it
  - Checks if pixel is in crack mask
  - Colors red if crack, original RGB otherwise
- **Output**: Colored PLY

### Phase 4: Measurement
- For each crack mask:
  - Rasterize polygon
  - Preprocess (blur, CLAHE, threshold)
  - Skeletonize
  - Measure length (MST)
  - Measure width (perpendicular scan)
  - Convert pixels → mm using calibration
- **Output**: CSV table

---

## 🆚 vs Original Pipeline

| Feature | Original | Simple |
|---------|----------|--------|
| **Phases** | 8 | 4 |
| **TSDF** | Yes | No |
| **Voxel Grid** | Yes | No |
| **Scale Align** | Umeyama | Pixel-to-MM |
| **Measurement** | 3D Voxel | 2D Image |
| **Speed** | Slower | Faster |
| **Complexity** | High | Low |
| **Accuracy** | High | Good |

**Use Simple when:**
- ✅ You want quick results
- ✅ You have RGB-D data
- ✅ 2D measurement is sufficient

**Use Original when:**
- ✅ You need high-precision 3D
- ✅ Multi-view fusion is critical
- ✅ You have GPU resources

---

## 🐛 Troubleshooting

**"CUDA not available" error:**
```yaml
sfm:
  dense: false  # ← Make sure this is false!
```
See `DENSE_SFM_SETUP.md` for Dense mode setup.

**"No pixel scales for image X":**
- Check depth images exist
- Verify RGB-Depth filename matching
- Run Phase 2 calibration first

**Measurements seem wrong:**
- Check `pixel_scales.json` values
- Verify depth unit is mm (not meters)
- Check camera calibration

**Point cloud has no red:**
- Verify YOLO masks exist
- Check mask polygons are valid
- Increase YOLO confidence threshold

---

## 📚 File Reference

### New Files

| File | Purpose |
|------|---------|
| `src/colmap_io.py` | Parse COLMAP binary files |
| `src/pixel_calibration.py` | Depth-based pixel-to-mm |
| `src/point_cloud_overlay.py` | Overlay masks on SFM cloud |
| `src/measure_cracks_simple.py` | 2D crack measurement |
| `configs/simple.yaml` | Simple pipeline config |
| `DENSE_SFM_SETUP.md` | Dense mode setup guide |

### Existing Files (Used)

| File | Usage |
|------|-------|
| `src/pipeline.py` | SFM and YOLO execution |
| `src/colmap_sfm.py` | COLMAP wrapper |
| `src/calib_io.py` | Camera calibration I/O |
| `src/utils.py` | Utilities |

---

## 🎯 Next Steps

1. **Run the pipeline** on your data
2. **Check `sfm_masked_cloud.ply`** in CloudCompare
3. **Verify measurements** in CSV
4. **Adjust thresholds** in `configs/simple.yaml`
5. **Share results**!

---

## 💡 Tips

**Better visualization:**
```bash
# Overlay on original RGB point cloud
python -m src.point_cloud_overlay \
  --min-track-length 3 \  # Filter noise
  --crack-color 255 0 0   # Bright red
```

**Better measurements:**
```yaml
measurement:
  min_length_mm: 5.0  # Catch smaller cracks
```

**Faster processing:**
```yaml
sfm:
  quality: medium  # Faster feature extraction
```

---

**Questions? Check `DENSE_SFM_SETUP.md` or open an issue!**
