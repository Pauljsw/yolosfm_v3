# YOLO + SFM 3D Fusion Pipeline - Quick Start Guide

## 🚀 5분 안에 시작하기

### 0. COLMAP 설치 (선택 사항)

RGB 이미지만으로 자동 포즈 추출을 원하면:

```bash
# Ubuntu/Linux
sudo apt-get install colmap

# macOS
brew install colmap

# Windows: https://github.com/colmap/colmap/releases
```

**자세한 설치 방법**: `COLMAP_GUIDE.md` 참조

### 1. 프로젝트 구조 확인

```
yolo_sfm_3d_fusion/
├── configs/default.yaml        # 설정 파일
├── src/                        # 소스 코드
├── README.md                   # 상세 문서
├── requirements.txt            # 의존성
├── generate_sample_data.py    # 샘플 데이터 생성기
└── tests/                      # 테스트
```

### 2. 의존성 설치

```bash
cd yolo_sfm_3d_fusion
pip install -r requirements.txt
```

필수 라이브러리:
- numpy, scipy, opencv-python
- scikit-learn, scikit-image
- PyYAML, matplotlib

### 3. 샘플 데이터 생성 (테스트용)

```bash
python generate_sample_data.py --num-images 5
```

생성되는 파일:
- `data/rgb/*.png` - RGB 이미지 (3840×2160)
- `data/depth/*.png` - 깊이 맵 (512×512)
- `data/yolo_masks/*.json` - YOLO 마스크
- `data/sfm/poses.json` - 카메라 포즈
- `calib/*.json` - 카메라 캘리브레이션

### 4. 파이프라인 실행

#### 옵션 A: 전체 파이프라인 한 번에 실행 (COLMAP 자동)

```bash
# RGB만 있으면 자동으로 SFM → Alignment → Fusion
python -m src.pipeline full --config configs/default.yaml
```

#### 옵션 B: SFM부터 단계별 실행

```bash
# 0단계: SFM (포즈 생성)
python -m src.pipeline sfm --config configs/default.yaml

# 1단계: Depth-RGB 정렬
python -m src.pipeline align --config configs/default.yaml

# 2단계: 3D 융합 및 계측
python -m src.pipeline fuse3d --config configs/default.yaml --reinfer auto

# 3단계: 리포트 생성
python -m src.pipeline report --config configs/default.yaml
```

#### 옵션 C: 기존 포즈 사용 (COLMAP 건너뛰기)

```bash
# poses.json이 이미 있으면 SFM 건너뜀
python -m src.pipeline full --config configs/default.yaml
```

### 5. 결과 확인

```bash
ls outputs/fused/
```

출력 파일:
- `A_cloud_labeled.ply` - 라벨링된 3D 포인트 클라우드
- `instances_3d.ply` - 인스턴스별 컬러링된 포인트 클라우드
- `instances.csv` - 계측 결과 테이블
- `instances_3d.geojson` - GeoJSON 형식
- `report.md` - 요약 리포트

### 6. 결과 시각화

PLY 파일을 다음 도구로 열어보세요:
- **CloudCompare** (추천)
- **MeshLab**
- **Open3D Viewer**

CSV 파일은 Excel이나 Python pandas로 분석:

```python
import pandas as pd
df = pd.read_csv('outputs/fused/instances.csv')
print(df[['class_name', 'length_m', 'area_m2', 'mean_confidence']])
```

---

## 🎯 실제 데이터로 사용하기

### 1. 데이터 준비

```
data/
├── rgb/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
├── depth/
│   ├── 000001.png  (same filenames as RGB)
│   ├── 000002.png
│   └── ...
├── yolo_masks/
│   ├── 000001.json
│   ├── 000002.json
│   └── ...
└── sfm/
    └── poses.json
```

### 2. 캘리브레이션 설정

`calib/rgb_camera_info.json`:
```json
{
  "width": 3840,
  "height": 2160,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "D": [k1, k2, p1, p2, k3, k4, k5, k6],
  "distortion_model": "rational_polynomial"
}
```

`calib/depth_camera_info.json`:
```json
{
  "width": 512,
  "height": 512,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "D": [k1, k2, p1, p2, k3],
  "distortion_model": "radial_tangential"
}
```

### 3. YOLO 마스크 형식

`data/yolo_masks/000001.json`:
```json
{
  "image_id": "000001",
  "masks": [
    {
      "class": "crack",
      "score": 0.85,
      "polygon": [[x1,y1], [x2,y2], ...],
      "instance_id": "i_000001_0"
    }
  ]
}
```

### 4. SFM 포즈 형식

`data/sfm/poses.json`:
```json
{
  "000001.png": {
    "filename": "000001.png",
    "R": [[r11,r12,r13], [r21,r22,r23], [r31,r32,r33]],
    "t": [tx, ty, tz]
  }
}
```

### 5. 설정 조정

`configs/default.yaml`에서 다음을 조정:

```yaml
fusion:
  voxel_size_cm: 1.0        # 큰 결함: 2-5cm, 작은 균열: 0.3-1cm
  prob_thresh: 0.55          # 신뢰도 임계값

merge:
  dbscan_eps_voxel_mul: 3.0  # DBSCAN 반경 (보셀 크기의 배수)
  iou_merge_thresh: 0.3      # 병합 IoU 임계값

reinfer:
  mode: auto                 # off/on/auto 선택
```

---

## 📊 주요 파라미터 가이드

### 보셀 크기 선택

| 결함 종류 | 권장 보셀 크기 | 설정 값 |
|---------|------------|--------|
| 미세 균열 | 2-5mm | `voxel_size_cm: 0.3` |
| 일반 균열 | 5-10mm | `voxel_size_cm: 0.5` |
| 큰 균열 | 1-2cm | `voxel_size_cm: 1.0` |
| 박리/백태 | 2-5cm | `voxel_size_cm: 3.0` |

### DBSCAN 파라미터

- `eps` = `voxel_size` × `dbscan_eps_voxel_mul`
- 일반적으로 2-5배가 적당
- 너무 작으면: 과도한 분할
- 너무 크면: 과도한 병합

### 재추론 모드

- **off**: 재추론 없음 (가장 빠름)
- **on**: 모든 타일 재추론 (가장 정확, 느림)
- **auto**: 품질 기준으로 선택적 재추론 (권장)

---

## 🔧 문제 해결

### "No valid depth values"
→ `configs/default.yaml`에서 `in_depth_unit: "mm"` 확인

### "Fusion result empty"
→ YOLO 마스크 파일 이름이 RGB 이미지와 일치하는지 확인

### "Poor alignment quality"
→ 카메라 캘리브레이션 파라미터 재확인

### "Too many small instances"
→ `dbscan_eps_voxel_mul`을 3.0→5.0으로 증가

### "Missing measurements"
→ `crack_min_length_cm`, `area_min_m2` 임계값 확인

---

## 📚 추가 리소스

- **상세 문서**: `README.md`
- **테스트 실행**: `pytest tests/`
- **모듈별 문서**: 각 `.py` 파일 docstring 참조

---

## 💡 팁

1. **처음엔 샘플 데이터로 테스트**: 전체 워크플로우 이해
2. **작은 데이터셋으로 시작**: 3-5장 이미지로 파라미터 튜닝
3. **로그 레벨 조정**: `--log-level DEBUG`로 상세 진행 상황 확인
4. **시각화 확인**: 각 단계 후 PLY 파일 열어서 품질 검증
5. **파라미터 실험**: 동일 데이터로 여러 설정 비교

---

## 🎓 워크플로우 요약

```
RGB + Depth + YOLO Masks + SFM Poses
           ↓
    [1. Alignment]
  Aligned Depth (3840×2160)
           ↓
   [2. Projection]
  3D Points in A-frame
           ↓
    [3. Fusion]
  Voxel Grid with Probabilities
           ↓
    [4. Merging]
  Unified Instances
           ↓
  [5. Measurement]
  Length/Area/Direction
           ↓
    [6. Export]
  PLY + CSV + GeoJSON + Report
```

---

**시작하기**: `python -m src.pipeline full --config configs/default.yaml`

**문의**: 문제가 발생하면 `--log-level DEBUG`로 실행 후 로그 확인
