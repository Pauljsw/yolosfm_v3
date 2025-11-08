# COLMAP Integration Guide

## Overview

COLMAP (COLlaborative MApping Platform)이 프로젝트에 통합되어 **RGB 이미지만으로 자동으로 카메라 포즈를 추출**할 수 있습니다.

## COLMAP 설치

### Ubuntu/Linux

```bash
# Option 1: apt (Ubuntu 18.04+)
sudo apt-get install colmap

# Option 2: Build from source
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j
sudo make install
```

### macOS

```bash
brew install colmap
```

### Windows

1. Download from: https://github.com/colmap/colmap/releases
2. Extract and add to PATH

### Docker (추천)

```bash
docker pull colmap/colmap:latest
```

자세한 설치 방법: https://colmap.github.io/install.html

## 사용 방법

### 1. 자동 SFM (권장)

RGB 이미지만 있으면 자동으로 포즈 추출:

```bash
# 전체 파이프라인 (SFM 포함)
python -m src.pipeline full --config configs/default.yaml

# 또는 SFM만 실행
python -m src.pipeline sfm --config configs/default.yaml
```

**입력**: `data/rgb/*.png` (RGB 이미지들)
**출력**: `data/sfm/poses.json` (카메라 포즈)

### 2. 독립 실행

COLMAP 모듈을 직접 실행:

```bash
python -m src.colmap_sfm \
    --image-dir data/rgb \
    --output-dir data/sfm \
    --poses-json data/sfm/poses.json \
    --quality high
```

### 3. Python API

```python
from colmap_sfm import run_colmap_sfm_auto

poses = run_colmap_sfm_auto(
    image_dir='data/rgb',
    output_dir='data/sfm',
    poses_json_output='data/sfm/poses.json',
    camera_model='OPENCV',  # 카메라 모델
    quality='high',         # low, medium, high, extreme
    dense=False             # Dense reconstruction (선택)
)

print(f"Reconstructed {len(poses)} images")
```

## 설정 옵션

### configs/default.yaml

```yaml
sfm:
  camera_model: OPENCV      # 카메라 모델
  quality: high             # 품질 설정
  use_gpu: true             # GPU 가속 사용
  dense: false              # Dense reconstruction
```

### 카메라 모델

| 모델 | 설명 | 사용 시기 |
|-----|------|---------|
| **OPENCV** | fx, fy, cx, cy + k1, k2, p1, p2 | 일반적인 왜곡 (권장) |
| PINHOLE | fx, fy, cx, cy (왜곡 없음) | 왜곡 보정된 이미지 |
| RADIAL | fx, fy, cx, cy + k1, k2 | 방사 왜곡만 |
| SIMPLE_RADIAL | f, cx, cy + k | 간단한 왜곡 |
| SIMPLE_PINHOLE | f, cx, cy | 정사각형 픽셀, 왜곡 없음 |

### 품질 설정

| 설정 | 이미지 크기 | Feature 수 | 속도 | 정확도 |
|-----|----------|-----------|------|--------|
| low | 1600px | 4K | 빠름 | 낮음 |
| medium | 2400px | 8K | 보통 | 보통 |
| **high** | 3200px | 16K | 느림 | 높음 |
| extreme | 4800px | 32K | 매우 느림 | 매우 높음 |

## 실행 예시

### 예시 1: 기본 SFM

```bash
# RGB 이미지 준비
ls data/rgb/
# 000001.png, 000002.png, ...

# SFM 실행
python -m src.pipeline sfm --config configs/default.yaml

# 결과 확인
cat data/sfm/poses.json
```

### 예시 2: 고품질 SFM

```yaml
# configs/default.yaml
sfm:
  camera_model: OPENCV
  quality: extreme
  use_gpu: true
```

```bash
python -m src.pipeline sfm --config configs/default.yaml
```

### 예시 3: Dense Reconstruction

```yaml
# configs/default.yaml
sfm:
  quality: high
  dense: true  # Dense point cloud 생성
```

```bash
python -m src.pipeline sfm --config configs/default.yaml

# Dense point cloud: data/sfm/dense/fused.ply
```

### 예시 4: 전체 파이프라인

```bash
# RGB만 있으면 자동으로 SFM → Alignment → Fusion
python -m src.pipeline full --config configs/default.yaml
```

## 출력 형식

### poses.json

```json
{
  "000001.png": {
    "filename": "000001.png",
    "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
    "t": [[tx], [ty], [tz]],
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  },
  ...
}
```

- **R**: 카메라-월드 회전 행렬 (camera-to-world)
- **t**: 카메라-월드 평행이동 (camera-to-world)
- **K**: 내부 파라미터 행렬

### COLMAP 원본 파일

SFM 실행 후 `data/sfm/sparse/0/`에 생성:
- `cameras.txt` - 카메라 내부 파라미터
- `images.txt` - 카메라 포즈
- `points3D.txt` - Sparse 3D points

## 성능

### 실행 시간 (예상)

| 이미지 수 | 품질 | GPU | 시간 |
|---------|------|-----|------|
| 10 | medium | Yes | ~2분 |
| 10 | high | Yes | ~5분 |
| 50 | high | Yes | ~15분 |
| 100 | high | Yes | ~30분 |
| 10 | high | No | ~15분 |

### 메모리

- Feature extraction: ~2-4GB
- Matching: ~4-8GB
- Reconstruction: ~2-4GB

## 문제 해결

### "COLMAP not found"

```bash
# 설치 확인
colmap --version

# PATH에 추가 (Linux/Mac)
export PATH=$PATH:/path/to/colmap

# Windows
set PATH=%PATH%;C:\path\to\colmap
```

### "Reconstruction failed"

**원인**:
- 이미지가 너무 적음 (최소 3장 이상 권장)
- 겹치는 영역이 없음
- 텍스처가 없는 이미지 (단색)

**해결**:
- 이미지 수 증가 (10장 이상 권장)
- 겹치는 영역 확보 (50-70% overlap)
- Feature-rich 영역 촬영

### "No valid poses found"

**원인**:
- 이미지 순서가 랜덤
- 회전이 너무 큼

**해결**:
- Sequential matching 사용
- 품질을 'high'나 'extreme'으로 설정
- 이미지를 시간 순서로 정렬

### GPU 메모리 부족

```yaml
# configs/default.yaml
sfm:
  use_gpu: false  # CPU 사용
  quality: medium # 품질 낮춤
```

### 느린 실행

```yaml
sfm:
  quality: low     # 또는 medium
  dense: false     # Dense 비활성화
```

## 고급 사용법

### 1. 특정 이미지만 처리

```python
from colmap_sfm import COLMAPRunner

runner = COLMAPRunner()

# SFM 실행
sparse_dir = runner.run_sfm_pipeline(
    image_dir='data/rgb',
    output_dir='data/sfm'
)

# 특정 이미지만 poses.json으로 변환
image_names = ['000001.png', '000003.png', '000005.png']
poses = runner.convert_to_poses_json(
    sparse_model_dir=sparse_dir,
    output_json='data/sfm/poses_subset.json',
    image_names=image_names
)
```

### 2. 기존 COLMAP 모델 변환

이미 COLMAP 결과가 있는 경우:

```python
from colmap_sfm import COLMAPRunner

runner = COLMAPRunner()

poses = runner.convert_to_poses_json(
    sparse_model_dir='path/to/colmap/sparse/0',
    output_json='poses.json'
)
```

### 3. Dense Point Cloud 생성

```python
from colmap_sfm import COLMAPRunner

runner = COLMAPRunner()

# Sparse SFM
sparse_dir = runner.run_sfm_pipeline(...)

# Dense reconstruction
dense_ply = runner.extract_dense_point_cloud(
    sparse_model_dir=sparse_dir,
    image_dir='data/rgb',
    output_dir='data/sfm',
    max_image_size=2000
)

print(f"Dense point cloud: {dense_ply}")
```

## 대안: 수동 포즈 입력

COLMAP을 사용하지 않고 수동으로 포즈를 제공할 수도 있습니다:

1. 다른 SFM 도구 사용 (OpenMVG, VisualSFM 등)
2. 결과를 `poses.json` 형식으로 변환
3. `data/sfm/poses.json`에 저장
4. `python -m src.pipeline align` 부터 실행

## 참고 자료

- **COLMAP 공식 문서**: https://colmap.github.io/
- **COLMAP GitHub**: https://github.com/colmap/colmap
- **튜토리얼**: https://colmap.github.io/tutorial.html
- **FAQ**: https://colmap.github.io/faq.html

## 요약

| 단계 | 명령어 |
|-----|--------|
| 1. COLMAP 설치 | `sudo apt-get install colmap` |
| 2. RGB 이미지 준비 | `data/rgb/*.png` |
| 3. SFM 실행 | `python -m src.pipeline sfm` |
| 4. 전체 파이프라인 | `python -m src.pipeline full` |

**자동화 완료!** 이제 RGB 이미지만 있으면 나머지는 자동으로 처리됩니다.
