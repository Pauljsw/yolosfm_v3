# 📐 Orbbec Femto Bolt Calibration Extractor

Factory calibration 파라미터를 Orbbec Femto Bolt 카메라에서 직접 추출하는 도구입니다.

## 🎯 목적

- RGB/Depth 카메라의 **정확한 intrinsic 파라미터** 추출
- **Extrinsic transformation** (Depth → Color) 추출
- **왜곡 계수 (distortion coefficients)** 확인
- Pipeline에서 사용 가능한 JSON 형식으로 자동 저장

---

## 📋 요구사항

### 1. OrbbecSDK 설치

```bash
# 다운로드
wget https://github.com/orbbec/OrbbecSDK/releases/download/v1.9.6/OrbbecSDK_v1.9.6_linux_x64.tar.gz

# 압축 해제
tar -xzf OrbbecSDK_v1.9.6_linux_x64.tar.gz
cd OrbbecSDK_v1.9.6_linux_x64

# 설치
sudo ./install.sh
```

### 2. 카메라 연결

- Orbbec Femto Bolt를 USB 3.0 포트에 연결
- 전원 공급 확인

### 3. 빌드 도구

```bash
sudo apt install cmake build-essential
```

---

## 🚀 사용 방법

### 자동 빌드 및 실행 (권장)

```bash
cd calib_extraction
./build_and_run.sh
```

### 수동 빌드

```bash
cd calib_extraction
mkdir build
cd build
cmake ..
make
./extract_calib
```

---

## 📂 출력 파일

모든 파일은 `../calib/` 디렉토리에 저장됩니다:

### 1. **rgb_camera_info.json** ✅
```json
{
  "width": 3840,
  "height": 2160,
  "distortion_model": "rational_polynomial",
  "D": [0.0779024, -0.106185, ...],  // 실제 왜곡 계수
  "K": [[2246.03, 0.0, 1903.48], ...]
}
```

### 2. **depth_camera_info.json** ✅
```json
{
  "width": 512,
  "height": 512,
  "distortion_model": "rational_polynomial",
  "D": [k1, k2, p1, p2, k3, k4, k5, k6],  // 🔥 중요: 실제 값
  "K": [[252.003, 0.0, 263.51], ...]
}
```

### 3. **extrinsic_depth_to_color.json** ✅
```json
{
  "R": [[0.994191, 0.006611, 0.007913], ...],
  "t": [-0.032196, -0.000814, 0.002416],  // meters
  "baseline_mm": 32.297165
}
```

### 4. **femto_bolt_calibration.txt** ℹ️
사람이 읽기 쉬운 텍스트 형식

### 5. **femto_bolt_calibration_numpy.txt** 🐍
Python/NumPy 코드 형식

---

## 🔍 결과 해석

### Depth 왜곡 계수 확인

실행 후 `calib/depth_camera_info.json`을 열어 `D` 값을 확인:

```bash
cat ../calib/depth_camera_info.json | grep -A 10 '"D"'
```

**케이스 1**: 왜곡이 거의 없음
```json
"D": [0.0001, -0.0002, 0.00003, ...]  // 모두 0에 가까움
```
→ ✅ **use_simple_resize: true** (현재 설정 유지)

**케이스 2**: 왜곡이 큼
```json
"D": [22.013, 12.4587, ...]  // 값이 큼
```
→ ⚠️ **use_simple_resize: false** + 왜곡 보정 필요

---

## 🐛 문제 해결

### 1. "OrbbecSDK not found"

```bash
# SDK 설치 확인
ldconfig -p | grep OrbbecSDK

# 없으면 재설치
sudo ./install.sh  # OrbbecSDK 디렉토리에서
sudo ldconfig
```

### 2. "No Orbbec camera detected"

```bash
# USB 연결 확인
lsusb | grep 2bc5

# 권한 문제 해결
sudo chmod 666 /dev/bus/usb/*/*
```

### 3. "Pipeline failed to start"

- 카메라 재연결
- 다른 프로그램이 카메라 사용 중인지 확인
- 재부팅 시도

---

## 📊 기대 출력 예시

```
==========================================================
Orbbec Femto Bolt Calibration Extractor
==========================================================

✅ Camera found!
   Name: Femto Bolt
   Serial: CL8855300BX
   Firmware: 1.0.9

Available Color Resolutions:
  - 3840x2160 @ 30fps
  - 1920x1080 @ 30fps
  ...
✅ Selected Color: 3840x2160

Available Depth Resolutions:
  - 512x512 @ 30fps
  - 640x576 @ 30fps
  ...
✅ Selected Depth: 512x512

==========================================================
📐 Extracting calibration...
==========================================================

📷 RGB Camera Intrinsic:
   Resolution: 3840x2160
   fx: 2246.03
   fy: 2244.84
   cx: 1903.48
   cy: 1091.56
   Distortion: [0.0779024, -0.106185, ...]

🎯 Depth Camera Intrinsic:
   Resolution: 512x512
   fx: 252.003
   fy: 251.966
   cx: 263.51
   cy: 260.239
   Distortion: [?, ?, ...]  ← 중요!

⭐ Extrinsic Transformation (Depth → Color):
   Rotation matrix (R):
     [0.994191, 0.006611, 0.007913]
     ...
   Translation (t) [meters]:
     [-0.032196, -0.000814, 0.002416]
   Baseline: 32.297 mm

==========================================================
💾 Saving calibration...
==========================================================
✅ Saved: ../calib/rgb_camera_info.json
✅ Saved: ../calib/depth_camera_info.json
✅ Saved: ../calib/extrinsic_depth_to_color.json
...

==========================================================
✨ SUCCESS! Calibration extraction complete!
==========================================================
```

---

## 🔄 다음 단계

### 1. Depth 왜곡 확인

```bash
python3 << 'EOF'
import json
with open('../calib/depth_camera_info.json') as f:
    calib = json.load(f)
D = calib['D']
print(f"Depth Distortion: {D}")
max_abs = max(abs(d) for d in D)
print(f"Max absolute value: {max_abs}")

if max_abs < 0.1:
    print("\n✅ Distortion is minimal (< 0.1)")
    print("   Keep: use_simple_resize = true")
else:
    print(f"\n⚠️  Distortion is significant (max={max_abs})")
    print("   Consider: use_simple_resize = false")
EOF
```

### 2. Alignment 재실행

```bash
cd ..
rm -rf outputs/aligned_depth
python -m src.pipeline align --config configs/default.yaml
```

### 3. Validation

```bash
python -m src.validate_alignment \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --aligned-dir outputs/aligned_depth \
    --calib-rgb calib/rgb_camera_info.json \
    --calib-depth calib/depth_camera_info.json \
    --num-samples 5
```

**기대 결과**: Coverage 30-40%, Edge overlap 60-80%

---

## 📝 주요 수정 사항

원본 코드 대비 수정된 부분:

### 수정 1: Depth Distortion 전체 사용
```cpp
// BEFORE (원본 - 잘못됨)
depth_json << "    " << depthDistortion.k3 << ",\n";
depth_json << "    0.0,\n";  // ← 강제로 0
depth_json << "    0.0,\n";

// AFTER (수정됨)
depth_json << "    " << depthDistortion.k3 << ",\n";
depth_json << "    " << depthDistortion.k4 << ",\n";  // ← 실제 값
depth_json << "    " << depthDistortion.k5 << ",\n";
depth_json << "    " << depthDistortion.k6 << "\n";
```

### 수정 2: NumPy 파일도 동일
8개 계수 모두 포함하도록 수정

---

## 🎓 참고 자료

- **OrbbecSDK**: https://github.com/orbbec/OrbbecSDK
- **OpenCV Distortion Models**: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- **Rational Polynomial Model**: k1-k6 (radial), p1-p2 (tangential)

---

**작성일**: 2025-11-08
**버전**: 1.0
**목적**: Factory calibration 정확한 추출
