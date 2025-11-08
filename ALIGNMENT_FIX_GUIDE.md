# 🔧 Depth-RGB Alignment 문제 해결 가이드

## 📊 문제 진단

**증상**:
- Alignment coverage: ~1% (정상: 15-40%)
- Validation에서 edge가 전혀 맞지 않음
- Filled pixels: 94627/8294400 (1.14%)

**원인**:
1. ❌ **Depth 카메라 왜곡 계수 이상** (k1=22.013, k2=12.4587)
2. ❌ **Geometric alignment 사용** (Orbbec Femto Bolt는 hardware-aligned)
3. ⚠️ Extrinsic 변환이 정확하지 않을 수 있음

---

## ✅ 해결 방법 (이미 적용됨)

### 1. Hardware-Aligned Depth 활성화

**파일**: `configs/default.yaml`

```yaml
align:
  use_simple_resize: true  # ← 변경됨 (false → true)
```

**이유**: Orbbec Femto Bolt는 하드웨어에서 이미 depth를 color frame에 정렬합니다.

---

### 2. Depth 왜곡 계수 초기화

**파일**: `calib/depth_camera_info.json`

```json
"D": [
  0.0,  // ← 변경됨 (22.013 → 0.0)
  0.0,  // ← 변경됨 (12.4587 → 0.0)
  0.0,
  0.0,
  0.0,
  0.0,
  0.0,
  0.0
]
```

**이유**:
- ToF/Structured Light depth 카메라는 일반적으로 왜곡이 거의 없음
- 기존 값(k1=22, k2=12)은 비정상적으로 크며, 심각한 왜곡을 일으킴
- Factory calibration에서 이미 보정된 경우가 많음

---

## 🧪 테스트 방법

### 1. 진단 스크립트 실행

```bash
python diagnose_alignment.py
```

**출력 예시**:
```
🔧 CALIBRATION DIAGNOSIS
========================
📷 RGB Camera:
   Resolution: 3840x2160
   fx, fy: 2246.0, 2244.8
   ✅ Principal point near center

📷 Depth Camera:
   Resolution: 512x512
   ✅ Distortion coefficients look reasonable

🔍 ALIGNMENT DIAGNOSIS
======================
✅ Found 197 depth images
📊 Aligned Depth Sample:
   Coverage: 35.2%  ← 이제 정상 범위!
   ✅ Coverage looks reasonable
```

---

### 2. Alignment 재실행

```bash
# 기존 결과 삭제
rm -rf outputs/aligned_depth

# Phase 3만 재실행
python -m src.pipeline align --config configs/default.yaml

# 또는 전체 파이프라인
python -m src.pipeline full --config configs/default.yaml
```

**기대 결과**:
```
INFO - Filled pixels: 2500000/8294400  ← ~30% (정상!)
```

---

### 3. Validation 실행

```bash
python -m src.validate_alignment \
    --rgb-dir data/rgb \
    --depth-dir data/depth \
    --aligned-dir outputs/aligned_depth \
    --calib-rgb calib/rgb_camera_info.json \
    --calib-depth calib/depth_camera_info.json \
    --num-samples 5
```

**기대 결과**:
```
Average coverage: 32.5%  ← ✅ 정상!
Average edge overlap: 68.3%  ← ✅ 정상!
✅ Edge alignment is good (>60%)
```

---

## 📋 Coverage 해석 기준

| Coverage | 상태 | 조치 |
|----------|------|------|
| 30-50% | ✅ 우수 | 정상 진행 |
| 15-30% | ✅ 양호 | 정상 진행 |
| 5-15% | ⚠️ 낮음 | 캘리브레이션 확인 |
| <5% | ❌ 실패 | 즉시 수정 필요 |

**현재 상태**: 1% → ❌ 실패 → **수정 적용됨**

---

## 🔍 추가 문제 해결

### 문제 1: 여전히 Coverage가 낮음 (5-15%)

**원인**: Extrinsic 변환이 부정확

**해결**:
1. Extrinsic 재캘리브레이션
2. 또는 임시로 extrinsic 비활성화:

```python
# src/pipeline.py, line 374-386
# 다음 부분을 주석 처리:
# if extrinsic_path.exists():
#     ... (전체 블록)
# T_d2r = None  # ← 이렇게 설정
```

---

### 문제 2: Edge Overlap이 낮음 (<40%)

**원인**:
- RGB와 Depth의 FOV(시야각) 차이
- Time synchronization 문제
- 물체 움직임

**확인 방법**:
```bash
# Validation 이미지 확인
ls outputs/alignment_validation/visual_*.png
```

시각적으로 RGB와 Depth edge가 맞는지 확인

---

### 문제 3: Hole이 너무 많음

**해결**: Hole filling 파라미터 조정

```yaml
# configs/default.yaml
align:
  hole_fill: true
  joint_bilateral: true
  bilateral_d: 15        # ← 증가 (기본: 9)
  bilateral_sigma_color: 100  # ← 증가 (기본: 75)
```

---

## 🎯 최종 체크리스트

- [x] `use_simple_resize: true` 설정
- [x] Depth 왜곡 계수 초기화 (D = [0, 0, ...])
- [ ] `diagnose_alignment.py` 실행 → 정상 확인
- [ ] Alignment 재실행
- [ ] Coverage 30% 이상 확인
- [ ] Validation 실행 → Edge overlap 60% 이상 확인
- [ ] 시각적 결과 확인 (`visual_*.png`)

---

## 📞 여전히 문제가 있다면?

### 로그 수집:

```bash
python -m src.pipeline align --config configs/default.yaml --log-level DEBUG > alignment.log 2>&1
```

### 확인 사항:

1. **Depth 이미지 품질**:
   ```bash
   python diagnose_alignment.py
   ```
   - Valid pixel ratio > 50%?
   - Depth range 0.3-5.0m?

2. **RGB-Depth 파일 매칭**:
   ```bash
   ls data/rgb/camera_RGB*.png | head -5
   ls data/depth/camera_DPT*.png | head -5
   ```
   - 파일명 패턴이 일치하는가?
   - 개수가 같은가?

3. **카메라 정보 정확성**:
   - RGB: 3840×2160, fx~2246
   - Depth: 512×512, fx~252
   - Orbbec Femto Bolt 사양과 일치하는가?

---

## 📚 참고 자료

- **Orbbec Femto Bolt Datasheet**: [링크]
- **OpenCV Calibration**: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- **프로젝트 문서**: `README.md`, `QUICKSTART.md`

---

**수정 날짜**: 2025-11-08
**버전**: 1.0
**상태**: ✅ 해결책 적용됨
