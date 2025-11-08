#!/usr/bin/env python3
"""
Alignment Diagnostic Tool
Analyzes why depth-to-RGB alignment is failing.

Usage:
    python diagnose_alignment.py
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys

def load_calib(path):
    """Load camera calibration"""
    with open(path) as f:
        calib = json.load(f)
    return calib

def diagnose_depth_image(depth_path):
    """Analyze a single depth image"""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"❌ Failed to load: {depth_path}")
        return None

    depth = depth.astype(np.float32)

    print(f"\n📊 Depth Image: {Path(depth_path).name}")
    print(f"   Shape: {depth.shape}")
    print(f"   Dtype: {depth.dtype}")
    print(f"   Min/Max: {depth.min():.1f} / {depth.max():.1f}")

    valid_mask = depth > 0
    if valid_mask.sum() == 0:
        print(f"   ❌ NO VALID PIXELS!")
        return None

    valid_depths = depth[valid_mask]
    print(f"   Valid pixels: {valid_mask.sum()}/{depth.size} ({valid_mask.sum()/depth.size*100:.1f}%)")
    print(f"   Median: {np.median(valid_depths):.1f}")
    print(f"   Mean: {np.mean(valid_depths):.1f}")
    print(f"   Std: {np.std(valid_depths):.1f}")

    # Auto-detect unit
    if np.median(valid_depths) > 100:
        unit = "mm"
        depth_m = depth / 1000.0
    else:
        unit = "m"
        depth_m = depth

    print(f"   Auto-detected unit: {unit}")

    valid_depths_m = depth_m[depth_m > 0]
    print(f"   Depth range (m): {valid_depths_m.min():.3f} - {valid_depths_m.max():.3f}")

    # Check for abnormal values
    warnings = []
    if valid_depths_m.max() > 10.0:
        warnings.append("⚠️  Very large depth values (>10m)")
    if valid_depths_m.min() < 0.1:
        warnings.append("⚠️  Very small depth values (<0.1m)")
    if valid_mask.sum() < depth.size * 0.5:
        warnings.append("⚠️  Low valid pixel ratio (<50%)")

    if warnings:
        for w in warnings:
            print(f"   {w}")
    else:
        print(f"   ✅ Depth values look normal")

    return depth_m

def diagnose_calibration():
    """Analyze calibration files"""
    print("\n" + "="*80)
    print("🔧 CALIBRATION DIAGNOSIS")
    print("="*80)

    # RGB calibration
    rgb_calib = load_calib('calib/rgb_camera_info.json')
    print(f"\n📷 RGB Camera:")
    print(f"   Resolution: {rgb_calib['width']}x{rgb_calib['height']}")
    K = np.array(rgb_calib['K'])
    print(f"   fx, fy: {K[0,0]:.1f}, {K[1,1]:.1f}")
    print(f"   cx, cy: {K[0,2]:.1f}, {K[1,2]:.1f}")
    print(f"   Distortion model: {rgb_calib.get('distortion_model', 'unknown')}")
    D = rgb_calib['D']
    print(f"   Distortion (first 4): {D[:4]}")

    # Check RGB calibration
    cx_expected = rgb_calib['width'] / 2
    cy_expected = rgb_calib['height'] / 2
    if abs(K[0,2] - cx_expected) > 100 or abs(K[1,2] - cy_expected) > 100:
        print(f"   ⚠️  Principal point far from center: ({K[0,2]:.0f}, {K[1,2]:.0f}) vs expected ({cx_expected:.0f}, {cy_expected:.0f})")
    else:
        print(f"   ✅ Principal point near center")

    # Depth calibration
    depth_calib = load_calib('calib/depth_camera_info.json')
    print(f"\n📷 Depth Camera:")
    print(f"   Resolution: {depth_calib['width']}x{depth_calib['height']}")
    K_d = np.array(depth_calib['K'])
    print(f"   fx, fy: {K_d[0,0]:.1f}, {K_d[1,1]:.1f}")
    print(f"   cx, cy: {K_d[0,2]:.1f}, {K_d[1,2]:.1f}")
    D_d = depth_calib['D']
    print(f"   Distortion (first 4): {D_d[:4]}")

    # Check depth distortion
    if any(abs(d) > 1.0 for d in D_d[:4]):
        print(f"   ❌ ABNORMAL DISTORTION COEFFICIENTS!")
        print(f"      Depth cameras typically have near-zero distortion")
        print(f"      Consider setting D = [0, 0, 0, 0, 0, 0, 0, 0]")
    else:
        print(f"   ✅ Distortion coefficients look reasonable")

    # Extrinsic
    try:
        extrinsic = load_calib('calib/extrinsic_depth_to_color.json')
        print(f"\n🔗 Extrinsic (Depth → RGB):")
        R = np.array(extrinsic['R'])
        t = np.array(extrinsic['t'])
        print(f"   Rotation determinant: {np.linalg.det(R):.6f} (should be ≈1.0)")
        print(f"   Translation (m): {t}")
        print(f"   Baseline: {np.linalg.norm(t)*1000:.1f} mm")

        # Check if R is close to identity
        R_diff = R - np.eye(3)
        if np.max(np.abs(R_diff)) < 0.1:
            print(f"   ℹ️  Rotation is nearly identity (cameras roughly aligned)")
        else:
            print(f"   ℹ️  Rotation has {np.max(np.abs(R_diff)):.3f} max deviation from identity")

    except FileNotFoundError:
        print(f"\n⚠️  No extrinsic calibration found")

def diagnose_alignment():
    """Diagnose actual alignment process"""
    print("\n" + "="*80)
    print("🔍 ALIGNMENT DIAGNOSIS")
    print("="*80)

    # Check if depth files exist
    depth_dir = Path('data/depth')
    if not depth_dir.exists():
        print(f"❌ Depth directory not found: {depth_dir}")
        return

    depth_files = list(depth_dir.glob('*.png'))
    if not depth_files:
        print(f"❌ No depth files found in {depth_dir}")
        return

    print(f"\n✅ Found {len(depth_files)} depth images")

    # Analyze first 3 depth images
    print(f"\n📊 Analyzing first 3 depth images:")
    for i, depth_path in enumerate(depth_files[:3]):
        diagnose_depth_image(str(depth_path))

    # Check aligned outputs
    aligned_dir = Path('outputs/aligned_depth')
    if aligned_dir.exists():
        aligned_files = list(aligned_dir.glob('*.png'))
        print(f"\n✅ Found {len(aligned_files)} aligned depth images")

        if aligned_files:
            # Analyze first aligned depth
            aligned = cv2.imread(str(aligned_files[0]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            valid_ratio = np.sum(aligned > 0) / aligned.size
            print(f"\n📊 Aligned Depth Sample: {aligned_files[0].name}")
            print(f"   Shape: {aligned.shape}")
            print(f"   Coverage: {valid_ratio*100:.1f}%")

            if valid_ratio < 0.05:
                print(f"   ❌ VERY LOW COVERAGE (<5%)")
                print(f"   → Alignment is likely failing")
            elif valid_ratio < 0.15:
                print(f"   ⚠️  Low coverage (<15%)")
            else:
                print(f"   ✅ Coverage looks reasonable")
    else:
        print(f"\n⚠️  No aligned depth directory found")

def main():
    print("="*80)
    print("🔬 DEPTH-TO-RGB ALIGNMENT DIAGNOSTIC TOOL")
    print("="*80)

    # Check if we're in the right directory
    if not Path('calib').exists():
        print("❌ Not in project root (calib/ not found)")
        print("   Please run from yolosfm_v3/")
        sys.exit(1)

    diagnose_calibration()
    diagnose_alignment()

    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    # Load current config
    import yaml
    with open('configs/default.yaml') as f:
        config = yaml.safe_load(f)

    use_simple_resize = config['align'].get('use_simple_resize', False)

    if use_simple_resize:
        print("✅ use_simple_resize is TRUE (good for Orbbec Femto Bolt)")
    else:
        print("❌ use_simple_resize is FALSE")
        print("   For Orbbec Femto Bolt, set to TRUE in configs/default.yaml")

    # Check depth distortion
    depth_calib = load_calib('calib/depth_camera_info.json')
    if any(abs(d) > 1.0 for d in depth_calib['D'][:4]):
        print("❌ Depth camera has abnormal distortion coefficients")
        print("   Set D = [0, 0, 0, 0, 0, 0, 0, 0] in calib/depth_camera_info.json")
    else:
        print("✅ Depth distortion looks OK")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
