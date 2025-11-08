#!/usr/bin/env python3
"""
Compare old and new calibration parameters.

Shows the difference between manually set and factory-extracted calibrations.
"""

import json
import sys
from pathlib import Path

def load_json(path):
    """Load JSON file"""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def compare_intrinsics(old, new, camera_name):
    """Compare intrinsic parameters"""
    print(f"\n{'='*60}")
    print(f"{camera_name} Camera Intrinsics")
    print(f"{'='*60}")

    if old is None:
        print("⚠️  No old calibration found")
        return

    if new is None:
        print("❌ No new calibration found - run extraction first!")
        return

    # Resolution
    print(f"\nResolution:")
    old_res = f"{old['width']}x{old['height']}"
    new_res = f"{new['width']}x{new['height']}"
    print(f"  Old: {old_res}")
    print(f"  New: {new_res}")
    if old_res == new_res:
        print(f"  ✅ MATCH")
    else:
        print(f"  ❌ MISMATCH")

    # Intrinsic matrix K
    print(f"\nIntrinsic Matrix K:")
    old_K = old['K']
    new_K = new['K']

    params = ['fx', 'fy', 'cx', 'cy']
    indices = [(0, 0), (1, 1), (0, 2), (1, 2)]

    for param, (i, j) in zip(params, indices):
        old_val = old_K[i][j]
        new_val = new_K[i][j]
        diff = abs(old_val - new_val)
        rel_diff = diff / new_val * 100 if new_val != 0 else 0

        print(f"  {param}:")
        print(f"    Old: {old_val:.4f}")
        print(f"    New: {new_val:.4f}")
        print(f"    Diff: {diff:.4f} ({rel_diff:.2f}%)")

        if rel_diff < 1:
            print(f"    ✅ CLOSE (<1% difference)")
        elif rel_diff < 5:
            print(f"    ⚠️  MODERATE (1-5% difference)")
        else:
            print(f"    ❌ LARGE (>5% difference)")

    # Distortion coefficients
    print(f"\nDistortion Coefficients:")
    old_D = old['D']
    new_D = new['D']

    coef_names = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']

    print(f"  {'Coef':<6} {'Old':>12} {'New':>12} {'Diff':>12} {'Status'}")
    print(f"  {'-'*60}")

    for i, name in enumerate(coef_names):
        if i < len(old_D) and i < len(new_D):
            old_val = old_D[i]
            new_val = new_D[i]
            diff = abs(old_val - new_val)

            # Status
            if abs(new_val) < 0.001:
                if abs(old_val) < 0.001:
                    status = "✅ Both ~0"
                else:
                    status = "⚠️  Old≠0, New~0"
            elif abs(old_val) < 0.001:
                status = "⚠️  Old~0, New≠0"
            elif diff / abs(new_val) < 0.1:
                status = "✅ Close"
            else:
                status = "❌ Different"

            print(f"  {name:<6} {old_val:>12.6f} {new_val:>12.6f} {diff:>12.6f} {status}")
        elif i < len(new_D):
            print(f"  {name:<6} {'N/A':>12} {new_D[i]:>12.6f} {'N/A':>12} ℹ️  New only")

    # Interpretation
    print(f"\n💡 Interpretation:")

    # Check if distortion is minimal
    max_abs_new = max(abs(d) for d in new_D)
    max_abs_old = max(abs(d) for d in old_D)

    print(f"  Max |D| (old): {max_abs_old:.6f}")
    print(f"  Max |D| (new): {max_abs_new:.6f}")

    if max_abs_new < 0.1:
        print(f"\n  ✅ New distortion is minimal (< 0.1)")
        print(f"     → use_simple_resize: true is appropriate")
    elif max_abs_new < 1.0:
        print(f"\n  ⚠️  New distortion is moderate (0.1-1.0)")
        print(f"     → Consider use_simple_resize: false for better accuracy")
    else:
        print(f"\n  ❌ New distortion is significant (> 1.0)")
        print(f"     → MUST use use_simple_resize: false")
        print(f"     → Enable distortion correction")

def compare_extrinsic(old, new):
    """Compare extrinsic parameters"""
    print(f"\n{'='*60}")
    print(f"Extrinsic Transformation (Depth → Color)")
    print(f"{'='*60}")

    if old is None:
        print("⚠️  No old extrinsic found")
        return

    if new is None:
        print("❌ No new extrinsic found - run extraction first!")
        return

    # Translation
    print(f"\nTranslation (meters):")
    old_t = old['t']
    new_t = new['t']

    for i, axis in enumerate(['x', 'y', 'z']):
        old_val = old_t[i]
        new_val = new_t[i]
        diff = abs(old_val - new_val)

        print(f"  {axis}: Old={old_val:.6f}, New={new_val:.6f}, Diff={diff:.6f}")

        if diff < 0.001:  # 1mm
            print(f"     ✅ MATCH (<1mm)")
        elif diff < 0.005:  # 5mm
            print(f"     ⚠️  SMALL DIFFERENCE (1-5mm)")
        else:
            print(f"     ❌ LARGE DIFFERENCE (>5mm)")

    # Rotation matrix
    print(f"\nRotation Matrix:")
    old_R = old['R']
    new_R = new['R']

    max_diff = 0
    for i in range(3):
        for j in range(3):
            diff = abs(old_R[i][j] - new_R[i][j])
            max_diff = max(max_diff, diff)

    print(f"  Max element difference: {max_diff:.6f}")

    if max_diff < 0.01:
        print(f"  ✅ Rotation matrices are very similar")
    elif max_diff < 0.05:
        print(f"  ⚠️  Moderate difference in rotation")
    else:
        print(f"  ❌ Large difference in rotation")

def main():
    print("="*60)
    print("Calibration Parameter Comparison")
    print("="*60)

    calib_dir = Path(__file__).parent.parent / 'calib'

    # Define old and new paths
    old_rgb_path = calib_dir / 'rgb_camera_info.json.old'
    new_rgb_path = calib_dir / 'rgb_camera_info.json'

    old_depth_path = calib_dir / 'depth_camera_info.json.old'
    new_depth_path = calib_dir / 'depth_camera_info.json'

    old_ext_path = calib_dir / 'extrinsic_depth_to_color.json.old'
    new_ext_path = calib_dir / 'extrinsic_depth_to_color.json'

    # Backup current files if .old doesn't exist
    if not old_depth_path.exists() and new_depth_path.exists():
        print(f"\nℹ️  Creating backup of current calibration as .old files...")
        import shutil
        if new_rgb_path.exists():
            shutil.copy(new_rgb_path, old_rgb_path)
        if new_depth_path.exists():
            shutil.copy(new_depth_path, old_depth_path)
        if new_ext_path.exists():
            shutil.copy(new_ext_path, old_ext_path)
        print(f"   Please run extraction first, then run this script again.")
        return

    # Load calibrations
    old_rgb = load_json(old_rgb_path)
    new_rgb = load_json(new_rgb_path)

    old_depth = load_json(old_depth_path)
    new_depth = load_json(new_depth_path)

    old_ext = load_json(old_ext_path)
    new_ext = load_json(new_ext_path)

    # Compare
    compare_intrinsics(old_rgb, new_rgb, "RGB")
    compare_intrinsics(old_depth, new_depth, "Depth")
    compare_extrinsic(old_ext, new_ext)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")

    if new_depth:
        max_D = max(abs(d) for d in new_depth['D'])
        print(f"\n📊 Depth Distortion Analysis:")
        print(f"   Max |D|: {max_D:.6f}")

        if max_D < 0.1:
            print(f"\n   ✅ RECOMMENDATION:")
            print(f"      use_simple_resize: true")
            print(f"      (Distortion is negligible)")
        else:
            print(f"\n   ⚠️  RECOMMENDATION:")
            print(f"      use_simple_resize: false")
            print(f"      (Distortion correction needed)")

    print(f"\n{'='*60}")

if __name__ == '__main__':
    main()
