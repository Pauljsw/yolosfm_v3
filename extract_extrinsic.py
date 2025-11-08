#!/usr/bin/env python3
"""
Extract Extrinsic Transformation from Orbbec Camera
Extracts the transformation from Depth to Color frame
"""

import sys
import json
import numpy as np

try:
    import pyorbbecsdk as ob
    print("✅ PyOrbbecSDK imported successfully")
except ImportError:
    print("❌ PyOrbbecSDK not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyorbbecsdk"])
    import pyorbbecsdk as ob
    print("✅ PyOrbbecSDK installed and imported")


def extract_calibration():
    """Extract camera calibration including extrinsic transformation"""

    print("\n" + "="*80)
    print("Orbbec Camera Extrinsic Extraction")
    print("="*80)

    # Create pipeline
    pipeline = ob.Pipeline()

    try:
        # Get device
        device = pipeline.get_device()
        print(f"\n📷 Device: {device.get_device_info().get_name()}")

        # Get camera parameters
        config = ob.Config()

        # Enable color stream
        color_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        if color_profiles:
            # Find 3840x2160 profile
            color_profile = None
            for i in range(color_profiles.get_count()):
                profile = color_profiles.get_profile(i)
                video_profile = profile.as_video_stream_profile()
                if video_profile.get_width() == 3840 and video_profile.get_height() == 2160:
                    color_profile = profile
                    break

            if not color_profile:
                color_profile = color_profiles.get_profile(0)

            config.enable_stream(color_profile)
            print(f"✅ Color: {color_profile.as_video_stream_profile().get_width()}x{color_profile.as_video_stream_profile().get_height()}")

        # Enable depth stream
        depth_profiles = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        if depth_profiles:
            depth_profile = depth_profiles.get_profile(0)
            config.enable_stream(depth_profile)
            print(f"✅ Depth: {depth_profile.as_video_stream_profile().get_width()}x{depth_profile.as_video_stream_profile().get_height()}")

        # Start pipeline
        pipeline.start(config)
        print("\n⏳ Pipeline started, waiting for frames...")

        # Wait for frames to stabilize
        for _ in range(10):
            pipeline.wait_for_frames(1000)

        # Get camera parameters
        camera_params = pipeline.get_camera_param()

        print("\n" + "="*80)
        print("RGB Camera Intrinsics")
        print("="*80)

        rgb_intrinsics = camera_params.rgb_intrinsic
        print(f"Width: {rgb_intrinsics.width}")
        print(f"Height: {rgb_intrinsics.height}")
        print(f"fx: {rgb_intrinsics.fx}")
        print(f"fy: {rgb_intrinsics.fy}")
        print(f"cx: {rgb_intrinsics.cx}")
        print(f"cy: {rgb_intrinsics.cy}")
        print(f"Distortion model: {rgb_intrinsics.distortion_model}")
        print(f"Distortion coeffs: {rgb_intrinsics.distortion}")

        print("\n" + "="*80)
        print("Depth Camera Intrinsics")
        print("="*80)

        depth_intrinsics = camera_params.depth_intrinsic
        print(f"Width: {depth_intrinsics.width}")
        print(f"Height: {depth_intrinsics.height}")
        print(f"fx: {depth_intrinsics.fx}")
        print(f"fy: {depth_intrinsics.fy}")
        print(f"cx: {depth_intrinsics.cx}")
        print(f"cy: {depth_intrinsics.cy}")
        print(f"Distortion model: {depth_intrinsics.distortion_model}")
        print(f"Distortion coeffs: {depth_intrinsics.distortion}")

        print("\n" + "="*80)
        print("⭐ Extrinsic Transformation (Depth → Color)")
        print("="*80)

        extrinsic = camera_params.transform

        # Extract rotation (3x3)
        rotation = np.array([
            [extrinsic.rot[0], extrinsic.rot[1], extrinsic.rot[2]],
            [extrinsic.rot[3], extrinsic.rot[4], extrinsic.rot[5]],
            [extrinsic.rot[6], extrinsic.rot[7], extrinsic.rot[8]]
        ])

        # Extract translation (3x1, in meters)
        translation = np.array([
            extrinsic.trans[0] / 1000.0,  # Convert mm to meters
            extrinsic.trans[1] / 1000.0,
            extrinsic.trans[2] / 1000.0
        ])

        print("Rotation matrix (R):")
        print(rotation)
        print(f"\nTranslation vector (t) [meters]:")
        print(translation)
        print(f"\nTranslation [mm]: [{extrinsic.trans[0]:.2f}, {extrinsic.trans[1]:.2f}, {extrinsic.trans[2]:.2f}]")

        # Save to JSON
        output = {
            "description": "Extrinsic transformation from Depth to Color frame",
            "camera": device.get_device_info().get_name(),
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "t_mm": [extrinsic.trans[0], extrinsic.trans[1], extrinsic.trans[2]]
        }

        output_path = "calib/extrinsic_depth_to_color.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✅ Saved extrinsic to: {output_path}")

        # Also update RGB camera_info with correct values
        rgb_calib = {
            "width": rgb_intrinsics.width,
            "height": rgb_intrinsics.height,
            "distortion_model": "rational_polynomial",
            "D": list(rgb_intrinsics.distortion),
            "K": [
                [rgb_intrinsics.fx, 0.0, rgb_intrinsics.cx],
                [0.0, rgb_intrinsics.fy, rgb_intrinsics.cy],
                [0.0, 0.0, 1.0]
            ]
        }

        rgb_path = "calib/rgb_camera_info_sdk.json"
        with open(rgb_path, 'w') as f:
            json.dump(rgb_calib, f, indent=2)
        print(f"✅ Saved RGB calibration to: {rgb_path}")

        # Update depth camera_info
        depth_calib = {
            "width": depth_intrinsics.width,
            "height": depth_intrinsics.height,
            "distortion_model": "rational_polynomial",
            "D": list(depth_intrinsics.distortion),
            "K": [
                [depth_intrinsics.fx, 0.0, depth_intrinsics.cx],
                [0.0, depth_intrinsics.fy, depth_intrinsics.cy],
                [0.0, 0.0, 1.0]
            ]
        }

        depth_path = "calib/depth_camera_info_sdk.json"
        with open(depth_path, 'w') as f:
            json.dump(depth_calib, f, indent=2)
        print(f"✅ Saved Depth calibration to: {depth_path}")

        print("\n" + "="*80)
        print("🎉 Extraction Complete!")
        print("="*80)
        print("\nNext steps:")
        print("1. Check calib/extrinsic_depth_to_color.json")
        print("2. Update code to use T_d2r parameter")
        print("3. Re-run Phase 3 alignment")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop pipeline
        pipeline.stop()
        print("\n✅ Pipeline stopped")


if __name__ == "__main__":
    extract_calibration()
