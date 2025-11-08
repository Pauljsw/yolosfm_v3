"""
Calibration I/O Module
Loads camera intrinsics and distortion parameters from JSON files.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class CameraCalibration:
    """Camera calibration data container"""
    
    def __init__(self, width: int, height: int, K: np.ndarray, D: np.ndarray, 
                 distortion_model: str = "rational_polynomial"):
        self.width = width
        self.height = height
        self.K = K  # 3x3 intrinsic matrix
        self.D = D  # distortion coefficients
        self.distortion_model = distortion_model
        
    def __repr__(self):
        return (f"CameraCalibration(size={self.width}x{self.height}, "
                f"model={self.distortion_model})")


def load_camera_info(json_path: str) -> CameraCalibration:
    """
    Load camera calibration from JSON file.
    
    Expected JSON format:
    {
        "width": 3840,
        "height": 2160,
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "D": [k1, k2, p1, p2, k3, k4, k5, k6] or [k1, k2, p1, p2, k3]
        "distortion_model": "rational_polynomial" or "radial_tangential"
    }
    
    Args:
        json_path: Path to camera info JSON file
        
    Returns:
        CameraCalibration object
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Camera info file not found: {json_path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    width = data['width']
    height = data['height']

    # Load K matrix - support multiple formats
    # Prefer K_matrix (nested array) but fall back to K (can be flat or nested)
    if 'K_matrix' in data:
        K = np.array(data['K_matrix'], dtype=np.float64)
    else:
        if 'K' not in data:
            raise ValueError("Neither 'K' nor 'K_matrix' found in calibration file")
        K = np.array(data['K'], dtype=np.float64)

    # Auto-reshape if flat array (ROS camera_info format)
    if K.ndim == 1 and K.shape[0] == 9:
        K = K.reshape(3, 3)

    if K.shape != (3, 3):
        raise ValueError(f"Invalid K matrix shape: {K.shape}, expected (3, 3)")

    D = np.array(data['D'], dtype=np.float64)
    distortion_model = data.get('distortion_model', 'rational_polynomial')
    
    return CameraCalibration(width, height, K, D, distortion_model)


def load_poses(json_path: str) -> Dict:
    """
    Load camera poses from SFM result.
    
    Expected JSON format:
    {
        "image_id": {
            "filename": "000123.png",
            "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
            "t": [tx, ty, tz],
            "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]  # optional, can use from calib
        },
        ...
    }
    
    Args:
        json_path: Path to poses JSON file
        
    Returns:
        Dictionary mapping image_id to pose data
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Poses file not found: {json_path}")
    
    with open(path, 'r') as f:
        poses = json.load(f)
    
    # Convert to numpy arrays
    for img_id, pose in poses.items():
        pose['R'] = np.array(pose['R'], dtype=np.float64)
        pose['t'] = np.array(pose['t'], dtype=np.float64).reshape(3, 1)
        if 'K' in pose:
            pose['K'] = np.array(pose['K'], dtype=np.float64)
    
    return poses


def load_transform(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load rigid transformation between depth and RGB cameras.
    
    Expected JSON format:
    {
        "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
        "t": [tx, ty, tz]
    }
    
    Args:
        json_path: Path to transform JSON file
        
    Returns:
        Tuple of (R, t) where R is 3x3 rotation and t is 3x1 translation
    """
    path = Path(json_path)
    if not path.exists():
        return np.eye(3), np.zeros((3, 1))  # Identity if not provided
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    R = np.array(data['R'], dtype=np.float64)
    t = np.array(data['t'], dtype=np.float64).reshape(3, 1)
    
    return R, t


def save_camera_info(calib: CameraCalibration, json_path: str):
    """
    Save camera calibration to JSON file.
    
    Args:
        calib: CameraCalibration object
        json_path: Path to output JSON file
    """
    data = {
        'width': calib.width,
        'height': calib.height,
        'K': calib.K.tolist(),
        'D': calib.D.tolist(),
        'distortion_model': calib.distortion_model
    }
    
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def create_sample_calibration_files(output_dir: str):
    """
    Create sample calibration files for testing.
    
    Args:
        output_dir: Directory to save sample files
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # RGB camera (3840x2160)
    rgb_calib = {
        'width': 3840,
        'height': 2160,
        'K': [
            [2800.0, 0.0, 1920.0],
            [0.0, 2800.0, 1080.0],
            [0.0, 0.0, 1.0]
        ],
        'D': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 8 coefficients for rational_polynomial
        'distortion_model': 'rational_polynomial'
    }
    
    # Depth camera (512x512)
    depth_calib = {
        'width': 512,
        'height': 512,
        'K': [
            [365.0, 0.0, 256.0],
            [0.0, 365.0, 256.0],
            [0.0, 0.0, 1.0]
        ],
        'D': [0.0, 0.0, 0.0, 0.0, 0.0],  # 5 coefficients for radial_tangential
        'distortion_model': 'radial_tangential'
    }
    
    with open(output_path / 'rgb_camera_info.json', 'w') as f:
        json.dump(rgb_calib, f, indent=2)
    
    with open(output_path / 'depth_camera_info.json', 'w') as f:
        json.dump(depth_calib, f, indent=2)
    
    print(f"Sample calibration files created in {output_dir}")


if __name__ == '__main__':
    # Test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'create-samples':
        create_sample_calibration_files('calib')
    else:
        print("Usage: python calib_io.py create-samples")
