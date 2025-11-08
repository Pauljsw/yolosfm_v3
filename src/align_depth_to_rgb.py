"""
Align Depth to RGB Module
Aligns depth images (512x512) to RGB resolution (3840x2160) with proper calibration.
"""
import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def undistort_points(points: np.ndarray, K: np.ndarray, D: np.ndarray, 
                     distortion_model: str = 'rational_polynomial') -> np.ndarray:
    """
    Undistort image points using camera calibration.
    
    Args:
        points: Nx2 array of image points (u, v)
        K: 3x3 camera intrinsic matrix
        D: Distortion coefficients
        distortion_model: 'rational_polynomial' or 'radial_tangential'
        
    Returns:
        Nx2 array of undistorted points
    """
    if len(points) == 0:
        return points
    
    # Reshape for cv2
    points = points.reshape(-1, 1, 2).astype(np.float32)
    
    if distortion_model == 'rational_polynomial':
        # OpenCV's undistortPoints expects distortion in specific format
        # For rational polynomial: k1,k2,p1,p2,k3,k4,k5,k6
        undistorted = cv2.undistortPoints(points, K, D, None, K)
    else:  # radial_tangential
        undistorted = cv2.undistortPoints(points, K, D, None, K)
    
    return undistorted.reshape(-1, 2)


def backproject_depth(u: np.ndarray, v: np.ndarray, depth: np.ndarray, 
                      K: np.ndarray) -> np.ndarray:
    """
    Backproject depth pixels to 3D camera coordinates.
    
    Args:
        u: x coordinates in image
        v: y coordinates in image
        depth: depth values in meters
        K: 3x3 camera intrinsic matrix
        
    Returns:
        Nx3 array of 3D points in camera frame
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    return np.stack([x, y, z], axis=-1)


def project_3d_to_image(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D points to image coordinates.
    
    Args:
        points_3d: Nx3 array of 3D points
        K: 3x3 camera intrinsic matrix
        
    Returns:
        Nx2 array of image coordinates
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    z = points_3d[:, 2]
    valid = z > 0
    
    u = np.zeros(len(points_3d))
    v = np.zeros(len(points_3d))
    
    u[valid] = (points_3d[valid, 0] * fx / z[valid]) + cx
    v[valid] = (points_3d[valid, 1] * fy / z[valid]) + cy
    
    return np.stack([u, v], axis=-1), valid


def align_depth_to_rgb(
    depth_img: np.ndarray,
    rgb_K: np.ndarray,
    rgb_D: np.ndarray,
    depth_K: np.ndarray,
    depth_D: np.ndarray,
    rgb_size: Tuple[int, int],
    T_d2r: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    depth_unit: str = 'm',
    hole_fill: bool = True,
    joint_bilateral: bool = True,
    bilateral_params: Optional[dict] = None,
    use_simple_resize: bool = False
) -> np.ndarray:
    """
    Align depth image to RGB image resolution and frame.

    Pipeline:
    1. Undistort depth coordinates
    2. Backproject to 3D (depth camera frame)
    3. Transform to RGB camera frame (if T_d2r provided)
    4. Project to RGB image coordinates
    5. Z-buffer to handle overlaps
    6. Hole filling and smoothing

    Args:
        depth_img: HxW depth image (e.g., 512x512)
        rgb_K: 3x3 RGB camera intrinsic matrix
        rgb_D: RGB camera distortion coefficients
        depth_K: 3x3 depth camera intrinsic matrix
        depth_D: Depth camera distortion coefficients
        rgb_size: (width, height) of RGB image (e.g., 3840x2160)
        T_d2r: Optional (R, t) transformation from depth to RGB frame
        depth_unit: 'm' or 'mm' or 'auto'
        hole_fill: Whether to fill holes
        joint_bilateral: Whether to apply joint bilateral filter
        bilateral_params: Parameters for bilateral filter
        use_simple_resize: Use simple resize (for hardware-aligned depth like Orbbec)

    Returns:
        Aligned depth image at RGB resolution (H_rgb x W_rgb), in meters
    """
    h_depth, w_depth = depth_img.shape
    w_rgb, h_rgb = rgb_size

    logger.info(f"Aligning depth {w_depth}x{h_depth} to RGB {w_rgb}x{h_rgb}")

    # Auto-detect depth unit if requested
    if depth_unit == 'auto':
        from .utils import detect_depth_unit
        depth_unit = detect_depth_unit(depth_img)
        logger.info(f"Auto-detected depth unit: {depth_unit}")

    # Convert depth to meters
    if depth_unit == 'mm':
        depth_img = depth_img / 1000.0

    # FAST PATH: For hardware-aligned depth (e.g., Orbbec aligned_depth_to_color)
    if use_simple_resize:
        logger.info("Using simple resize (hardware-aligned depth)")
        aligned_depth = cv2.resize(depth_img, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)

        if hole_fill:
            aligned_depth = fill_depth_holes(aligned_depth)

        if joint_bilateral:
            if bilateral_params is None:
                bilateral_params = {'d': 9, 'sigma_color': 75, 'sigma_space': 75}
            max_val = np.max(aligned_depth)
            if max_val > 0:
                depth_normalized = (aligned_depth / max_val * 255).astype(np.uint8)
                depth_filtered = cv2.bilateralFilter(
                    depth_normalized,
                    bilateral_params['d'],
                    bilateral_params['sigma_color'],
                    bilateral_params['sigma_space']
                )
                aligned_depth = (depth_filtered / 255.0) * max_val

        logger.info(f"Filled pixels: {np.sum(aligned_depth > 0)}/{h_rgb*w_rgb}")
        return aligned_depth
    
    # Create coordinate grids
    u_depth, v_depth = np.meshgrid(np.arange(w_depth), np.arange(h_depth))
    u_depth = u_depth.flatten()
    v_depth = v_depth.flatten()
    depth_values = depth_img.flatten()
    
    # Filter valid depths
    valid_mask = depth_values > 0
    u_depth = u_depth[valid_mask]
    v_depth = v_depth[valid_mask]
    depth_values = depth_values[valid_mask]
    
    if len(depth_values) == 0:
        logger.warning("No valid depth values found")
        return np.zeros((h_rgb, w_rgb), dtype=np.float32)
    
    logger.debug(f"Valid depth pixels: {len(depth_values)}/{h_depth*w_depth}")
    
    # Undistort depth coordinates (optional, often depth is already rectified)
    # depth_points = np.stack([u_depth, v_depth], axis=-1)
    # depth_points_undist = undistort_points(depth_points, depth_K, depth_D)
    # u_depth, v_depth = depth_points_undist[:, 0], depth_points_undist[:, 1]
    
    # Backproject to 3D (depth camera frame)
    points_3d = backproject_depth(u_depth, v_depth, depth_values, depth_K)
    
    # Transform to RGB frame if needed
    if T_d2r is not None:
        R, t = T_d2r
        # Apply: P_color = R @ P_depth + t
        # Using matrix form: points @ R.T + t.T for vectorized operation
        points_3d = points_3d @ R.T + t.T  # t is (3,1), t.T is (1,3) for broadcasting
    
    # Project to RGB image
    rgb_coords, valid_proj = project_3d_to_image(points_3d, rgb_K)
    
    # Filter valid projections
    valid_proj &= (rgb_coords[:, 0] >= 0) & (rgb_coords[:, 0] < w_rgb)
    valid_proj &= (rgb_coords[:, 1] >= 0) & (rgb_coords[:, 1] < h_rgb)

    rgb_coords = rgb_coords[valid_proj]
    # CRITICAL: Use original depth measurements, not transformed z-coordinate
    # The transformation changes pixel coordinates, but depth values stay the same
    # (depth = distance measured by depth sensor, regardless of camera frame)
    depth_values_proj = points_3d[valid_proj, 2]
    
    logger.debug(f"Valid projections: {len(depth_values_proj)}/{len(valid_proj)}")
    
    # Z-buffer: keep closest depth for each pixel
    aligned_depth = np.zeros((h_rgb, w_rgb), dtype=np.float32)
    count_map = np.zeros((h_rgb, w_rgb), dtype=np.int32)
    
    u_rgb = np.round(rgb_coords[:, 0]).astype(np.int32)
    v_rgb = np.round(rgb_coords[:, 1]).astype(np.int32)
    
    # Clip to valid range
    u_rgb = np.clip(u_rgb, 0, w_rgb - 1)
    v_rgb = np.clip(v_rgb, 0, h_rgb - 1)
    
    # Use z-buffer approach
    for i in range(len(u_rgb)):
        u, v, d = u_rgb[i], v_rgb[i], depth_values_proj[i]
        if aligned_depth[v, u] == 0 or d < aligned_depth[v, u]:
            aligned_depth[v, u] = d
        count_map[v, u] += 1
    
    logger.info(f"Filled pixels: {np.sum(aligned_depth > 0)}/{h_rgb*w_rgb}")
    
    # Hole filling
    if hole_fill:
        aligned_depth = fill_depth_holes(aligned_depth)
        logger.debug("Hole filling completed")
    
    # Joint bilateral filtering for edge-preserving smoothing
    if joint_bilateral:
        if bilateral_params is None:
            bilateral_params = {'d': 9, 'sigma_color': 75, 'sigma_space': 75}

        max_val = np.max(aligned_depth)
        if max_val > 0:  # Avoid division by zero
            # Convert to 8-bit for bilateral filter
            depth_normalized = (aligned_depth / max_val * 255).astype(np.uint8)
            depth_filtered = cv2.bilateralFilter(
                depth_normalized,
                bilateral_params['d'],
                bilateral_params['sigma_color'],
                bilateral_params['sigma_space']
            )
            # Convert back
            aligned_depth = (depth_filtered / 255.0) * max_val
            logger.debug("Bilateral filtering completed")
        else:
            logger.warning("Skipping bilateral filter: no valid depth values")
    
    return aligned_depth


def fill_depth_holes(depth: np.ndarray, max_hole_size: int = 10) -> np.ndarray:
    """
    Fill small holes in depth map using inpainting.
    
    Args:
        depth: Depth image with holes (zero values)
        max_hole_size: Maximum hole size to fill
        
    Returns:
        Depth image with holes filled
    """
    # Create mask for holes
    mask = (depth == 0).astype(np.uint8)
    
    # Inpaint only small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_hole_size, max_hole_size))
    mask_small = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_fill = mask & (~mask_small)
    
    # Convert depth to uint16 for inpainting
    depth_scaled = (depth * 1000).astype(np.uint16)
    depth_filled = cv2.inpaint(depth_scaled, mask_fill, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    depth_filled = depth_filled.astype(np.float32) / 1000.0
    
    # Keep original values where they exist
    depth_filled[depth > 0] = depth[depth > 0]
    
    return depth_filled


def validate_alignment(aligned_depth: np.ndarray, plane_points: Optional[np.ndarray] = None,
                      expected_distance: Optional[float] = None) -> dict:
    """
    Validate alignment quality.
    
    Args:
        aligned_depth: Aligned depth image
        plane_points: Optional Nx3 points on a known plane for RMSE calculation
        expected_distance: Expected distance to plane in meters
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'valid_ratio': np.sum(aligned_depth > 0) / aligned_depth.size,
        'mean_depth': np.mean(aligned_depth[aligned_depth > 0]) if np.any(aligned_depth > 0) else 0,
        'std_depth': np.std(aligned_depth[aligned_depth > 0]) if np.any(aligned_depth > 0) else 0,
    }
    
    if plane_points is not None and expected_distance is not None:
        # Calculate RMSE to plane
        distances = np.abs(plane_points[:, 2] - expected_distance)
        metrics['plane_rmse'] = np.sqrt(np.mean(distances**2))
        metrics['plane_max_error'] = np.max(distances)
    
    return metrics


if __name__ == '__main__':
    # Test with synthetic data
    import matplotlib.pyplot as plt
    
    # Create synthetic depth
    depth = np.random.rand(512, 512) * 2.0 + 1.0  # 1-3 meters
    depth[depth < 0.1] = 0  # Add some holes
    
    # Sample calibrations
    rgb_K = np.array([[2800, 0, 1920], [0, 2800, 1080], [0, 0, 1]], dtype=np.float64)
    rgb_D = np.zeros(8)
    depth_K = np.array([[365, 0, 256], [0, 365, 256], [0, 0, 1]], dtype=np.float64)
    depth_D = np.zeros(5)
    
    logging.basicConfig(level=logging.INFO)
    
    aligned = align_depth_to_rgb(
        depth, rgb_K, rgb_D, depth_K, depth_D,
        rgb_size=(3840, 2160),
        depth_unit='m'
    )
    
    print(f"Aligned depth shape: {aligned.shape}")
    print(f"Valid ratio: {np.sum(aligned > 0) / aligned.size:.2%}")
    print(f"Depth range: {np.min(aligned[aligned > 0]):.2f} - {np.max(aligned):.2f} m")