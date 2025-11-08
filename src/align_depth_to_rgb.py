"""
Align Depth to RGB Module (Improved with Sub-pixel Splatting)
Aligns depth images (512x512) to RGB resolution (3840x2160) with proper calibration.

Key improvements:
- Sub-pixel splatting: Bilinear distribution to 4-neighbor pixels
- Auto-direction probe: Automatically choose correct extrinsics direction
- Z-buffer with sub-pixel accuracy
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


def project_3d_to_image(points_3d: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to image coordinates.

    Args:
        points_3d: Nx3 array of 3D points
        K: 3x3 camera intrinsic matrix

    Returns:
        Tuple of (Nx2 array of image coordinates, valid mask)
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


def splat_depth_subpixel(rgb_coords: np.ndarray, depth_values: np.ndarray,
                         w_rgb: int, h_rgb: int) -> np.ndarray:
    """
    Splat depth values to RGB image with sub-pixel accuracy using bilinear weights.

    Each depth point is distributed to its 4-neighbor pixels with bilinear weights,
    and Z-buffer keeps the closest depth value for each pixel.

    Args:
        rgb_coords: Nx2 array of projected coordinates (u, v) in RGB image (float)
        depth_values: N array of depth values in meters
        w_rgb: RGB image width
        h_rgb: RGB image height

    Returns:
        Aligned depth image (h_rgb, w_rgb) in meters
    """
    aligned_depth = np.zeros((h_rgb, w_rgb), dtype=np.float32)
    z_buffer = np.full((h_rgb, w_rgb), np.inf, dtype=np.float32)

    x = rgb_coords[:, 0]
    y = rgb_coords[:, 1]
    z = depth_values

    # Floor coordinates
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts (bilinear weights)
    wx = x - x0
    wy = y - y0

    # 4-neighbor weights
    w00 = (1 - wx) * (1 - wy)  # (x0, y0)
    w10 = wx * (1 - wy)        # (x1, y0)
    w01 = (1 - wx) * wy        # (x0, y1)
    w11 = wx * wy              # (x1, y1)

    # Helper function: Splat to pixel with z-buffer
    def splat_to_pixel(ix: np.ndarray, iy: np.ndarray, weights: np.ndarray, z_vals: np.ndarray):
        """Splat depth values to pixels, keeping minimum z (closest)."""
        # Filter pixels within bounds and with non-negligible weight
        valid = (weights > 1e-8) & (ix >= 0) & (ix < w_rgb) & (iy >= 0) & (iy < h_rgb)
        if not np.any(valid):
            return

        ix_v = ix[valid]
        iy_v = iy[valid]
        z_v = z_vals[valid]

        # Update z-buffer: keep closest depth
        for i in range(len(ix_v)):
            xx, yy, zz = ix_v[i], iy_v[i], z_v[i]
            if zz < z_buffer[yy, xx]:
                z_buffer[yy, xx] = zz
                aligned_depth[yy, xx] = zz

    # Splat to 4 neighbors
    splat_to_pixel(x0, y0, w00, z)
    splat_to_pixel(x1, y0, w10, z)
    splat_to_pixel(x0, y1, w01, z)
    splat_to_pixel(x1, y1, w11, z)

    return aligned_depth


def probe_extrinsics_direction(
    u_depth: np.ndarray, v_depth: np.ndarray, depth_values: np.ndarray,
    depth_K: np.ndarray, rgb_K: np.ndarray,
    R_fwd: np.ndarray, t_fwd: np.ndarray,
    w_rgb: int, h_rgb: int,
    sample_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auto-detect correct extrinsics direction by comparing forward and inverse transformations.

    Tests both R,t (Depth→RGB) and R^T,-R^T*t (RGB→Depth) on a sample of points,
    and returns the one that projects more points in-bounds.

    Args:
        u_depth, v_depth: Depth pixel coordinates
        depth_values: Depth values in meters
        depth_K: Depth camera intrinsic matrix
        rgb_K: RGB camera intrinsic matrix
        R_fwd, t_fwd: Forward transformation (assumed Depth→RGB)
        w_rgb, h_rgb: RGB image dimensions
        sample_ratio: Fraction of points to sample for testing

    Returns:
        (R_best, t_best): Best extrinsics to use
    """
    # Sample points for speed
    n_total = len(u_depth)
    n_sample = max(1000, int(n_total * sample_ratio))
    n_sample = min(n_sample, n_total)

    if n_total > n_sample:
        indices = np.random.choice(n_total, n_sample, replace=False)
        u_s = u_depth[indices]
        v_s = v_depth[indices]
        z_s = depth_values[indices]
    else:
        u_s, v_s, z_s = u_depth, v_depth, depth_values

    # Backproject to 3D
    P_depth = backproject_depth(u_s, v_s, z_s, depth_K)

    # Compute inverse extrinsics (RGB→Depth)
    R_inv = R_fwd.T
    t_inv = -R_inv @ t_fwd

    def score_transform(R, t):
        """Score: ratio of points projecting in-bounds to RGB."""
        P_rgb = P_depth @ R.T + t.T
        uv_rgb, valid = project_3d_to_image(P_rgb, rgb_K)
        x_rgb, y_rgb = uv_rgb[:, 0], uv_rgb[:, 1]
        in_bounds = (x_rgb >= 0) & (x_rgb < w_rgb) & (y_rgb >= 0) & (y_rgb < h_rgb)
        return float(np.mean(valid & in_bounds))

    score_fwd = score_transform(R_fwd, t_fwd)
    score_inv = score_transform(R_inv, t_inv)

    logger.info(f"Auto-direction probe: forward={score_fwd:.3f}, inverse={score_inv:.3f}")

    if score_fwd >= score_inv:
        logger.info("Using forward extrinsics (Depth→RGB)")
        return R_fwd, t_fwd
    else:
        logger.info("Using inverse extrinsics (RGB→Depth equivalent)")
        return R_inv, t_inv


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
    use_simple_resize: bool = False,
    auto_direction: bool = True
) -> np.ndarray:
    """
    Align depth image to RGB image resolution and frame with sub-pixel accuracy.

    Pipeline:
    1. Undistort depth coordinates
    2. Backproject to 3D (depth camera frame)
    3. Transform to RGB camera frame (if T_d2r provided)
       - Optional: Auto-detect correct extrinsics direction
    4. Project to RGB image coordinates
    5. Sub-pixel splatting with Z-buffer
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
        auto_direction: Auto-detect extrinsics direction (forward vs inverse)

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

    # Undistort depth coordinates
    depth_points = np.stack([u_depth, v_depth], axis=-1)
    depth_points_undist = undistort_points(depth_points, depth_K, depth_D)
    u_depth, v_depth = depth_points_undist[:, 0], depth_points_undist[:, 1]
    logger.debug(f"Undistorted depth coordinates")

    # Auto-detect extrinsics direction (before backprojection for efficiency)
    R_use, t_use = None, None
    if T_d2r is not None:
        R_fwd, t_fwd = T_d2r
        if auto_direction:
            R_use, t_use = probe_extrinsics_direction(
                u_depth, v_depth, depth_values,
                depth_K, rgb_K,
                R_fwd, t_fwd,
                w_rgb, h_rgb,
                sample_ratio=0.1
            )
        else:
            R_use, t_use = R_fwd, t_fwd
            logger.info("Using provided extrinsics without auto-detection")

    # Backproject to 3D (depth camera frame)
    points_3d = backproject_depth(u_depth, v_depth, depth_values, depth_K)

    # Transform to RGB frame if needed
    if R_use is not None and t_use is not None:
        # Apply: P_color = R @ P_depth + t
        # Using matrix form: points @ R.T + t.T for vectorized operation
        points_3d = points_3d @ R_use.T + t_use.T  # t is (3,1), t.T is (1,3) for broadcasting

    # Project to RGB image
    rgb_coords, valid_proj = project_3d_to_image(points_3d, rgb_K)

    # Filter valid projections (z > 0)
    valid_proj &= (rgb_coords[:, 0] >= 0) & (rgb_coords[:, 0] < w_rgb - 1)
    valid_proj &= (rgb_coords[:, 1] >= 0) & (rgb_coords[:, 1] < h_rgb - 1)

    rgb_coords = rgb_coords[valid_proj]
    depth_values_proj = points_3d[valid_proj, 2]

    logger.debug(f"Valid projections: {len(depth_values_proj)}/{len(valid_proj)}")

    # Sub-pixel splatting with Z-buffer
    aligned_depth = splat_depth_subpixel(rgb_coords, depth_values_proj, w_rgb, h_rgb)

    logger.info(f"Filled pixels: {np.sum(aligned_depth > 0)}/{h_rgb*w_rgb} ({np.sum(aligned_depth > 0)/(h_rgb*w_rgb)*100:.1f}%)")

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

    if np.count_nonzero(mask) == 0:
        return depth

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
