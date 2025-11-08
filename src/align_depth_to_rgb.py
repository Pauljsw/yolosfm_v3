"""
Align Depth to RGB Module (Dense Completion Support)
Aligns depth images (512x512) to RGB resolution (3840x2160) with proper calibration.

Key features:
- Geometric alignment with extrinsics (Depth→RGB)
- Sub-pixel splatting (NN or bilinear)
- Safe hole filling with limits
- Dense completion via JBU (Joint Bilateral Upsampling) when RGB provided
- Confidence map support
- Optional plane filling
"""
import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ======================== Geometry Utilities ========================

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


# ======================== Splatting Methods ========================

def splat_nn(rgb_w: int, rgb_h: int, xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor splat with Z-buffer.

    Args:
        rgb_w, rgb_h: Output image dimensions
        xy: Nx2 array of projected coordinates (u, v)
        z: N array of depth values in meters

    Returns:
        Aligned depth image (rgb_h, rgb_w) in meters
    """
    out = np.zeros((rgb_h, rgb_w), dtype=np.float32)
    u = np.rint(xy[:, 0]).astype(np.int32)
    v = np.rint(xy[:, 1]).astype(np.int32)
    ok = (u >= 0) & (u < rgb_w) & (v >= 0) & (v < rgb_h)
    u, v, z = u[ok], v[ok], z[ok]

    for i in range(len(u)):
        if out[v[i], u[i]] == 0 or z[i] < out[v[i], u[i]]:
            out[v[i], u[i]] = z[i]

    return out


def splat_bilinear(rgb_w: int, rgb_h: int, xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Bilinear splat with Z-buffer.

    Distributes each depth point to 4-neighbor pixels with bilinear weights.
    Z-buffer keeps closest depth for each pixel.

    Args:
        rgb_w, rgb_h: Output image dimensions
        xy: Nx2 array of projected coordinates (u, v)
        z: N array of depth values in meters

    Returns:
        Aligned depth image (rgb_h, rgb_w) in meters
    """
    out = np.zeros((rgb_h, rgb_w), dtype=np.float32)

    for i in range(len(z)):
        x, y, zz = xy[i, 0], xy[i, 1], z[i]
        if not (zz > 0):
            continue

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1

        if x1 < 0 or y1 < 0 or x0 >= rgb_w or y0 >= rgb_h:
            continue

        # Bilinear weights
        wx = x - x0
        wy = y - y0

        # Distribute to 4 neighbors
        for (xx, yy, w) in [
            (x0, y0, (1 - wx) * (1 - wy)),
            (x1, y0, wx * (1 - wy)),
            (x0, y1, (1 - wx) * wy),
            (x1, y1, wx * wy),
        ]:
            if 0 <= xx < rgb_w and 0 <= yy < rgb_h and w > 1e-8:
                # Z-buffer: keep closest depth
                if out[yy, xx] == 0 or zz < out[yy, xx]:
                    out[yy, xx] = zz

    return out


# ======================== Hole Filling ========================

def fill_depth_holes(depth: np.ndarray, max_hole_size: int = 10,
                     max_fill_ratio: float = 0.15) -> np.ndarray:
    """
    Fill small holes in depth map using inpainting with safety limits.

    Args:
        depth: Depth image with holes (zero values)
        max_hole_size: Maximum hole size to fill (pixels)
        max_fill_ratio: Maximum ratio of image to fill (safety limit)

    Returns:
        Depth image with holes filled
    """
    h, w = depth.shape
    mask = (depth <= 0).astype(np.uint8)

    if np.count_nonzero(mask) == 0:
        return depth

    # Only fill small holes (morphological opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_hole_size, max_hole_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    small_holes = cv2.subtract(mask, opened)

    # Safety check: don't fill too much
    fill_ratio = float(small_holes.sum()) / (h * w)
    if fill_ratio > max_fill_ratio:
        logger.warning(f"Fill ratio {fill_ratio:.2%} exceeds limit {max_fill_ratio:.2%}, skipping hole fill")
        return depth

    # Inpaint small holes
    depth_scaled = (depth * 1000.0).astype(np.uint16)
    filled_scaled = cv2.inpaint(depth_scaled, small_holes, 3, cv2.INPAINT_TELEA)
    filled = filled_scaled.astype(np.float32) / 1000.0

    # Keep original measurements
    filled[depth > 0] = depth[depth > 0]

    return filled


# ======================== Dense Completion ========================

def jbu_complete_depth_to_dense(aligned_m: np.ndarray, rgb_bgr: np.ndarray,
                                d: int = 9, sigma_color: float = 75.0,
                                sigma_space: float = 75.0,
                                max_fill_ratio: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    RGB-guided Joint Bilateral Upsampling for dense depth completion.

    Fills all pixels using RGB guidance and simple diffusion, with confidence map.

    Args:
        aligned_m: Sparse aligned depth (meters)
        rgb_bgr: RGB image (BGR format)
        d: JBU window size
        sigma_color: JBU color sigma
        sigma_space: JBU space sigma
        max_fill_ratio: Maximum fill ratio for confidence scaling

    Returns:
        Tuple of (dense_depth, confidence_map)
    """
    H, W = aligned_m.shape
    has_measurement = aligned_m > 0
    meas_ratio = float(np.count_nonzero(has_measurement) / (H * W))
    conf_base = np.clip(meas_ratio / 0.15, 0.0, 1.0)

    dense = aligned_m.copy()

    # JBU on valid regions
    maxv = float(dense.max(initial=0.0))
    if maxv > 0:
        src8 = (dense / maxv * 255.0).astype(np.uint8)
        try:
            import cv2.ximgproc as xip
            guide = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
            jbu_result = xip.jointBilateralFilter(guide, src8, d, sigma_color, sigma_space)
            dense = (jbu_result.astype(np.float32) / 255.0) * maxv
        except Exception:
            # Fallback to standard bilateral filter
            bilateral_result = cv2.bilateralFilter(src8, d, sigma_color, sigma_space)
            dense = (bilateral_result.astype(np.float32) / 255.0) * maxv

    # Simple diffusion to fill remaining holes (3 iterations)
    for _ in range(3):
        zero_mask = dense <= 0
        if not np.any(zero_mask):
            break
        avg = cv2.blur(dense, (3, 3))
        dense[zero_mask] = avg[zero_mask]

    # Restore original measurements
    dense[has_measurement] = aligned_m[has_measurement]

    # Confidence map
    confidence = np.full((H, W), conf_base * 0.7, dtype=np.float32)
    confidence[has_measurement] = 1.0

    # Boost confidence near edges (RGB boundaries likely align with depth boundaries)
    edges = cv2.Canny(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY), 50, 150) > 0
    confidence[edges] = np.maximum(confidence[edges], conf_base * 0.85)

    # Scale confidence if too much was filled
    fill_ratio = 1.0 - meas_ratio
    if fill_ratio > max_fill_ratio:
        confidence *= (max_fill_ratio / fill_ratio)

    return dense.astype(np.float32), np.clip(confidence, 0, 1)


def ransac_plane_fill(aligned_m: np.ndarray, K: np.ndarray,
                      tol: float = 0.02, zmax: float = 10.0) -> np.ndarray:
    """
    Fill large holes using RANSAC plane fitting (optional, for planar scenes).

    Args:
        aligned_m: Sparse aligned depth
        K: Camera intrinsic matrix
        tol: RANSAC inlier tolerance (meters)
        zmax: Maximum valid depth (meters)

    Returns:
        Depth with plane-filled holes
    """
    H, W = aligned_m.shape
    yy, xx = np.where(aligned_m > 0)
    z = aligned_m[yy, xx]

    if len(z) < 500:
        logger.warning("Not enough points for plane fitting, skipping")
        return aligned_m

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xx - cx) * z / fx
    Y = (yy - cy) * z / fy
    P = np.stack([X, Y, z], axis=1)

    # RANSAC plane fitting
    rng = np.random.default_rng(42)
    best_inliers = 0
    best_abc = (0, 0, 0)

    for _ in range(200):
        idx = rng.choice(len(P), 3, replace=False)
        A = np.c_[P[idx, 0], P[idx, 1], np.ones(3)]
        b = P[idx, 2]
        try:
            a, b_coef, c = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        z_pred = P[:, 0] * a + P[:, 1] * b_coef + c
        n_inliers = int(np.sum(np.abs(z_pred - P[:, 2]) < tol))

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_abc = (a, b_coef, c)

    a, b_coef, c = best_abc
    logger.info(f"Plane fitting: {best_inliers}/{len(P)} inliers ({100*best_inliers/len(P):.1f}%)")

    # Fill holes with plane
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    Xf = (xs - cx) / fx
    Yf = (ys - cy) / fy
    z_plane = a * Xf + b_coef * Yf + c
    z_plane[(z_plane <= 0) | (z_plane > zmax)] = 0

    out = aligned_m.copy()
    zero_mask = out <= 0
    out[zero_mask] = z_plane[zero_mask].astype(np.float32)

    return out


# ======================== Main Alignment Function ========================

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
    joint_bilateral: bool = False,
    bilateral_params: Optional[dict] = None,
    use_simple_resize: bool = False,
    rgb_img: Optional[np.ndarray] = None,
    splat_mode: str = 'bilinear',
    do_dense: bool = False,
    plane_fill: bool = False,
    undistort_depth: bool = False
) -> np.ndarray:
    """
    Align depth image to RGB image resolution and frame.

    Pipeline:
    1. Undistort depth (optional)
    2. Backproject to 3D (depth camera frame)
    3. Transform to RGB camera frame
    4. Project to RGB image coordinates
    5. Splat with Z-buffer (NN or bilinear)
    6. Hole filling (optional, safe limits)
    7. Dense completion via JBU (optional, requires rgb_img)
    8. Plane filling (optional)

    Args:
        depth_img: HxW depth image (e.g., 512x512)
        rgb_K: 3x3 RGB camera intrinsic matrix
        rgb_D: RGB camera distortion coefficients
        depth_K: 3x3 depth camera intrinsic matrix
        depth_D: Depth camera distortion coefficients
        rgb_size: (width, height) of RGB image (e.g., 3840x2160)
        T_d2r: Optional (R, t) transformation from depth to RGB frame
        depth_unit: 'm' or 'mm' or 'auto'
        hole_fill: Whether to fill small holes
        joint_bilateral: Legacy parameter (use do_dense instead)
        bilateral_params: Parameters for bilateral filter
        use_simple_resize: Use simple resize (for hardware-aligned depth)
        rgb_img: Optional RGB image for dense completion (BGR format)
        splat_mode: 'nn' or 'bilinear'
        do_dense: Enable dense completion via JBU (requires rgb_img)
        plane_fill: Enable plane-based hole filling
        undistort_depth: Apply depth undistortion

    Returns:
        Aligned depth image at RGB resolution (H_rgb x W_rgb), in meters
    """
    h_depth, w_depth = depth_img.shape
    w_rgb, h_rgb = rgb_size

    logger.info(f"Aligning depth {w_depth}x{h_depth} to RGB {w_rgb}x{h_rgb}")
    logger.info(f"  Mode: splat={splat_mode}, dense={do_dense}, plane_fill={plane_fill}")

    # Auto-detect depth unit if requested
    if depth_unit == 'auto':
        max_val = float(depth_img.max(initial=0))
        if max_val > 100:
            depth_unit = 'mm'
        else:
            depth_unit = 'm'
        logger.info(f"Auto-detected depth unit: {depth_unit}")

    # Convert depth to meters
    depth_m = depth_img.astype(np.float32)
    if depth_unit == 'mm':
        depth_m /= 1000.0

    # FAST PATH: For hardware-aligned depth (simple resize)
    if use_simple_resize:
        logger.info("Using simple resize (hardware-aligned depth)")
        aligned_depth = cv2.resize(depth_m, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)

        if hole_fill:
            aligned_depth = fill_depth_holes(aligned_depth)

        # Legacy bilateral filter support
        if joint_bilateral and not do_dense:
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

    # Undistort depth (optional)
    if undistort_depth and depth_D is not None and len(depth_D) > 0 and np.any(depth_D != 0):
        logger.info("Undistorting depth image")
        depth_m = cv2.undistort(depth_m, depth_K, depth_D)

    # Create coordinate grids
    u_depth, v_depth = np.meshgrid(np.arange(w_depth), np.arange(h_depth))
    u_depth = u_depth.flatten().astype(np.float32)
    v_depth = v_depth.flatten().astype(np.float32)
    depth_values = depth_m.flatten()

    # Filter valid depths
    valid_mask = depth_values > 0
    u_depth = u_depth[valid_mask]
    v_depth = v_depth[valid_mask]
    depth_values = depth_values[valid_mask]

    if len(depth_values) == 0:
        logger.warning("No valid depth values found")
        return np.zeros((h_rgb, w_rgb), dtype=np.float32)

    logger.debug(f"Valid depth pixels: {len(depth_values)}/{h_depth*w_depth} ({100*len(depth_values)/(h_depth*w_depth):.1f}%)")

    # Backproject to 3D (depth camera frame)
    points_3d = backproject_depth(u_depth, v_depth, depth_values, depth_K)

    # Transform to RGB frame if extrinsics provided
    if T_d2r is not None:
        R, t = T_d2r
        # P_rgb = R @ P_depth + t
        points_3d = points_3d @ R.T + t.reshape(1, 3)

    # Project to RGB image
    rgb_coords, valid_proj = project_3d_to_image(points_3d, rgb_K)

    # Filter valid projections
    valid_proj &= (rgb_coords[:, 0] >= 0) & (rgb_coords[:, 0] < w_rgb)
    valid_proj &= (rgb_coords[:, 1] >= 0) & (rgb_coords[:, 1] < h_rgb)

    rgb_coords = rgb_coords[valid_proj]
    depth_values_proj = points_3d[valid_proj, 2]

    logger.debug(f"Valid projections: {len(depth_values_proj)}/{len(valid_proj)} ({100*np.sum(valid_proj)/len(valid_proj):.1f}%)")

    # Splat with Z-buffer
    if splat_mode == 'bilinear':
        aligned_depth = splat_bilinear(w_rgb, h_rgb, rgb_coords, depth_values_proj)
    else:
        aligned_depth = splat_nn(w_rgb, h_rgb, rgb_coords, depth_values_proj)

    coverage = np.sum(aligned_depth > 0) / (h_rgb * w_rgb)
    logger.info(f"Splat coverage: {coverage:.1%}")

    # Hole filling (small holes only, with safety limits)
    if hole_fill:
        aligned_depth = fill_depth_holes(aligned_depth, max_hole_size=10, max_fill_ratio=0.15)
        logger.debug("Hole filling completed")

    # Dense completion (requires RGB image)
    if do_dense:
        if rgb_img is not None:
            logger.info("Performing dense completion with JBU")
            if bilateral_params is None:
                bilateral_params = {'d': 9, 'sigma_color': 75, 'sigma_space': 75}

            aligned_depth, confidence = jbu_complete_depth_to_dense(
                aligned_depth, rgb_img,
                d=bilateral_params.get('d', 9),
                sigma_color=bilateral_params.get('sigma_color', 75),
                sigma_space=bilateral_params.get('sigma_space', 75)
            )
            logger.debug(f"Dense completion: mean confidence = {confidence.mean():.2f}")
        else:
            logger.warning("Dense completion requested but no RGB image provided, skipping")

    # Plane filling (optional, for planar scenes)
    if plane_fill:
        logger.info("Applying plane-based hole filling")
        aligned_depth = ransac_plane_fill(aligned_depth, rgb_K, tol=0.02, zmax=10.0)

    final_coverage = np.sum(aligned_depth > 0) / (h_rgb * w_rgb)
    logger.info(f"Final coverage: {final_coverage:.1%}")

    return aligned_depth


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
        depth_unit='m',
        splat_mode='bilinear'
    )

    print(f"Aligned depth shape: {aligned.shape}")
    print(f"Valid ratio: {np.sum(aligned > 0) / aligned.size:.2%}")
    print(f"Depth range: {np.min(aligned[aligned > 0]):.2f} - {np.max(aligned):.2f} m")
