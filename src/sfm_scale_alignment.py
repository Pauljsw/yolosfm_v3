"""
Phase 2: SFM Scale Alignment
Aligns SFM poses to absolute scale using depth-derived ground truth.
Uses Umeyama/Procrustes algorithm for similarity transformation estimation.
"""
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    logger.warning("Open3D not installed. Point cloud features unavailable.")
    HAS_OPEN3D = False


def umeyama_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
    estimate_scale: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama algorithm for similarity transformation estimation.

    Computes s, R, t such that: target = s * R @ source + t

    Args:
        source_points: Source points (N, 3)
        target_points: Target points (N, 3)
        estimate_scale: Whether to estimate scale

    Returns:
        (scale, rotation, translation)
    """
    assert source_points.shape == target_points.shape, "Point sets must have same shape"
    assert source_points.shape[1] == 3, "Points must be 3D"

    n = source_points.shape[0]

    # Compute centroids
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Center the points
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute scale
    if estimate_scale:
        source_var = np.sum(source_centered ** 2) / n
        target_var = np.sum(target_centered ** 2) / n
        scale = np.sqrt(target_var / source_var)
    else:
        scale = 1.0

    # Compute rotation using SVD
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        logger.warning("Reflection detected in Umeyama, correcting...")
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = target_centroid - scale * R @ source_centroid

    logger.info(f"Umeyama alignment: scale={scale:.4f}")
    logger.debug(f"Rotation:\n{R}")
    logger.debug(f"Translation: {t}")

    return scale, R, t


def extract_sparse_points_from_colmap(sparse_model_dir: str) -> np.ndarray:
    """
    Extract 3D points from COLMAP sparse reconstruction.
    Supports both text (.txt) and binary (.bin) formats.

    Args:
        sparse_model_dir: COLMAP sparse model directory

    Returns:
        points3D array (N, 3)
    """
    import struct

    sparse_path = Path(sparse_model_dir)
    points3d_bin = sparse_path / 'points3D.bin'
    points3d_txt = sparse_path / 'points3D.txt'

    # Try binary format first (more common)
    if points3d_bin.exists():
        logger.info(f"Reading COLMAP binary format: {points3d_bin}")
        points = []

        with open(points3d_bin, 'rb') as f:
            # Read number of points
            num_points = struct.unpack('Q', f.read(8))[0]

            for _ in range(num_points):
                # point3D_id (uint64)
                point_id = struct.unpack('Q', f.read(8))[0]

                # xyz (3 x double)
                x, y, z = struct.unpack('ddd', f.read(24))
                points.append([x, y, z])

                # rgb (3 x uint8)
                r, g, b = struct.unpack('BBB', f.read(3))

                # error (double)
                error = struct.unpack('d', f.read(8))[0]

                # track length (uint64)
                track_length = struct.unpack('Q', f.read(8))[0]

                # skip track elements (image_id + point2D_idx = 4 + 4 bytes each)
                f.read(track_length * 8)

        points = np.array(points)
        logger.info(f"Extracted {len(points)} sparse points from COLMAP (binary)")

    # Try text format as fallback
    elif points3d_txt.exists():
        logger.info(f"Reading COLMAP text format: {points3d_txt}")
        points = []

        with open(points3d_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                # Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append([x, y, z])

        points = np.array(points)
        logger.info(f"Extracted {len(points)} sparse points from COLMAP (text)")

    else:
        raise FileNotFoundError(
            f"COLMAP points3D file not found in {sparse_model_dir}. "
            f"Tried: points3D.bin, points3D.txt"
        )

    return points


def depth_to_pointcloud(
    depth_img: np.ndarray,
    K: np.ndarray,
    rgb_img: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert depth image to 3D point cloud.

    Args:
        depth_img: Depth image (H, W) in meters
        K: Camera intrinsic matrix (3, 3)
        rgb_img: Optional RGB image (H, W, 3)

    Returns:
        points (N, 3) or (N, 6) if RGB provided
    """
    h, w = depth_img.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    z = depth_img.flatten()

    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)

    if rgb_img is not None:
        colors = rgb_img.reshape(-1, 3)[valid]
        points = np.concatenate([points, colors], axis=-1)

    return points


def match_point_clouds_fpfh(
    source_pcd: 'o3d.geometry.PointCloud',
    target_pcd: 'o3d.geometry.PointCloud',
    voxel_size: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match two point clouds using FPFH features and RANSAC.

    Args:
        source_pcd: Source point cloud
        target_pcd: Target point cloud
        voxel_size: Voxel size for downsampling

    Returns:
        (source_points, target_points) correspondence arrays (N, 3)
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D required for point cloud matching")

    # Downsample
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # Compute FPFH features
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )

    # RANSAC registration
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    # Extract correspondences
    correspondences = np.asarray(result.correspondence_set)

    if len(correspondences) < 3:
        raise ValueError(f"Insufficient correspondences: {len(correspondences)}")

    source_points = np.asarray(source_down.points)[correspondences[:, 0]]
    target_points = np.asarray(target_down.points)[correspondences[:, 1]]

    logger.info(f"Found {len(correspondences)} correspondences")

    return source_points, target_points


def extract_frame_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract frame ID from RGB filename.

    Example: "camera_RGB_1758853283_533442048.png" → "1758853283_533442048"

    Args:
        filename: Image filename

    Returns:
        Frame ID string or None if not matched
    """
    import re
    match = re.match(r'camera_RGB_(\d+_\d+)', filename)
    if match:
        return match.group(1)
    return None


def match_trajectory_and_poses(
    depth_trajectory: List[Dict],
    sfm_poses: Dict
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Match camera positions between depth trajectory and SFM poses.

    Args:
        depth_trajectory: List of dicts with 'frame_id' and 'T_world_cam'
        sfm_poses: Dict of {filename: {'R': ..., 't': ...}}

    Returns:
        (depth_camera_centers, sfm_camera_centers) as lists of (3,) arrays
    """
    depth_centers = []
    sfm_centers = []
    matched_count = 0

    # Build frame_id → SFM pose mapping
    sfm_frame_map = {}
    for img_name, pose_data in sfm_poses.items():
        filename = pose_data.get('filename', img_name)
        frame_id = extract_frame_id_from_filename(filename)
        if frame_id:
            sfm_frame_map[frame_id] = pose_data

    # Match trajectories
    for traj_item in depth_trajectory:
        frame_id = traj_item['frame_id']

        if frame_id in sfm_frame_map:
            # Extract depth camera center from T_world_cam
            T_world_cam = np.array(traj_item['T_world_cam'])
            depth_center = T_world_cam[:3, 3]  # Translation part

            # Extract SFM camera center from R, t
            sfm_pose = sfm_frame_map[frame_id]
            R = np.array(sfm_pose['R'])
            t = np.array(sfm_pose['t']).reshape(3)
            sfm_center = -R.T @ t  # Camera center in world coordinates

            depth_centers.append(depth_center)
            sfm_centers.append(sfm_center)
            matched_count += 1

    logger.info(f"Matched {matched_count} camera frames (from {len(depth_trajectory)} depth, {len(sfm_poses)} SFM)")

    if matched_count < 3:
        raise ValueError(f"Insufficient matched frames: {matched_count}, need at least 3")

    return np.array(depth_centers), np.array(sfm_centers)


def align_sfm_to_depth_groundtruth(
    sfm_sparse_points: np.ndarray,
    depth_gt_pointcloud: np.ndarray,
    sfm_poses: Optional[Dict] = None,
    depth_trajectory: Optional[List[Dict]] = None,
    use_camera_trajectory: bool = True,
    use_feature_matching: bool = True,
    max_points: int = 10000
) -> Dict:
    """
    Align SFM sparse reconstruction to depth ground truth.

    Tries methods in order:
    1. Camera trajectory alignment (most reliable if available)
    2. FPFH feature matching
    3. Error - no fallback to bad centroid matching

    Args:
        sfm_sparse_points: SFM sparse points (N, 3)
        depth_gt_pointcloud: Depth ground truth points (M, 3)
        sfm_poses: SFM poses dict (required if use_camera_trajectory=True)
        depth_trajectory: Depth trajectory list (required if use_camera_trajectory=True)
        use_camera_trajectory: Try camera trajectory alignment first
        use_feature_matching: Try FPFH feature matching
        max_points: Maximum points to use for alignment

    Returns:
        Dictionary with scale, rotation, translation
    """
    source_corr = None
    target_corr = None
    alignment_method = "unknown"

    # Method 1: Camera trajectory alignment (BEST)
    if use_camera_trajectory and sfm_poses is not None and depth_trajectory is not None:
        logger.info("Using camera trajectory alignment (most robust)...")
        try:
            depth_centers, sfm_centers = match_trajectory_and_poses(depth_trajectory, sfm_poses)
            source_corr = sfm_centers
            target_corr = depth_centers
            alignment_method = "camera_trajectory"
            logger.info(f"Using {len(source_corr)} matched camera positions")
        except Exception as e:
            logger.warning(f"Camera trajectory alignment failed: {e}")
            source_corr = None

    # Method 2: FPFH feature matching (FALLBACK)
    if source_corr is None and use_feature_matching and HAS_OPEN3D:
        logger.info("Using FPFH feature matching for alignment...")

        # Subsample if too many points
        if len(sfm_sparse_points) > max_points:
            indices = np.random.choice(len(sfm_sparse_points), max_points, replace=False)
            sfm_pts = sfm_sparse_points[indices]
        else:
            sfm_pts = sfm_sparse_points

        if len(depth_gt_pointcloud) > max_points:
            indices = np.random.choice(len(depth_gt_pointcloud), max_points, replace=False)
            depth_pts = depth_gt_pointcloud[indices]
        else:
            depth_pts = depth_gt_pointcloud

        sfm_pcd = o3d.geometry.PointCloud()
        sfm_pcd.points = o3d.utility.Vector3dVector(sfm_pts)

        depth_pcd = o3d.geometry.PointCloud()
        depth_pcd.points = o3d.utility.Vector3dVector(depth_pts)

        try:
            source_corr, target_corr = match_point_clouds_fpfh(sfm_pcd, depth_pcd, voxel_size=0.02)
            alignment_method = "fpfh"
        except Exception as e:
            logger.warning(f"FPFH matching failed: {e}")
            source_corr = None

    # No valid method succeeded
    if source_corr is None:
        raise RuntimeError(
            "All alignment methods failed. Cannot proceed with scale alignment.\n"
            "Tried: camera_trajectory=" + str(use_camera_trajectory) +
            ", fpfh=" + str(use_feature_matching)
        )

    # Run Umeyama alignment
    logger.info("Running Umeyama alignment...")
    scale, R, t = umeyama_alignment(source_corr, target_corr, estimate_scale=True)

    # Compute alignment error
    aligned_source = scale * (R @ source_corr.T).T + t
    errors = np.linalg.norm(aligned_source - target_corr, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))

    logger.info(f"Alignment RMSE: {rmse:.4f} meters")
    logger.info(f"Estimated scale: {scale:.4f}")

    return {
        'scale': float(scale),
        'rotation': R.tolist(),
        'translation': t.tolist(),
        'rmse': float(rmse),
        'num_correspondences': len(source_corr),
        'alignment_method': alignment_method
    }


def apply_alignment_to_poses(
    poses: Dict,
    scale: float,
    R_align: np.ndarray,
    t_align: np.ndarray
) -> Dict:
    """
    Apply global alignment to all SFM poses.

    Transform: R' = R_align @ R
               t' = scale * (R_align @ t) + t_align

    Args:
        poses: Dictionary of poses {image_name: {'R': ..., 't': ...}}
        scale: Scale factor
        R_align: Alignment rotation (3, 3)
        t_align: Alignment translation (3,)

    Returns:
        Aligned poses dictionary
    """
    aligned_poses = {}

    for img_name, pose_data in poses.items():
        R = np.array(pose_data['R'])
        t = np.array(pose_data['t']).reshape(3)

        # Apply alignment
        R_new = R_align @ R
        t_new = scale * (R_align @ t) + t_align

        aligned_poses[img_name] = {
            'filename': pose_data['filename'],
            'R': R_new.tolist(),
            't': t_new.reshape(3, 1).tolist()
        }

        # Preserve K if present
        if 'K' in pose_data:
            aligned_poses[img_name]['K'] = pose_data['K']

    logger.info(f"Applied alignment to {len(aligned_poses)} poses")

    return aligned_poses


def run_sfm_scale_alignment(
    sfm_poses_path: str,
    sfm_sparse_dir: str,
    depth_gt_pcd_path: str,
    output_poses_path: str,
    depth_trajectory_path: Optional[str] = None,
    use_camera_trajectory: bool = True,
    use_feature_matching: bool = True
) -> Dict:
    """
    Complete SFM scale alignment pipeline.

    Args:
        sfm_poses_path: Path to SFM poses.json
        sfm_sparse_dir: Path to COLMAP sparse model directory
        depth_gt_pcd_path: Path to depth ground truth point cloud (.ply)
        output_poses_path: Output path for aligned poses
        depth_trajectory_path: Path to depth trajectory.json (from Phase 1)
        use_camera_trajectory: Try camera trajectory alignment first
        use_feature_matching: Try FPFH feature matching as fallback

    Returns:
        Alignment results dictionary
    """
    logger.info("=" * 80)
    logger.info("Phase 2: SFM Scale Alignment")
    logger.info("=" * 80)

    # Load SFM poses
    logger.info(f"Loading SFM poses: {sfm_poses_path}")
    with open(sfm_poses_path, 'r') as f:
        sfm_poses = json.load(f)
    logger.info(f"Loaded {len(sfm_poses)} poses")

    # Load depth trajectory (if available)
    depth_trajectory = None
    if use_camera_trajectory and depth_trajectory_path:
        from pathlib import Path
        traj_path = Path(depth_trajectory_path)
        if traj_path.exists():
            logger.info(f"Loading depth trajectory: {depth_trajectory_path}")
            with open(traj_path, 'r') as f:
                depth_trajectory = json.load(f)
            logger.info(f"Loaded {len(depth_trajectory)} trajectory frames")
        else:
            logger.warning(f"Trajectory file not found: {depth_trajectory_path}, will skip camera trajectory alignment")

    # Extract SFM sparse points
    logger.info(f"Extracting COLMAP sparse points: {sfm_sparse_dir}")
    sfm_sparse_points = extract_sparse_points_from_colmap(sfm_sparse_dir)

    # Load depth ground truth
    logger.info(f"Loading depth ground truth: {depth_gt_pcd_path}")
    if HAS_OPEN3D:
        depth_pcd = o3d.io.read_point_cloud(depth_gt_pcd_path)
        depth_gt_points = np.asarray(depth_pcd.points)
    else:
        # Fallback: manual PLY parsing
        raise NotImplementedError("Manual PLY parsing not implemented")

    logger.info(f"Loaded {len(depth_gt_points)} ground truth points")

    # Run alignment
    alignment_result = align_sfm_to_depth_groundtruth(
        sfm_sparse_points,
        depth_gt_points,
        sfm_poses=sfm_poses,
        depth_trajectory=depth_trajectory,
        use_camera_trajectory=use_camera_trajectory,
        use_feature_matching=use_feature_matching
    )

    # Apply alignment to poses
    scale = alignment_result['scale']
    R_align = np.array(alignment_result['rotation'])
    t_align = np.array(alignment_result['translation'])

    aligned_poses = apply_alignment_to_poses(sfm_poses, scale, R_align, t_align)

    # Save aligned poses
    output_path = Path(output_poses_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(aligned_poses, f, indent=2)

    logger.info(f"Saved aligned poses: {output_path}")

    # Save alignment metadata
    metadata_path = output_path.parent / 'alignment_metadata.json'
    metadata = {
        'scale': alignment_result['scale'],
        'rotation': alignment_result['rotation'],
        'translation': alignment_result['translation'],
        'rmse': alignment_result['rmse'],
        'num_correspondences': alignment_result['num_correspondences'],
        'alignment_method': alignment_result.get('alignment_method', 'unknown'),
        'sfm_poses_path': sfm_poses_path,
        'depth_gt_path': depth_gt_pcd_path
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved alignment metadata: {metadata_path}")

    logger.info("=" * 80)
    logger.info("SFM Scale Alignment Complete!")
    logger.info(f"Scale factor: {scale:.4f}")
    logger.info(f"Alignment RMSE: {alignment_result['rmse']:.4f} m")
    logger.info("=" * 80)

    return alignment_result


if __name__ == '__main__':
    # Test
    import argparse
    import sys
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='SFM scale alignment')
    parser.add_argument('--sfm-poses', required=True, help='SFM poses.json')
    parser.add_argument('--sfm-sparse', required=True, help='COLMAP sparse directory')
    parser.add_argument('--depth-gt', required=True, help='Depth ground truth .ply')
    parser.add_argument('--output', required=True, help='Output aligned poses.json')
    parser.add_argument('--no-feature-matching', action='store_true',
                       help='Disable FPFH feature matching')

    args = parser.parse_args()

    setup_logging('INFO')

    try:
        result = run_sfm_scale_alignment(
            args.sfm_poses,
            args.sfm_sparse,
            args.depth_gt,
            args.output,
            use_feature_matching=not args.no_feature_matching
        )

        print(f"\n✅ Alignment complete!")
        print(f"   Scale: {result['scale']:.4f}")
        print(f"   RMSE: {result['rmse']:.4f} m")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Alignment failed: {e}", exc_info=True)
        sys.exit(1)
