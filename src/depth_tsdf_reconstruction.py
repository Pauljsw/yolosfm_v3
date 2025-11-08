"""
Phase 1: Depth-only Ground Truth Reconstruction with Robust ICP Odometry
Generates absolute-scale 3D model from depth images using TSDF fusion.
This serves as the ground truth for SFM scale alignment.

Features:
- Robust point-to-plane ICP for accurate odometry
- Optional undistortion for depth images
- Fitness and RMSE reporting per frame
- Proper camera pose accumulation
- Highly configurable parameters
"""
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    logger.warning("Open3D not installed. TSDF reconstruction unavailable.")
    HAS_OPEN3D = False


def create_undistortion_map(K: np.ndarray, D: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create undistortion maps for depth image rectification.

    Args:
        K: Camera intrinsic matrix (3x3)
        D: Distortion coefficients
        width: Image width
        height: Image height

    Returns:
        (map1, map2): Undistortion maps for cv2.remap
    """
    # Use the same K for both input and output (preserve field of view)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (width, height), cv2.CV_32FC1
    )
    return map1, map2


def undistort_depth_image(depth: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    """
    Apply undistortion to depth image.

    Args:
        depth: Depth image
        map1, map2: Undistortion maps from create_undistortion_map

    Returns:
        Undistorted depth image
    """
    return cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)


def depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    depth_trunc: float = 10.0
) -> o3d.geometry.PointCloud:
    """
    Convert depth image to Open3D point cloud (depth-only, no color).

    Args:
        depth: Depth image in meters
        K: Camera intrinsic matrix
        depth_trunc: Maximum depth value to consider

    Returns:
        Open3D PointCloud (without colors)
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten().astype(np.float32)
    v = v.flatten().astype(np.float32)
    z = depth.flatten().astype(np.float32)

    # Filter valid depths
    valid = (z > 0) & (z < depth_trunc)
    u, v, z = u[valid], v[valid], z[valid]

    # Backproject to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)

    # Create point cloud (no color information)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def robust_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    initial_transform: np.ndarray,
    voxel_size: float = 0.02,
    max_correspondence_distance: float = 0.05,
    max_iterations: int = 50
) -> Tuple[np.ndarray, float, float]:
    """
    Robust point-to-plane ICP registration.

    Args:
        source_pcd: Source point cloud (current frame)
        target_pcd: Target point cloud (previous frame)
        initial_transform: Initial transformation guess (4x4)
        voxel_size: Voxel size for downsampling
        max_correspondence_distance: Max distance for point correspondence
        max_iterations: Maximum ICP iterations

    Returns:
        (transformation, fitness, rmse): 4x4 transformation matrix, fitness score, and RMSE
    """
    # Downsample for efficiency
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    # Estimate normals for point-to-plane ICP
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )

    # Point-to-plane ICP
    reg_result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        )
    )

    return reg_result.transformation, reg_result.fitness, reg_result.inlier_rmse


class DepthTSDFReconstructor:
    """TSDF-based depth reconstruction with robust odometry"""

    def __init__(
        self,
        tsdf_voxel_length: float = 0.01,     # 1cm TSDF voxels
        tsdf_trunc_factor: float = 4.0,       # Truncation = voxel_length * factor
        depth_scale: float = 1.0,             # 1.0 for meters, 1000.0 for mm->m conversion
        depth_trunc: float = 10.0,            # Max depth 10m
        icp_voxel_size: float = 0.02,         # 2cm voxels for ICP downsampling
        icp_max_corr_dist: float = 0.05,      # 5cm max correspondence distance
        use_undistortion: bool = False,       # Enable depth undistortion
        K: Optional[np.ndarray] = None,       # Required if use_undistortion=True
        D: Optional[np.ndarray] = None,       # Required if use_undistortion=True
        width: int = 512,                     # Depth image width
        height: int = 512,                    # Depth image height
    ):
        """
        Initialize TSDF reconstructor with robust odometry.

        Args:
            tsdf_voxel_length: TSDF voxel size in meters
            tsdf_trunc_factor: Truncation distance = voxel_length * factor
            depth_scale: Depth scale (1.0 for meters, 1000.0 for mm)
            depth_trunc: Maximum valid depth in meters
            icp_voxel_size: Voxel size for ICP downsampling
            icp_max_corr_dist: Maximum correspondence distance for ICP
            use_undistortion: Whether to undistort depth images
            K: Camera intrinsic matrix (required if undistortion enabled)
            D: Distortion coefficients (required if undistortion enabled)
            width: Depth image width
            height: Depth image height
        """
        if not HAS_OPEN3D:
            raise ImportError("Open3D is required for TSDF reconstruction")

        self.tsdf_voxel_length = tsdf_voxel_length
        self.tsdf_trunc = tsdf_voxel_length * tsdf_trunc_factor
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.icp_voxel_size = icp_voxel_size
        self.icp_max_corr_dist = icp_max_corr_dist
        self.use_undistortion = use_undistortion

        # Create TSDF volume (no color - depth only)
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=tsdf_voxel_length,
            sdf_trunc=self.tsdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )

        # Odometry state
        self.T_world_cam = np.eye(4)  # Current camera pose in world frame
        self.prev_pcd = None          # Previous frame point cloud for ICP
        self.trajectory = []          # Store all poses
        self.odometry_log = []        # Store ICP fitness/RMSE per frame

        # Undistortion maps
        self.undist_map1 = None
        self.undist_map2 = None
        if use_undistortion:
            if K is None or D is None:
                raise ValueError("K and D must be provided when use_undistortion=True")
            self.undist_map1, self.undist_map2 = create_undistortion_map(K, D, width, height)
            logger.info("Created undistortion maps for depth images")

    def integrate_frame(
        self,
        depth_img: np.ndarray,
        K: np.ndarray,
        frame_id: str,
        use_icp: bool = True
    ) -> Dict:
        """
        Integrate one depth frame into TSDF volume with ICP odometry.

        Note: This is depth-only reconstruction. No RGB/color information is used.

        Args:
            depth_img: Depth image (H, W), float32 in meters
            K: Camera intrinsic matrix (3, 3)
            frame_id: Frame identifier
            use_icp: Whether to use ICP for pose estimation

        Returns:
            Dictionary with odometry statistics
        """
        h, w = depth_img.shape

        # Apply undistortion if enabled
        if self.use_undistortion:
            depth_img = undistort_depth_image(depth_img, self.undist_map1, self.undist_map2)

        # Convert to point cloud for ICP (depth only, no color)
        current_pcd = depth_to_pointcloud(
            depth_img, K, self.depth_trunc
        )

        # Estimate camera pose using ICP
        fitness, rmse = 0.0, 0.0
        if use_icp and self.prev_pcd is not None:
            try:
                # Run ICP to get relative transformation from previous to current
                T_prev_to_curr, fitness, rmse = robust_icp(
                    current_pcd,
                    self.prev_pcd,
                    initial_transform=np.eye(4),  # Identity as initial guess
                    voxel_size=self.icp_voxel_size,
                    max_correspondence_distance=self.icp_max_corr_dist
                )

                # Update world pose: T_world_cam = T_world_cam @ T_prev_to_curr
                self.T_world_cam = self.T_world_cam @ T_prev_to_curr

                logger.debug(f"ICP for {frame_id}: fitness={fitness:.3f}, RMSE={rmse:.4f}m")

            except Exception as e:
                logger.warning(f"ICP failed for {frame_id}: {e}. Using identity transform.")
                fitness, rmse = 0.0, 0.0
        else:
            # First frame or ICP disabled - keep current pose
            logger.debug(f"Frame {frame_id}: No ICP (first frame or disabled)")

        # Convert depth to Open3D format for TSDF integration
        # Open3D expects depth in depth_scale units (e.g., mm if scale=1000)
        depth_o3d_array = (depth_img * self.depth_scale).astype(np.uint16)
        depth_o3d = o3d.geometry.Image(depth_o3d_array)

        # Create dummy color image (required by RGBDImage API, but ignored by NoColor TSDF)
        dummy_color = np.zeros((h, w, 3), dtype=np.uint8)
        color_o3d = o3d.geometry.Image(dummy_color)

        # Create RGBD image (color will be ignored since TSDF is NoColor type)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )

        # Create intrinsic
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2]
        )

        # Integrate into TSDF volume
        # Open3D expects camera-to-world pose (inverse of world-to-camera)
        T_cam_world = np.linalg.inv(self.T_world_cam)
        self.volume.integrate(rgbd, intrinsic, T_cam_world)

        # Store trajectory and odometry log
        self.trajectory.append({
            'frame_id': frame_id,
            'T_world_cam': self.T_world_cam.tolist(),
            'T_cam_world': T_cam_world.tolist()
        })

        self.odometry_log.append({
            'frame_id': frame_id,
            'fitness': float(fitness),
            'rmse': float(rmse),
            'translation_norm': float(np.linalg.norm(self.T_world_cam[:3, 3]))
        })

        # Update previous point cloud
        self.prev_pcd = current_pcd

        logger.debug(f"Integrated frame {frame_id}, pose translation: {self.T_world_cam[:3, 3]}")

        return {
            'fitness': fitness,
            'rmse': rmse,
            'num_points': len(current_pcd.points)
        }

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """Extract triangle mesh from TSDF volume"""
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_point_cloud(self, num_points: int = 500000) -> o3d.geometry.PointCloud:
        """Extract point cloud from TSDF volume"""
        mesh = self.extract_mesh()
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        return pcd

    def save_results(self, output_dir: str):
        """
        Save reconstruction results.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract and save mesh
        logger.info("Extracting mesh...")
        mesh = self.extract_mesh()
        mesh_path = output_path / 'fused_mesh.ply'
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        logger.info(f"Saved mesh: {mesh_path} ({len(mesh.vertices)} vertices)")

        # Extract and save point cloud
        logger.info("Extracting point cloud...")
        pcd = self.extract_point_cloud()
        pcd_path = output_path / 'fused_pointcloud.ply'
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        logger.info(f"Saved point cloud: {pcd_path} ({len(pcd.points)} points)")

        # Save trajectory
        trajectory_path = output_path / 'trajectory.json'
        with open(trajectory_path, 'w') as f:
            json.dump(self.trajectory, f, indent=2)
        logger.info(f"Saved trajectory: {trajectory_path}")

        # Save odometry log
        odometry_log_path = output_path / 'odometry_log.json'
        with open(odometry_log_path, 'w') as f:
            json.dump(self.odometry_log, f, indent=2)

        # Compute statistics
        if len(self.odometry_log) > 1:
            fitnesses = [log['fitness'] for log in self.odometry_log[1:]]  # Skip first frame
            rmses = [log['rmse'] for log in self.odometry_log[1:]]
            mean_fitness = np.mean(fitnesses) if fitnesses else 0.0
            mean_rmse = np.mean(rmses) if rmses else 0.0
            logger.info(f"Odometry statistics: Mean fitness={mean_fitness:.3f}, Mean RMSE={mean_rmse:.4f}m")

        logger.info(f"Saved odometry log: {odometry_log_path}")

        return {
            'mesh_path': str(mesh_path),
            'pcd_path': str(pcd_path),
            'trajectory_path': str(trajectory_path),
            'odometry_log_path': str(odometry_log_path),
            'num_vertices': len(mesh.vertices),
            'num_points': len(pcd.points),
            'num_frames': len(self.trajectory)
        }


def run_depth_reconstruction(
    rgb_depth_pairs: List[Tuple[str, str, str]],
    depth_K: np.ndarray,
    output_dir: str = 'output_depth_tsdf',
    tsdf_voxel_size: float = 0.01,
    tsdf_trunc_factor: float = 4.0,
    depth_unit: str = 'mm',
    use_icp: bool = True,
    icp_voxel_size: float = 0.02,
    icp_max_corr_dist: float = 0.05,
    use_undistortion: bool = False,
    depth_D: Optional[np.ndarray] = None,
    depth_width: int = 512,
    depth_height: int = 512
) -> Dict:
    """
    Run depth-only TSDF reconstruction with robust ICP odometry.

    Args:
        rgb_depth_pairs: List of (rgb_path, depth_path, pair_id)
        depth_K: Depth camera intrinsic matrix
        output_dir: Output directory
        tsdf_voxel_size: TSDF voxel size in meters
        tsdf_trunc_factor: TSDF truncation distance = voxel_size * factor
        depth_unit: Depth unit ('m', 'mm', or 'auto' for auto-detection)
        use_icp: Whether to use ICP for pose estimation
        icp_voxel_size: Voxel size for ICP downsampling
        icp_max_corr_dist: Maximum correspondence distance for ICP
        use_undistortion: Whether to undistort depth images
        depth_D: Depth distortion coefficients (required if use_undistortion=True)
        depth_width: Depth image width
        depth_height: Depth image height

    Returns:
        Dictionary with reconstruction results
    """
    if not HAS_OPEN3D:
        raise ImportError("Open3D required for depth reconstruction")

    # Auto-detect depth unit if needed
    if depth_unit == 'auto':
        if len(rgb_depth_pairs) == 0:
            raise ValueError("Cannot auto-detect depth unit: no depth images provided")

        # Load first depth image to detect unit
        first_depth_path = rgb_depth_pairs[0][1]
        first_depth = cv2.imread(first_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        from .utils import detect_depth_unit
        depth_unit = detect_depth_unit(first_depth)
        logger.info(f"Auto-detected depth unit: {depth_unit}")

    depth_scale = 1000.0 if depth_unit == 'mm' else 1.0

    # Initialize reconstructor
    reconstructor = DepthTSDFReconstructor(
        tsdf_voxel_length=tsdf_voxel_size,
        tsdf_trunc_factor=tsdf_trunc_factor,
        depth_scale=depth_scale,
        depth_trunc=10.0,
        icp_voxel_size=icp_voxel_size,
        icp_max_corr_dist=icp_max_corr_dist,
        use_undistortion=use_undistortion,
        K=depth_K if use_undistortion else None,
        D=depth_D if use_undistortion else None,
        width=depth_width,
        height=depth_height
    )

    logger.info("=" * 80)
    logger.info(f"Starting depth reconstruction with {len(rgb_depth_pairs)} frames")
    logger.info(f"TSDF voxel size: {tsdf_voxel_size*100:.1f}cm")
    logger.info(f"TSDF truncation: {tsdf_voxel_size*tsdf_trunc_factor*100:.1f}cm")
    logger.info(f"Depth unit: {depth_unit}")
    logger.info(f"ICP enabled: {use_icp}")
    if use_icp:
        logger.info(f"ICP voxel size: {icp_voxel_size*100:.1f}cm")
        logger.info(f"ICP max correspondence: {icp_max_corr_dist*100:.1f}cm")
    logger.info(f"Undistortion enabled: {use_undistortion}")
    logger.info("=" * 80)

    for idx, (rgb_path, depth_path, pair_id) in enumerate(rgb_depth_pairs):
        # Load depth image only (RGB not used in Phase 1)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            logger.warning(f"Failed to load depth image: {depth_path}")
            continue
        depth_img = depth_img.astype(np.float32)

        # Convert depth to meters
        if depth_unit == 'mm':
            depth_img = depth_img / 1000.0

        # Integrate frame with ICP odometry (depth only)
        stats = reconstructor.integrate_frame(
            depth_img, depth_K, pair_id, use_icp=use_icp
        )

        logger.info(
            f"Frame {idx+1}/{len(rgb_depth_pairs)}: {pair_id} | "
            f"Points: {stats['num_points']} | "
            f"Fitness: {stats['fitness']:.3f} | "
            f"RMSE: {stats['rmse']:.4f}m"
        )

    # Save results
    logger.info("=" * 80)
    logger.info("Saving reconstruction results...")
    results = reconstructor.save_results(output_dir)

    logger.info("=" * 80)
    logger.info("Depth reconstruction completed!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mesh: {results['num_vertices']} vertices")
    logger.info(f"Point cloud: {results['num_points']} points")
    logger.info(f"Frames integrated: {results['num_frames']}")
    logger.info("=" * 80)

    return results


if __name__ == '__main__':
    # Test and CLI
    import argparse
    import sys
    from .utils import find_rgb_depth_pairs, setup_logging
    from .calib_io import load_camera_info

    parser = argparse.ArgumentParser(description='Depth-only TSDF reconstruction with robust ICP odometry')
    parser.add_argument('--rgb-dir', required=True, help='RGB images directory')
    parser.add_argument('--depth-dir', required=True, help='Depth images directory')
    parser.add_argument('--calib', required=True, help='Depth camera calibration JSON')
    parser.add_argument('--output-dir', default='output_depth_tsdf', help='Output directory')

    # TSDF parameters
    parser.add_argument('--tsdf-voxel', type=float, default=0.01, help='TSDF voxel size in meters (default: 0.01)')
    parser.add_argument('--tsdf-trunc-factor', type=float, default=4.0, help='TSDF truncation factor (default: 4.0)')

    # Depth parameters
    parser.add_argument('--depth-unit', choices=['m', 'mm'], default='mm', help='Depth unit (default: mm)')

    # ICP parameters
    parser.add_argument('--use-icp', action='store_true', help='Enable ICP for pose estimation')
    parser.add_argument('--icp-voxel', type=float, default=0.02, help='ICP voxel size for downsampling (default: 0.02)')
    parser.add_argument('--icp-max-corr', type=float, default=0.05, help='ICP max correspondence distance (default: 0.05)')

    # Undistortion
    parser.add_argument('--undistort', action='store_true', help='Enable depth undistortion')

    # Logging
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        # Load calibration
        calib = load_camera_info(args.calib)
        logger.info(f"Loaded calibration: {calib}")

        # Find RGB-Depth pairs
        pairs = find_rgb_depth_pairs(args.rgb_dir, args.depth_dir)

        if not pairs:
            logger.error("No RGB-Depth pairs found!")
            sys.exit(1)

        logger.info(f"Found {len(pairs)} RGB-Depth pairs")

        # Run reconstruction
        results = run_depth_reconstruction(
            pairs,
            calib.K,
            output_dir=args.output_dir,
            tsdf_voxel_size=args.tsdf_voxel,
            tsdf_trunc_factor=args.tsdf_trunc_factor,
            depth_unit=args.depth_unit,
            use_icp=args.use_icp,
            icp_voxel_size=args.icp_voxel,
            icp_max_corr_dist=args.icp_max_corr,
            use_undistortion=args.undistort,
            depth_D=calib.D if args.undistort else None,
            depth_width=calib.width,
            depth_height=calib.height
        )

        print(f"\n✅ Reconstruction complete!")
        print(f"   Mesh: {results['mesh_path']} ({results['num_vertices']} vertices)")
        print(f"   Point cloud: {results['pcd_path']} ({results['num_points']} points)")
        print(f"   Trajectory: {results['trajectory_path']}")
        print(f"   Odometry log: {results['odometry_log_path']}")

    except Exception as e:
        logger.error(f"Reconstruction failed: {e}", exc_info=True)
        sys.exit(1)
