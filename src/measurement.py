"""
Measurement Module
Computes geometric measurements for 3D instances (length, area, orientation).
Includes RGB pixel-level width refinement based on academic paper methodology.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize_3d, skeletonize
from sklearn.decomposition import PCA
import logging
import cv2
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class InstanceMeasurement:
    """Geometric measurements for 3D instances"""

    def __init__(self, config: Optional[Dict] = None, data_paths: Optional[Dict] = None):
        """
        Initialize measurement module.

        Args:
            config: Configuration dictionary
            data_paths: Dictionary with paths to RGB, depth, masks, etc.
        """
        self.config = config or {}
        self.crack_smooth = self.config.get('crack_skeleton_smooth', True)
        self.min_crack_length = self.config.get('crack_min_length_cm', 5.0) / 100  # to meters
        self.min_area = self.config.get('area_min_m2', 0.001)

        # Data paths for RGB refinement
        self.data_paths = data_paths or {}
        self.rgb_dir = Path(self.data_paths.get('rgb_dir', '')) if self.data_paths.get('rgb_dir') else None
        self.aligned_depth_dir = Path(self.data_paths.get('aligned_depth_dir', '')) if self.data_paths.get('aligned_depth_dir') else None
        self.masks_dir = Path(self.data_paths.get('masks_dir', '')) if self.data_paths.get('masks_dir') else None
        self.camera_K = self.data_paths.get('camera_K')  # 3x3 intrinsics matrix

        # Check if RGB refinement is available
        self.rgb_refinement_available = (
            self.rgb_dir is not None and self.rgb_dir.exists() and
            self.aligned_depth_dir is not None and self.aligned_depth_dir.exists() and
            self.masks_dir is not None and self.masks_dir.exists() and
            self.camera_K is not None
        )

        if self.rgb_refinement_available:
            logger.info("RGB pixel-level width refinement enabled")
        else:
            logger.info("RGB refinement disabled (missing data paths or camera intrinsics)")
    
    def measure_instance(
        self,
        instance: 'Instance3D',
        voxel_size: float
    ) -> Dict:
        """
        Compute measurements for an instance based on its class.
        
        Args:
            instance: Instance3D object
            voxel_size: Voxel size in meters
            
        Returns:
            Dictionary with measurements
        """
        measurements = {
            'instance_id': instance.instance_id,
            'class_id': instance.class_id,
            'class_name': instance.class_name,
            'num_voxels': instance.num_voxels,
            'volume_m3': instance.num_voxels * (voxel_size ** 3),
            'mean_confidence': instance.mean_confidence,
            'view_count': instance.view_count,
            'entropy': instance.entropy
        }
        
        # Add bounding box
        if instance.bbox_min is not None:
            measurements['bbox_min'] = instance.bbox_min.tolist()
            measurements['bbox_max'] = instance.bbox_max.tolist()
            measurements['bbox_size'] = (instance.bbox_max - instance.bbox_min).tolist()
            measurements['centroid'] = instance.centroid.tolist()
        
        # Class-specific measurements
        class_name = instance.class_name.lower()
        
        if 'crack' in class_name:
            # Measure crack: length, width, orientation
            crack_measures = self.measure_crack(instance, voxel_size)
            measurements.update(crack_measures)
        
        elif 'spall' in class_name or 'efflor' in class_name or 'rebar' in class_name:
            # Measure area-based defects: area, depth
            area_measures = self.measure_area_defect(instance, voxel_size)
            measurements.update(area_measures)
        
        return measurements
    
    def measure_crack(
        self,
        instance: 'Instance3D',
        voxel_size: float
    ) -> Dict:
        """
        Measure crack: skeletonization, length, orientation.
        Uses hybrid approach: 3D voxel for length, RGB pixel-level for width.

        Args:
            instance: Crack instance
            voxel_size: Voxel size in meters

        Returns:
            Dictionary with crack measurements
        """
        points = instance.voxel_centers

        # Voxelize points for skeletonization
        voxel_grid, grid_origin = self._voxelize_points(points, voxel_size)

        # Skeletonize
        skeleton = skeletonize_3d(voxel_grid).astype(bool)
        skeleton_points = np.argwhere(skeleton) * voxel_size + grid_origin

        if len(skeleton_points) < 2:
            logger.warning(f"Crack {instance.instance_id}: Insufficient skeleton points")
            return {
                'length_m': 0.0,
                'num_branches': 0,
                'orientation': None,
                'endpoints': []
            }

        # Compute skeleton length (sum of edge lengths) - 3D MST method
        length = self._compute_skeleton_length(skeleton_points, voxel_size)

        # Find endpoints and branches
        endpoints, num_branches = self._analyze_skeleton_topology(skeleton)

        # Compute principal orientation
        pca = PCA(n_components=1)
        pca.fit(skeleton_points)
        orientation = pca.components_[0]

        # 3D voxel-based width (baseline method)
        width_3d_m = self._estimate_crack_width(points, skeleton_points, voxel_size)

        measurements = {
            'length_m': float(length),
            'width_3d_m': float(width_3d_m),  # 3D voxel-based width
            'num_branches': num_branches,
            'num_endpoints': len(endpoints),
            'orientation': orientation.tolist(),
            'skeleton_points': len(skeleton_points)
        }

        # RGB pixel-level width refinement (paper methodology)
        if self.rgb_refinement_available:
            width_rgb_mm = self._refine_crack_width_from_rgb(instance, voxel_size)

            if width_rgb_mm is not None:
                measurements['width_rgb_mm'] = float(width_rgb_mm)
                measurements['width_rgb_m'] = float(width_rgb_mm / 1000.0)

                # Use RGB width as primary if available
                measurements['width_m'] = measurements['width_rgb_m']

                logger.info(f"  {instance.instance_id}: RGB width = {width_rgb_mm:.2f}mm "
                           f"(3D baseline = {width_3d_m*1000:.2f}mm)")
            else:
                # Fallback to 3D width
                measurements['width_m'] = measurements['width_3d_m']
                logger.debug(f"  {instance.instance_id}: Using 3D width (RGB refinement failed)")
        else:
            # No RGB refinement available
            measurements['width_m'] = measurements['width_3d_m']

        return measurements
    
    def measure_area_defect(
        self,
        instance: 'Instance3D',
        voxel_size: float
    ) -> Dict:
        """
        Measure area-based defect: surface area, depth.
        
        Args:
            instance: Area defect instance
            voxel_size: Voxel size in meters
            
        Returns:
            Dictionary with area measurements
        """
        points = instance.voxel_centers
        
        # Estimate surface area (simplified: project to dominant plane)
        area_m2 = self._estimate_surface_area(points, voxel_size)
        
        # Estimate depth/thickness
        depth_m = self._estimate_depth(points)
        
        # Fit plane and compute normal
        plane_normal, plane_offset = self._fit_plane(points)
        
        return {
            'area_m2': float(area_m2),
            'depth_m': float(depth_m),
            'plane_normal': plane_normal.tolist() if plane_normal is not None else None,
            'plane_offset': float(plane_offset) if plane_offset is not None else None
        }
    
    def _voxelize_points(
        self,
        points: np.ndarray,
        voxel_size: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert point cloud to 3D binary voxel grid.
        
        Returns:
            Tuple of (voxel_grid, grid_origin)
        """
        # Compute grid bounds
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        
        # Grid dimensions
        grid_size = np.ceil((max_bound - min_bound) / voxel_size).astype(np.int32) + 1
        
        # Create grid
        voxel_grid = np.zeros(grid_size, dtype=bool)
        
        # Fill grid
        indices = np.floor((points - min_bound) / voxel_size).astype(np.int32)
        indices = np.clip(indices, 0, grid_size - 1)
        
        voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        
        return voxel_grid, min_bound
    
    def _compute_skeleton_length(
        self,
        skeleton_points: np.ndarray,
        voxel_size: float
    ) -> float:
        """
        Compute total skeleton length by connecting nearby points.
        
        Simplified: uses MST (minimum spanning tree) approximation.
        """
        if len(skeleton_points) < 2:
            return 0.0
        
        from scipy.spatial.distance import pdist, squareform
        from scipy.sparse.csgraph import minimum_spanning_tree
        
        # Compute pairwise distances
        dist_matrix = squareform(pdist(skeleton_points))
        
        # Threshold: only connect nearby points (e.g., < 3*voxel_size)
        threshold = 3 * voxel_size
        dist_matrix[dist_matrix > threshold] = 0
        
        # Minimum spanning tree
        mst = minimum_spanning_tree(dist_matrix)
        
        # Sum of edge weights = total length
        total_length = mst.sum()
        
        return total_length
    
    def _analyze_skeleton_topology(
        self,
        skeleton: np.ndarray
    ) -> Tuple[List, int]:
        """
        Analyze skeleton topology: find endpoints and branches.
        
        Returns:
            Tuple of (endpoints, num_branches)
        """
        from scipy import ndimage
        
        # Count neighbors for each skeleton voxel
        struct = ndimage.generate_binary_structure(3, 3)
        skeleton_bool = skeleton.astype(bool)
        neighbor_count = ndimage.convolve(skeleton_bool.astype(np.int32), struct.astype(np.int32), mode='constant')
        neighbor_count[~skeleton_bool] = 0
        neighbor_count -= 1  # Subtract self
        
        # Endpoints: 1 neighbor
        endpoints = np.argwhere((neighbor_count == 1) & skeleton)
        
        # Branches: 3+ neighbors
        num_branches = np.sum((neighbor_count >= 3) & skeleton)
        
        return endpoints.tolist(), int(num_branches)
    
    def _estimate_crack_width(
        self,
        points: np.ndarray,
        skeleton_points: np.ndarray,
        voxel_size: float
    ) -> float:
        """
        Estimate average crack width as distance from points to skeleton.
        """
        from scipy.spatial import cKDTree
        
        if len(skeleton_points) < 2:
            return voxel_size  # Default to voxel size
        
        tree = cKDTree(skeleton_points)
        distances, _ = tree.query(points, k=1)
        
        # Average distance * 2 = width
        avg_width = np.mean(distances) * 2
        
        return avg_width
    
    def _estimate_surface_area(
        self,
        points: np.ndarray,
        voxel_size: float
    ) -> float:
        """
        Estimate surface area by projecting to dominant plane.
        """
        # Fit plane
        plane_normal, _ = self._fit_plane(points)
        
        if plane_normal is None:
            # Fallback: use XY projection
            plane_normal = np.array([0, 0, 1])
        
        # Project points to plane
        # Create 2D coordinate system on plane
        u = self._get_perpendicular_vector(plane_normal)
        v = np.cross(plane_normal, u)
        
        # Project
        points_2d = np.column_stack([
            np.dot(points, u),
            np.dot(points, v)
        ])
        
        # Estimate area using convex hull or voxel counting
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(points_2d)
            area = hull.volume  # In 2D, volume = area
        except:
            # Fallback: count unique 2D voxels
            voxels_2d = np.unique(np.floor(points_2d / voxel_size).astype(np.int32), axis=0)
            area = len(voxels_2d) * (voxel_size ** 2)
        
        return area
    
    def _estimate_depth(self, points: np.ndarray) -> float:
        """
        Estimate depth/thickness of defect.
        """
        # Fit plane
        plane_normal, plane_offset = self._fit_plane(points)
        
        if plane_normal is None:
            return 0.0
        
        # Compute distances to plane
        distances = np.abs(np.dot(points, plane_normal) - plane_offset)
        
        # Depth = max distance
        depth = np.max(distances)
        
        return depth
    
    def _fit_plane(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Fit plane to points using PCA.
        
        Returns:
            Tuple of (normal, offset)
        """
        if len(points) < 3:
            return None, None
        
        # Center points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # PCA
        try:
            pca = PCA(n_components=3)
            pca.fit(centered)
            
            # Normal is the component with smallest variance
            normal = pca.components_[-1]
            
            # Offset
            offset = np.dot(centroid, normal)
            
            return normal, offset
        except:
            return None, None
    
    def _get_perpendicular_vector(self, v: np.ndarray) -> np.ndarray:
        """Get a vector perpendicular to v"""
        # Choose axis that is not parallel to v
        if abs(v[0]) < 0.9:
            axis = np.array([1, 0, 0])
        else:
            axis = np.array([0, 1, 0])

        # Cross product
        perp = np.cross(v, axis)
        perp = perp / (np.linalg.norm(perp) + 1e-8)

        return perp

    # ========================================================================
    # RGB Pixel-Level Width Refinement (Paper-based methodology)
    # ========================================================================

    def _preprocess_mask_for_width(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocess mask for precise width measurement using paper methodology.

        Pipeline:
        1. Median blur (noise reduction)
        2. Bilateral filter (edge-preserving smoothing)
        3. Gamma correction (enhance dark regions)
        4. RCLAHE (local contrast enhancement)
        5. Adaptive threshold (binarization)

        Args:
            mask: Binary mask (0-255)

        Returns:
            Preprocessed binary mask
        """
        # Ensure 8-bit format
        mask = mask.astype(np.uint8)

        # 1. Median blur for noise reduction
        mask = cv2.medianBlur(mask, ksize=3)

        # 2. Bilateral filter (edge-preserving smoothing)
        mask = cv2.bilateralFilter(mask, d=5, sigmaColor=75, sigmaSpace=75)

        # 3. Gamma correction (enhance dark regions, gamma < 1 brightens)
        gamma = 0.7
        mask = np.power(mask / 255.0, gamma) * 255
        mask = mask.astype(np.uint8)

        # 4. RCLAHE (Recurrent CLAHE for local contrast)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        mask = clahe.apply(mask)

        # 5. Adaptive threshold for final binarization
        mask = cv2.adaptiveThreshold(
            mask, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        return mask

    def _estimate_local_tangent(
        self,
        skeleton: np.ndarray,
        y: int,
        x: int,
        window: int = 5
    ) -> np.ndarray:
        """
        Estimate local tangent direction at a skeleton point.

        Args:
            skeleton: 2D skeleton (binary)
            y, x: Point coordinates
            window: Window size for tangent estimation

        Returns:
            Unit tangent vector [dy, dx]
        """
        # Extract local window
        h, w = skeleton.shape
        y_min = max(0, y - window)
        y_max = min(h, y + window + 1)
        x_min = max(0, x - window)
        x_max = min(w, x + window + 1)

        local_skel = skeleton[y_min:y_max, x_min:x_max]
        local_pts = np.argwhere(local_skel)

        if len(local_pts) < 2:
            # Fallback: horizontal tangent
            return np.array([0.0, 1.0])

        # Offset to global coordinates
        local_pts = local_pts + np.array([y_min, x_min])

        # PCA to find principal direction
        try:
            pca = PCA(n_components=1)
            pca.fit(local_pts)
            tangent = pca.components_[0]  # [dy, dx]

            # Normalize
            norm = np.linalg.norm(tangent)
            if norm > 1e-8:
                tangent = tangent / norm
            else:
                tangent = np.array([0.0, 1.0])

            return tangent
        except:
            return np.array([0.0, 1.0])

    def _scan_perpendicular_width(
        self,
        mask: np.ndarray,
        y: int,
        x: int,
        normal: np.ndarray,
        max_dist: int = 50
    ) -> float:
        """
        Scan along perpendicular direction to measure crack width.

        Args:
            mask: Binary mask (0 or 255)
            y, x: Skeleton point
            normal: Normal direction [dy, dx]
            max_dist: Maximum scan distance in pixels

        Returns:
            Width in pixels
        """
        h, w = mask.shape

        # Scan in both directions along normal
        def scan_direction(direction):
            for dist in range(1, max_dist + 1):
                py = int(y + direction * normal[0] * dist)
                px = int(x + direction * normal[1] * dist)

                if py < 0 or py >= h or px < 0 or px >= w:
                    return dist - 1

                if mask[py, px] == 0:  # Hit background
                    return dist - 1

            return max_dist

        # Scan positive and negative directions
        dist_pos = scan_direction(1)
        dist_neg = scan_direction(-1)

        # Total width
        width_px = dist_pos + dist_neg

        return width_px

    def _refine_crack_width_from_rgb(
        self,
        instance: 'Instance3D',
        voxel_size: float
    ) -> Optional[float]:
        """
        Refine crack width using RGB pixel-level analysis (multi-view).

        For each associated RGB image:
        1. Load RGB image, mask, and aligned depth
        2. Preprocess mask
        3. 2D skeletonization
        4. Perpendicular width scanning
        5. Convert pixels to mm using depth

        Multi-view aggregation: median of all measurements.

        Args:
            instance: Crack instance
            voxel_size: Voxel size in meters

        Returns:
            Median width in mm, or None if insufficient data
        """
        if not self.rgb_refinement_available:
            return None

        # Find associated RGB images by scanning all masks
        # Check which masks overlap with instance bbox
        associated_images = self._find_associated_images(instance)

        if len(associated_images) == 0:
            logger.debug(f"No associated images found for {instance.instance_id}")
            return None

        logger.debug(f"Found {len(associated_images)} associated images for {instance.instance_id}")

        widths_mm = []

        for image_id in associated_images:
            try:
                width_mm = self._measure_width_in_image(instance, image_id, voxel_size)
                if width_mm is not None and width_mm > 0:
                    widths_mm.append(width_mm)
            except Exception as e:
                logger.warning(f"Failed to measure width in {image_id}: {e}")
                continue

        if len(widths_mm) == 0:
            return None

        # Multi-view aggregation: median
        median_width_mm = float(np.median(widths_mm))

        logger.debug(f"{instance.instance_id}: RGB width = {median_width_mm:.2f}mm "
                    f"(from {len(widths_mm)} views, range: {min(widths_mm):.2f}-{max(widths_mm):.2f}mm)")

        return median_width_mm

    def _find_associated_images(self, instance: 'Instance3D') -> List[str]:
        """
        Find RGB images associated with this instance.

        Strategy: Check all mask JSON files for masks overlapping with instance bbox.

        Args:
            instance: Instance to find images for

        Returns:
            List of image IDs
        """
        associated = []

        # Get instance 3D bbox
        bbox_min = instance.bbox_min
        bbox_max = instance.bbox_max

        # Scan all mask JSON files
        for mask_path in self.masks_dir.glob('*.json'):
            image_id = mask_path.stem

            try:
                with open(mask_path, 'r') as f:
                    mask_data = json.load(f)

                masks = mask_data.get('masks', [])

                # Check if any mask matches instance class
                for mask_info in masks:
                    if mask_info['class'].lower() == instance.class_name.lower():
                        # Found a crack mask in this image
                        associated.append(image_id)
                        break

            except Exception as e:
                logger.warning(f"Failed to read {mask_path}: {e}")
                continue

        return associated

    def _measure_width_in_image(
        self,
        instance: 'Instance3D',
        image_id: str,
        voxel_size: float
    ) -> Optional[float]:
        """
        Measure crack width in a single RGB image.

        Args:
            instance: Crack instance
            image_id: RGB image ID
            voxel_size: Voxel size

        Returns:
            Average width in mm, or None
        """
        # Load RGB image
        rgb_path = self.rgb_dir / f"{image_id}.png"
        if not rgb_path.exists():
            return None

        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            return None

        h, w = rgb_img.shape[:2]

        # Load mask JSON
        mask_path = self.masks_dir / f"{image_id}.json"
        if not mask_path.exists():
            return None

        with open(mask_path, 'r') as f:
            mask_data = json.load(f)

        # Find crack masks matching instance class
        crack_masks = [
            m for m in mask_data.get('masks', [])
            if m['class'].lower() == instance.class_name.lower()
        ]

        if len(crack_masks) == 0:
            return None

        # Load aligned depth (for pixel-to-mm conversion)
        # Convert RGB image_id to DPT image_id if needed
        depth_image_id = image_id.replace("camera_RGB_", "camera_DPT_")
        depth_path = self.aligned_depth_dir / f"{depth_image_id}.png"

        if not depth_path.exists():
            return None

        aligned_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # to meters

        # Process each crack mask
        widths_px = []

        for crack_mask in crack_masks:
            polygon = crack_mask['polygon']

            # Create binary mask from polygon
            mask_binary = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask_binary, [pts], 255)

            # Preprocess mask
            mask_processed = self._preprocess_mask_for_width(mask_binary)

            # 2D skeletonization
            skeleton_2d = skeletonize(mask_processed > 0).astype(np.uint8) * 255

            skeleton_pts = np.argwhere(skeleton_2d > 0)

            if len(skeleton_pts) < 10:
                continue

            # Sample skeleton points (every 2 pixels to reduce computation)
            stride = 2
            sampled_pts = skeleton_pts[::stride]

            # Perpendicular width scan for each skeleton point
            for (y, x) in sampled_pts:
                # Estimate local tangent
                tangent = self._estimate_local_tangent(skeleton_2d, y, x, window=5)

                # Normal = perpendicular to tangent (90° rotation)
                normal = np.array([-tangent[1], tangent[0]])

                # Scan width
                width_px = self._scan_perpendicular_width(mask_processed, y, x, normal, max_dist=50)

                if width_px > 0:
                    widths_px.append((y, x, width_px))

        if len(widths_px) == 0:
            return None

        # Convert pixels to mm using depth and camera K
        widths_mm = []

        fx = self.camera_K[0, 0]

        for (y, x, width_px) in widths_px:
            # Get depth at this pixel
            depth_m = aligned_depth[y, x]

            if depth_m > 0.1 and depth_m < 10.0:  # Valid depth range
                # Compute mm per pixel at this depth
                mm_per_px = (depth_m / fx) * 1000.0

                # Convert width
                width_mm = width_px * mm_per_px

                widths_mm.append(width_mm)

        if len(widths_mm) == 0:
            return None

        # Average width in this image
        avg_width_mm = float(np.mean(widths_mm))

        return avg_width_mm


def measure_all_instances(
    instances: List['Instance3D'],
    voxel_size: float,
    config: Optional[Dict] = None,
    data_paths: Optional[Dict] = None
) -> List[Dict]:
    """
    Measure all instances.

    Args:
        instances: List of Instance3D objects
        voxel_size: Voxel size in meters
        config: Configuration dictionary
        data_paths: Dictionary with paths to RGB, depth, masks, and camera K

    Returns:
        List of measurement dictionaries
    """
    measurer = InstanceMeasurement(config, data_paths)

    logger.info(f"Measuring {len(instances)} instances...")

    measurements = []
    for i, instance in enumerate(instances):
        logger.debug(f"Measuring instance {i+1}/{len(instances)}: {instance.instance_id}")

        measure = measurer.measure_instance(instance, voxel_size)
        measurements.append(measure)

        # Log key measurements
        if 'length_m' in measure:
            width_info = ""
            if 'width_rgb_mm' in measure:
                width_info = f"width(RGB)={measure['width_rgb_mm']:.2f}mm"
            else:
                width_info = f"width(3D)={measure.get('width_m', 0)*1000:.2f}mm"

            logger.info(f"  {instance.instance_id}: length={measure['length_m']:.3f}m, {width_info}")
        elif 'area_m2' in measure:
            logger.info(f"  {instance.instance_id}: area={measure['area_m2']:.4f}m², "
                       f"depth={measure.get('depth_m', 0):.4f}m")

    logger.info("Measurement complete")

    return measurements


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    
    from instance_merge import Instance3D
    
    # Create synthetic crack instance
    np.random.seed(42)
    
    # Linear crack
    t = np.linspace(0, 1, 100)
    crack_points = np.column_stack([
        t * 0.5,
        t * 0.1 + np.random.randn(100) * 0.005,  # Slight noise
        np.ones(100) * 2.0 + np.random.randn(100) * 0.005
    ])
    
    crack_probs = np.zeros((100, 5))
    crack_probs[:, 0] = 0.8
    
    crack_instance = Instance3D(
        instance_id='test_crack',
        class_id=0,
        class_name='crack',
        voxel_centers=crack_points,
        probabilities=crack_probs
    )
    
    # Measure
    measurer = InstanceMeasurement()
    measurements = measurer.measure_instance(crack_instance, voxel_size=0.005)
    
    print("\nCrack Measurements:")
    for key, value in measurements.items():
        if not isinstance(value, (list, np.ndarray)):
            print(f"  {key}: {value}")
