"""
Measurement Module
Computes geometric measurements for 3D instances (length, area, orientation).
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize_3d
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class InstanceMeasurement:
    """Geometric measurements for 3D instances"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize measurement module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.crack_smooth = self.config.get('crack_skeleton_smooth', True)
        self.min_crack_length = self.config.get('crack_min_length_cm', 5.0) / 100  # to meters
        self.min_area = self.config.get('area_min_m2', 0.001)
    
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
        
        # Compute skeleton length (sum of edge lengths)
        length = self._compute_skeleton_length(skeleton_points, voxel_size)
        
        # Find endpoints and branches
        endpoints, num_branches = self._analyze_skeleton_topology(skeleton)
        
        # Compute principal orientation
        pca = PCA(n_components=1)
        pca.fit(skeleton_points)
        orientation = pca.components_[0]
        
        # Average width (distance from skeleton to boundary)
        avg_width = self._estimate_crack_width(points, skeleton_points, voxel_size)
        
        return {
            'length_m': float(length),
            'width_m': float(avg_width),
            'num_branches': num_branches,
            'num_endpoints': len(endpoints),
            'orientation': orientation.tolist(),
            'skeleton_points': len(skeleton_points)
        }
    
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


def measure_all_instances(
    instances: List['Instance3D'],
    voxel_size: float,
    config: Optional[Dict] = None
) -> List[Dict]:
    """
    Measure all instances.
    
    Args:
        instances: List of Instance3D objects
        voxel_size: Voxel size in meters
        config: Configuration dictionary
        
    Returns:
        List of measurement dictionaries
    """
    measurer = InstanceMeasurement(config)
    
    logger.info(f"Measuring {len(instances)} instances...")
    
    measurements = []
    for i, instance in enumerate(instances):
        logger.debug(f"Measuring instance {i+1}/{len(instances)}: {instance.instance_id}")
        
        measure = measurer.measure_instance(instance, voxel_size)
        measurements.append(measure)
        
        # Log key measurements
        if 'length_m' in measure:
            logger.info(f"  {instance.instance_id}: length={measure['length_m']:.3f}m, "
                       f"width={measure.get('width_m', 0):.4f}m")
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
