"""
Project YOLO Mask to 3D Module
Projects 2D YOLO segmentation masks to 3D using aligned depth and camera poses.
"""
import numpy as np
import cv2
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MaskProjector:
    """Projects 2D masks to 3D points in global frame"""
    
    def __init__(self, K: np.ndarray, D: np.ndarray, 
                 R: np.ndarray, t: np.ndarray,
                 class_names: List[str]):
        """
        Initialize mask projector.
        
        Args:
            K: 3x3 camera intrinsic matrix
            D: Distortion coefficients
            R: 3x3 rotation matrix (camera to world)
            t: 3x1 translation vector (camera to world)
            class_names: List of class names
        """
        self.K = K
        self.D = D
        self.R = R
        self.t = t
        self.class_names = class_names
        self.class_to_id = {name: i for i, name in enumerate(class_names)}
        
    def load_yolo_masks(self, json_path: str) -> List[Dict]:
        """
        Load YOLO mask annotations from JSON.
        
        Expected format:
        {
            "image_id": "000123",
            "masks": [
                {
                    "class": "crack",
                    "score": 0.78,
                    "polygon": [[x1,y1], [x2,y2], ...],
                    "instance_id": "i_0001"
                },
                ...
            ]
        }
        
        Args:
            json_path: Path to YOLO mask JSON file
            
        Returns:
            List of mask dictionaries
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data.get('masks', [])
    
    def rasterize_polygon(self, polygon: List[List[float]], 
                         image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Rasterize polygon to binary mask.
        
        Args:
            polygon: List of [x, y] coordinates
            image_shape: (height, width) of output mask
            
        Returns:
            Binary mask (HxW, uint8)
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert polygon to numpy array
        pts = np.array(polygon, dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(mask, [pts], 1)
        
        return mask
    
    def backproject_mask_to_3d(
        self,
        mask: np.ndarray,
        aligned_depth: np.ndarray,
        min_depth: float = 0.1,
        max_depth: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backproject mask pixels to 3D camera coordinates.
        
        Args:
            mask: Binary mask (HxW)
            aligned_depth: Aligned depth map (HxW) in meters
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth
            
        Returns:
            Tuple of (points_3d_camera, valid_mask)
            - points_3d_camera: Nx3 array in camera frame
            - valid_mask: Boolean array indicating valid points
        """
        h, w = mask.shape
        
        # Get mask pixel coordinates
        v_coords, u_coords = np.where(mask > 0)
        
        if len(u_coords) == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=bool)
        
        # Get depths at mask locations
        depths = aligned_depth[v_coords, u_coords]
        
        # Filter by valid depth range
        valid = (depths > min_depth) & (depths < max_depth)
        
        u_coords = u_coords[valid]
        v_coords = v_coords[valid]
        depths = depths[valid]
        
        if len(u_coords) == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=bool)
        
        # Backproject to 3D
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        x = (u_coords - cx) * depths / fx
        y = (v_coords - cy) * depths / fy
        z = depths
        
        points_3d_camera = np.stack([x, y, z], axis=-1)
        
        return points_3d_camera, valid
    
    def transform_to_world(self, points_camera: np.ndarray) -> np.ndarray:
        """
        Transform points from camera frame to world frame (A).
        
        Args:
            points_camera: Nx3 points in camera frame
            
        Returns:
            Nx3 points in world frame
        """
        # x_world = R * x_camera + t
        points_world = (self.R @ points_camera.T).T + self.t.T
        
        return points_world
    
    def visibility_check(
        self,
        points_world: np.ndarray,
        surface_points: Optional[np.ndarray] = None,
        distance_threshold: float = 0.05
    ) -> np.ndarray:
        """
        Check visibility of projected points against known surface.
        
        Args:
            points_world: Nx3 points to check
            surface_points: Mx3 known surface points (optional)
            distance_threshold: Maximum distance to surface (meters)
            
        Returns:
            Boolean array indicating visible points
        """
        if surface_points is None:
            # No surface to check against, all visible
            return np.ones(len(points_world), dtype=bool)
        
        # For each point, check distance to nearest surface point
        # This is a simplified version; in production, use KD-tree or TSDF
        from scipy.spatial import cKDTree
        
        tree = cKDTree(surface_points)
        distances, _ = tree.query(points_world, k=1)
        
        visible = distances < distance_threshold
        
        return visible
    
    def project_mask_to_world(
        self,
        mask_dict: Dict,
        aligned_depth: np.ndarray,
        image_shape: Tuple[int, int],
        surface_points: Optional[np.ndarray] = None,
        angle_weight: float = 1.0
    ) -> Dict:
        """
        Project a single mask to 3D world coordinates.
        
        Args:
            mask_dict: Mask dictionary with polygon, class, score, instance_id
            aligned_depth: Aligned depth map
            image_shape: (height, width)
            surface_points: Optional surface for visibility check
            angle_weight: Weight based on viewing angle (0-1)
            
        Returns:
            Dictionary with 3D projection data:
            {
                'points_3d': Nx3 array in world frame,
                'class': class name,
                'class_id': class index,
                'score': confidence score,
                'instance_id': instance identifier,
                'view_weight': combined weight for fusion
            }
        """
        # Rasterize polygon
        mask = self.rasterize_polygon(mask_dict['polygon'], image_shape)
        
        # Backproject to 3D camera frame
        points_camera, valid = self.backproject_mask_to_3d(mask, aligned_depth)
        
        if len(points_camera) == 0:
            logger.warning(f"No valid 3D points for mask {mask_dict.get('instance_id', 'unknown')}")
            return {
                'points_3d': np.zeros((0, 3)),
                'class': mask_dict['class'],
                'class_id': self.class_to_id.get(mask_dict['class'], -1),
                'score': mask_dict['score'],
                'instance_id': mask_dict.get('instance_id', 'unknown'),
                'view_weight': 0.0,
                'num_points': 0
            }
        
        # Transform to world frame
        points_world = self.transform_to_world(points_camera)
        
        # Visibility check
        if surface_points is not None:
            visible = self.visibility_check(points_world, surface_points)
            points_world = points_world[visible]
        
        # Calculate view weight
        # Combine: angle_weight * confidence * (distance_weight if needed)
        view_weight = angle_weight * mask_dict['score']
        
        return {
            'points_3d': points_world,
            'class': mask_dict['class'],
            'class_id': self.class_to_id.get(mask_dict['class'], -1),
            'score': mask_dict['score'],
            'instance_id': mask_dict.get('instance_id', 'unknown'),
            'view_weight': view_weight,
            'num_points': len(points_world)
        }
    
    def calculate_viewing_angle_weight(
        self,
        points_3d: np.ndarray,
        surface_normals: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate weight based on viewing angle.
        
        Args:
            points_3d: Nx3 points in world frame
            surface_normals: Nx3 surface normals (optional)
            
        Returns:
            Viewing angle weight (0-1)
        """
        if len(points_3d) == 0:
            return 0.0
        
        # Camera center in world frame
        camera_center = -self.R.T @ self.t
        
        # Average point
        mean_point = np.mean(points_3d, axis=0)
        
        # View direction (from camera to point)
        view_dir = mean_point - camera_center.flatten()
        view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
        
        if surface_normals is None:
            # Estimate normal from point cloud (simplified)
            # In practice, should use proper normal estimation
            return 1.0  # No penalization if normals unknown
        
        # Average normal
        mean_normal = np.mean(surface_normals, axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)
        
        # Cosine of angle between view direction and normal
        cos_angle = np.abs(np.dot(view_dir, mean_normal))
        
        return cos_angle


def project_all_masks(
    masks_json_path: str,
    aligned_depth: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    class_names: List[str],
    image_shape: Tuple[int, int],
    surface_points: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    Project all masks from a single image to 3D.
    
    Args:
        masks_json_path: Path to YOLO masks JSON
        aligned_depth: Aligned depth map
        K, D, R, t: Camera calibration and pose
        class_names: List of class names
        image_shape: (height, width)
        surface_points: Optional surface for visibility check
        
    Returns:
        List of projection dictionaries
    """
    projector = MaskProjector(K, D, R, t, class_names)
    
    # Load masks
    masks = projector.load_yolo_masks(masks_json_path)
    
    logger.info(f"Projecting {len(masks)} masks from {Path(masks_json_path).name}")
    
    # Project each mask
    projections = []
    for mask_dict in masks:
        # Calculate viewing angle weight
        angle_weight = 1.0  # Can be computed from surface normals if available
        
        proj = projector.project_mask_to_world(
            mask_dict, aligned_depth, image_shape, surface_points, angle_weight
        )
        
        if proj['num_points'] > 0:
            projections.append(proj)
            logger.debug(f"Projected {proj['num_points']} points for "
                        f"{proj['class']} (score={proj['score']:.2f})")
    
    logger.info(f"Successfully projected {len(projections)}/{len(masks)} masks")
    
    return projections


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    K = np.array([[2800, 0, 1920], [0, 2800, 1080], [0, 0, 1]], dtype=np.float64)
    D = np.zeros(8)
    R = np.eye(3)
    t = np.zeros((3, 1))
    class_names = ['crack', 'spalling', 'efflorescence', 'exposed_rebar']
    
    projector = MaskProjector(K, D, R, t, class_names)
    
    # Test polygon rasterization
    polygon = [[100, 100], [200, 100], [200, 200], [100, 200]]
    mask = projector.rasterize_polygon(polygon, (2160, 3840))
    print(f"Mask shape: {mask.shape}, Non-zero: {np.sum(mask)}")
    
    # Test backprojection
    depth = np.ones((2160, 3840)) * 2.0  # 2 meters
    points_3d, valid = projector.backproject_mask_to_3d(mask, depth)
    print(f"3D points: {len(points_3d)}, Valid: {np.sum(valid)}")
    
    if len(points_3d) > 0:
        points_world = projector.transform_to_world(points_3d)
        print(f"World points shape: {points_world.shape}")
        print(f"World points range: [{points_world.min(axis=0)}, {points_world.max(axis=0)}]")
