"""
3D Label Fusion Module
Fuses 2D labels into 3D voxel grid using probabilistic accumulation.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
from scipy.special import expit, logit

logger = logging.getLogger(__name__)


class VoxelGrid:
    """3D voxel grid for label fusion"""
    
    def __init__(
        self,
        voxel_size: float,
        bounds: Optional[np.ndarray] = None,
        num_classes: int = 5
    ):
        """
        Initialize voxel grid.
        
        Args:
            voxel_size: Voxel size in meters
            bounds: 2x3 array [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            num_classes: Number of classes (including background)
        """
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.bounds = bounds
        
        # Voxel data: key=(ix,iy,iz), value=VoxelData
        self.voxels = {}
        
        # Grid dimensions (computed after first points)
        self.origin = None
        self.grid_dims = None
        
        logger.info(f"VoxelGrid initialized: voxel_size={voxel_size}m, "
                   f"num_classes={num_classes}")
    
    def _world_to_voxel(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel indices"""
        if self.origin is None:
            # Initialize origin from bounds or first points
            if self.bounds is not None:
                self.origin = self.bounds[0]
            else:
                self.origin = np.floor(points.min(axis=0) / self.voxel_size) * self.voxel_size
            logger.debug(f"Origin set to {self.origin}")
        
        voxel_coords = np.floor((points - self.origin) / self.voxel_size).astype(np.int32)
        return voxel_coords
    
    def _voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """Convert voxel indices to world coordinates (center)"""
        return voxel_coords * self.voxel_size + self.origin + self.voxel_size / 2
    
    def get_or_create_voxel(self, voxel_idx: Tuple[int, int, int]) -> 'VoxelData':
        """Get existing voxel or create new one"""
        if voxel_idx not in self.voxels:
            self.voxels[voxel_idx] = VoxelData(self.num_classes)
        return self.voxels[voxel_idx]
    
    def accumulate(
        self,
        points_3d: np.ndarray,
        class_id: int,
        score: float,
        view_weight: float = 1.0,
        angle_weight: float = 1.0,
        distance_weight: float = 1.0
    ):
        """
        Accumulate points into voxel grid with probabilistic fusion.
        
        Args:
            points_3d: Nx3 points in world frame
            class_id: Class index
            score: Detection confidence
            view_weight: Weight for this view
            angle_weight: Viewing angle weight (cosine)
            distance_weight: Distance-based weight
        """
        if len(points_3d) == 0:
            return
        
        # Convert to voxel coordinates
        voxel_coords = self._world_to_voxel(points_3d)
        
        # Compute combined weight
        combined_weight = view_weight * angle_weight * distance_weight
        
        # Accumulate into voxels
        unique_voxels = np.unique(voxel_coords, axis=0)
        
        for voxel_coord in unique_voxels:
            voxel_idx = tuple(voxel_coord)
            voxel = self.get_or_create_voxel(voxel_idx)
            
            # Count points in this voxel
            mask = np.all(voxel_coords == voxel_coord, axis=1)
            num_points = np.sum(mask)
            
            # Update voxel with log-odds
            voxel.update_logodds(class_id, score, combined_weight, num_points)
    
    def finalize(self, prob_thresh: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Finalize fusion and extract labeled voxels.
        
        Args:
            prob_thresh: Minimum probability threshold for labeling
            
        Returns:
            Tuple of (voxel_centers, labels, probabilities)
            - voxel_centers: Nx3 world coordinates
            - labels: N class labels
            - probabilities: NxC probability distribution
        """
        if len(self.voxels) == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int32), np.zeros((0, self.num_classes))
        
        logger.info(f"Finalizing {len(self.voxels)} voxels...")
        
        voxel_centers = []
        labels = []
        probabilities = []
        
        for voxel_idx, voxel_data in self.voxels.items():
            # Convert log-odds to probabilities
            probs = voxel_data.get_probabilities()
            
            # Get best class
            best_class = np.argmax(probs)
            best_prob = probs[best_class]
            
            # Apply threshold
            if best_prob >= prob_thresh:
                # Convert voxel index to world coordinates
                voxel_coord = np.array(voxel_idx)
                center = self._voxel_to_world(voxel_coord[np.newaxis, :])[0]
                
                voxel_centers.append(center)
                labels.append(best_class)
                probabilities.append(probs)
        
        voxel_centers = np.array(voxel_centers)
        labels = np.array(labels, dtype=np.int32)
        probabilities = np.array(probabilities)
        
        logger.info(f"Extracted {len(labels)} labeled voxels (thresh={prob_thresh})")
        
        # Class distribution
        for cls in range(self.num_classes):
            count = np.sum(labels == cls)
            if count > 0:
                logger.info(f"  Class {cls}: {count} voxels")
        
        return voxel_centers, labels, probabilities
    
    def get_voxel_data(self) -> Dict:
        """Get raw voxel data for debugging"""
        return {
            'voxels': self.voxels,
            'origin': self.origin,
            'voxel_size': self.voxel_size,
            'num_voxels': len(self.voxels)
        }


class VoxelData:
    """Data stored in each voxel"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        # Log-odds for each class
        self.logodds = np.zeros(num_classes, dtype=np.float32)
        # View count
        self.view_count = 0
        # Total weight accumulated
        self.total_weight = 0.0
        # Point count
        self.point_count = 0
    
    def update_logodds(
        self,
        class_id: int,
        score: float,
        weight: float,
        num_points: int
    ):
        """
        Update log-odds for a class using Bayesian fusion.
        
        Log-odds update:
        L_new = L_old + weight * (logit(score) - logit(0.5))
        
        This assumes a uniform prior (0.5) and updates based on new evidence.
        """
        if class_id < 0 or class_id >= self.num_classes:
            return
        
        # Convert score to log-odds
        # Clip score to avoid numerical issues
        score = np.clip(score, 0.01, 0.99)
        
        # Log-odds relative to uniform prior
        prior_logodds = 0.0  # logit(0.5) = 0
        evidence_logodds = logit(score)
        delta_logodds = evidence_logodds - prior_logodds
        
        # Update with weight
        self.logodds[class_id] += weight * delta_logodds
        
        # Update metadata
        self.view_count += 1
        self.total_weight += weight
        self.point_count += num_points
    
    def get_probabilities(self) -> np.ndarray:
        """Convert log-odds to probabilities using softmax"""
        # Apply sigmoid to get probabilities
        probs = expit(self.logodds)
        
        # Normalize to sum to 1 (softmax-like)
        probs = probs / (np.sum(probs) + 1e-8)
        
        return probs
    
    def get_entropy(self) -> float:
        """Calculate entropy of probability distribution"""
        probs = self.get_probabilities()
        # Avoid log(0)
        probs = np.clip(probs, 1e-8, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return entropy


class LabelFusion:
    """High-level interface for 3D label fusion"""
    
    def __init__(
        self,
        voxel_size: float,
        num_classes: int,
        class_names: List[str],
        config: Optional[Dict] = None
    ):
        """
        Initialize label fusion.
        
        Args:
            voxel_size: Voxel size in meters
            num_classes: Number of classes
            class_names: List of class names
            config: Configuration dictionary
        """
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.class_names = class_names
        self.config = config or {}
        
        # Create voxel grid
        self.voxel_grid = VoxelGrid(voxel_size, num_classes=num_classes)
        
        # Fusion statistics
        self.stats = {
            'num_views': 0,
            'num_masks': 0,
            'total_points': 0,
            'class_points': defaultdict(int)
        }
        
        logger.info(f"LabelFusion initialized: voxel_size={voxel_size}m")
    
    def fuse_projection(self, projection: Dict):
        """
        Fuse a single mask projection into the voxel grid.
        
        Args:
            projection: Dictionary from project_mask_to_A
        """
        points_3d = projection['points_3d']
        class_id = projection['class_id']
        score = projection['score']
        view_weight = projection['view_weight']
        
        if len(points_3d) == 0:
            return
        
        # Accumulate into voxel grid
        self.voxel_grid.accumulate(
            points_3d,
            class_id,
            score,
            view_weight=view_weight
        )
        
        # Update statistics
        self.stats['num_masks'] += 1
        self.stats['total_points'] += len(points_3d)
        self.stats['class_points'][class_id] += len(points_3d)
    
    def fuse_image(self, projections: List[Dict]):
        """
        Fuse all mask projections from a single image.
        
        Args:
            projections: List of projection dictionaries
        """
        logger.info(f"Fusing {len(projections)} masks from image")
        
        for proj in projections:
            self.fuse_projection(proj)
        
        self.stats['num_views'] += 1
    
    def finalize(self, prob_thresh: float = 0.55) -> Dict:
        """
        Finalize fusion and return results.
        
        Args:
            prob_thresh: Minimum probability threshold
            
        Returns:
            Dictionary with:
            - voxel_centers: Nx3 coordinates
            - labels: N class labels
            - probabilities: NxC probability distribution
            - stats: Fusion statistics
        """
        logger.info("Finalizing label fusion...")
        
        voxel_centers, labels, probabilities = self.voxel_grid.finalize(prob_thresh)
        
        # Compute quality metrics
        labeled_voxels = len(labels)
        
        # Entropy statistics
        entropies = []
        for voxel_data in self.voxel_grid.voxels.values():
            entropies.append(voxel_data.get_entropy())
        
        mean_entropy = np.mean(entropies) if entropies else 0
        
        # Class conflict rate (simplified: voxels with high entropy)
        high_entropy_thresh = np.log(self.num_classes) * 0.7  # 70% of max entropy
        conflict_rate = np.sum(np.array(entropies) > high_entropy_thresh) / max(len(entropies), 1)
        
        self.stats['labeled_voxels'] = labeled_voxels
        self.stats['mean_entropy'] = mean_entropy
        self.stats['conflict_rate'] = conflict_rate
        
        logger.info(f"Fusion complete: {labeled_voxels} voxels labeled")
        logger.info(f"  Views: {self.stats['num_views']}, Masks: {self.stats['num_masks']}")
        logger.info(f"  Mean entropy: {mean_entropy:.3f}, Conflict rate: {conflict_rate:.1%}")
        
        return {
            'voxel_centers': voxel_centers,
            'labels': labels,
            'probabilities': probabilities,
            'stats': self.stats,
            'voxel_grid': self.voxel_grid
        }


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Create fusion
    fusion = LabelFusion(
        voxel_size=0.01,  # 1cm
        num_classes=5,
        class_names=['crack', 'spalling', 'efflorescence', 'exposed_rebar', 'background']
    )
    
    # Simulate projections
    for view_idx in range(3):
        # Random points for class 0 (crack)
        points = np.random.randn(100, 3) * 0.1 + [view_idx * 0.5, 0, 2.0]
        
        projection = {
            'points_3d': points,
            'class': 'crack',
            'class_id': 0,
            'score': 0.8,
            'view_weight': 1.0,
            'num_points': len(points)
        }
        
        fusion.fuse_projection(projection)
    
    # Finalize
    result = fusion.finalize(prob_thresh=0.5)
    
    print(f"\nResults:")
    print(f"  Labeled voxels: {len(result['labels'])}")
    print(f"  Label distribution: {np.bincount(result['labels'])}")
    print(f"  Mean entropy: {result['stats']['mean_entropy']:.3f}")
    print(f"  Conflict rate: {result['stats']['conflict_rate']:.1%}")
