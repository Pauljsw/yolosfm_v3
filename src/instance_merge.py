"""
Instance Merge Module
Merges fragmented instances using DBSCAN clustering and IoU-based merging.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class Instance3D:
    """3D instance representation"""
    
    def __init__(
        self,
        instance_id: str,
        class_id: int,
        class_name: str,
        voxel_centers: np.ndarray,
        probabilities: np.ndarray
    ):
        """
        Initialize 3D instance.
        
        Args:
            instance_id: Unique instance identifier
            class_id: Class index
            class_name: Class name
            voxel_centers: Nx3 voxel coordinates
            probabilities: NxC probability distribution per voxel
        """
        self.instance_id = instance_id
        self.class_id = class_id
        self.class_name = class_name
        self.voxel_centers = voxel_centers
        self.probabilities = probabilities
        
        # Derived properties
        self.num_voxels = len(voxel_centers)
        self.mean_confidence = np.mean(probabilities[:, class_id]) if len(probabilities) > 0 else 0
        self.view_count = 1  # Will be updated during merge
        self.entropy = self._calculate_entropy()
        
        # Geometric properties
        self.bbox_min = np.min(voxel_centers, axis=0) if len(voxel_centers) > 0 else None
        self.bbox_max = np.max(voxel_centers, axis=0) if len(voxel_centers) > 0 else None
        self.centroid = np.mean(voxel_centers, axis=0) if len(voxel_centers) > 0 else None
        
        # Metadata
        self.source_views = []
        self.merged_from = []
    
    def _calculate_entropy(self) -> float:
        """Calculate average entropy of voxels"""
        if len(self.probabilities) == 0:
            return 0.0
        
        probs = np.clip(self.probabilities, 1e-8, 1.0)
        entropies = -np.sum(probs * np.log(probs), axis=1)
        return np.mean(entropies)
    
    def get_bbox_3d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D bounding box"""
        return self.bbox_min, self.bbox_max
    
    def compute_iou_3d(self, other: 'Instance3D', voxel_size: float) -> float:
        """
        Compute 3D IoU with another instance.
        
        Uses voxel-based approximation:
        IoU = |A ∩ B| / |A ∪ B|
        
        Args:
            other: Another Instance3D
            voxel_size: Voxel size for discretization
            
        Returns:
            IoU value [0, 1]
        """
        # Convert voxel centers to discrete grid coordinates
        def to_grid(centers):
            return set(map(tuple, np.round(centers / voxel_size).astype(np.int32)))
        
        grid_a = to_grid(self.voxel_centers)
        grid_b = to_grid(other.voxel_centers)
        
        intersection = len(grid_a & grid_b)
        union = len(grid_a | grid_b)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def compute_min_distance(self, other: 'Instance3D') -> float:
        """
        Compute minimum distance between two instances.
        
        Args:
            other: Another Instance3D
            
        Returns:
            Minimum distance in meters
        """
        from scipy.spatial.distance import cdist
        
        if len(self.voxel_centers) == 0 or len(other.voxel_centers) == 0:
            return float('inf')
        
        # Use sampling for large instances
        max_points = 1000
        points_a = self.voxel_centers
        points_b = other.voxel_centers
        
        if len(points_a) > max_points:
            idx = np.random.choice(len(points_a), max_points, replace=False)
            points_a = points_a[idx]
        
        if len(points_b) > max_points:
            idx = np.random.choice(len(points_b), max_points, replace=False)
            points_b = points_b[idx]
        
        distances = cdist(points_a, points_b)
        return np.min(distances)
    
    def merge_with(self, other: 'Instance3D'):
        """
        Merge another instance into this one.
        
        Args:
            other: Instance to merge
        """
        # Combine voxels
        self.voxel_centers = np.vstack([self.voxel_centers, other.voxel_centers])
        self.probabilities = np.vstack([self.probabilities, other.probabilities])
        
        # Update properties
        self.num_voxels = len(self.voxel_centers)
        self.mean_confidence = np.mean(self.probabilities[:, self.class_id])
        self.view_count += other.view_count
        self.entropy = self._calculate_entropy()
        
        # Update geometry
        self.bbox_min = np.minimum(self.bbox_min, other.bbox_min)
        self.bbox_max = np.maximum(self.bbox_max, other.bbox_max)
        self.centroid = np.mean(self.voxel_centers, axis=0)
        
        # Update metadata
        self.source_views.extend(other.source_views)
        self.merged_from.append(other.instance_id)
        
        logger.debug(f"Merged {other.instance_id} into {self.instance_id}: "
                    f"{self.num_voxels} voxels, {self.view_count} views")
    
    def to_dict(self) -> Dict:
        """Export instance to dictionary"""
        return {
            'instance_id': self.instance_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'num_voxels': self.num_voxels,
            'mean_confidence': float(self.mean_confidence),
            'view_count': self.view_count,
            'entropy': float(self.entropy),
            'bbox_min': self.bbox_min.tolist() if self.bbox_min is not None else None,
            'bbox_max': self.bbox_max.tolist() if self.bbox_max is not None else None,
            'centroid': self.centroid.tolist() if self.centroid is not None else None,
            'merged_from': self.merged_from
        }


class InstanceMerger:
    """Merges fragmented 3D instances"""
    
    def __init__(
        self,
        voxel_size: float,
        class_names: List[str],
        config: Optional[Dict] = None
    ):
        """
        Initialize instance merger.
        
        Args:
            voxel_size: Voxel size in meters
            class_names: List of class names
            config: Configuration dictionary
        """
        self.voxel_size = voxel_size
        self.class_names = class_names
        self.config = config or {}
        
        # DBSCAN parameters
        self.eps = self.config.get('dbscan_eps_voxel_mul', 3.0) * voxel_size
        self.min_samples = self.config.get('dbscan_min_pts', 10)
        
        # Merge parameters
        self.iou_thresh = self.config.get('iou_merge_thresh', 0.3)
        self.distance_thresh = self.config.get('skeleton_gap_thresh_cm', 2.0) / 100  # to meters
        
        logger.info(f"InstanceMerger initialized: eps={self.eps:.4f}m, "
                   f"min_samples={self.min_samples}, iou_thresh={self.iou_thresh}")
    
    def cluster_class_voxels(
        self,
        voxel_centers: np.ndarray,
        class_id: int
    ) -> np.ndarray:
        """
        Cluster voxels of a single class using DBSCAN.
        
        Args:
            voxel_centers: Nx3 voxel coordinates
            class_id: Class index
            
        Returns:
            N-length array of cluster labels (-1 for noise)
        """
        if len(voxel_centers) < self.min_samples:
            logger.warning(f"Class {class_id}: Too few voxels ({len(voxel_centers)}) for clustering")
            return np.zeros(len(voxel_centers), dtype=np.int32)
        
        logger.info(f"Clustering class {class_id}: {len(voxel_centers)} voxels")
        
        # DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(voxel_centers)
        
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = np.sum(labels == -1)
        
        logger.info(f"  Found {num_clusters} clusters, {num_noise} noise points")
        
        return labels
    
    def create_instances_from_clusters(
        self,
        voxel_centers: np.ndarray,
        cluster_labels: np.ndarray,
        class_id: int,
        probabilities: np.ndarray
    ) -> List[Instance3D]:
        """
        Create Instance3D objects from cluster labels.
        
        Args:
            voxel_centers: Nx3 voxel coordinates
            cluster_labels: N cluster labels
            class_id: Class index
            probabilities: NxC probability distribution
            
        Returns:
            List of Instance3D objects
        """
        instances = []
        
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for cluster_id in unique_labels:
            mask = cluster_labels == cluster_id
            
            instance_id = f"class{class_id}_cluster{cluster_id}"
            instance = Instance3D(
                instance_id=instance_id,
                class_id=class_id,
                class_name=self.class_names[class_id],
                voxel_centers=voxel_centers[mask],
                probabilities=probabilities[mask]
            )
            
            instances.append(instance)
        
        return instances
    
    def merge_nearby_instances(
        self,
        instances: List[Instance3D]
    ) -> List[Instance3D]:
        """
        Merge instances that are close or overlapping.
        
        Uses IoU and minimum distance criteria.
        
        Args:
            instances: List of Instance3D objects
            
        Returns:
            List of merged instances
        """
        if len(instances) <= 1:
            return instances
        
        logger.info(f"Merging {len(instances)} instances...")
        
        # Build merge graph
        n = len(instances)
        merge_graph = defaultdict(set)
        
        for i in range(n):
            for j in range(i + 1, n):
                inst_a = instances[i]
                inst_b = instances[j]
                
                # Check if same class
                if inst_a.class_id != inst_b.class_id:
                    continue
                
                # Compute IoU
                iou = inst_a.compute_iou_3d(inst_b, self.voxel_size)
                
                # Compute minimum distance
                min_dist = inst_a.compute_min_distance(inst_b)
                
                # Merge if IoU > threshold OR distance < threshold
                should_merge = (iou > self.iou_thresh) or (min_dist < self.distance_thresh)
                
                if should_merge:
                    merge_graph[i].add(j)
                    merge_graph[j].add(i)
                    logger.debug(f"Merge candidate: {inst_a.instance_id} <-> {inst_b.instance_id} "
                               f"(IoU={iou:.3f}, dist={min_dist:.4f}m)")
        
        # Find connected components (instances to merge together)
        visited = set()
        merged_instances = []
        
        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor in merge_graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for i in range(n):
            if i not in visited:
                component = set()
                dfs(i, component)
                
                # Merge all instances in component
                component_list = sorted(component)
                merged_inst = instances[component_list[0]]
                
                for j in component_list[1:]:
                    merged_inst.merge_with(instances[j])
                
                merged_instances.append(merged_inst)
        
        logger.info(f"Merged to {len(merged_instances)} instances")
        
        return merged_instances
    
    def merge_instances(
        self,
        voxel_centers: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray
    ) -> List[Instance3D]:
        """
        Main instance merging pipeline.
        
        Args:
            voxel_centers: Nx3 voxel coordinates
            labels: N class labels
            probabilities: NxC probability distribution
            
        Returns:
            List of merged Instance3D objects
        """
        logger.info("Starting instance merging pipeline...")
        
        all_instances = []
        
        # Process each class separately
        unique_classes = np.unique(labels)
        
        for class_id in unique_classes:
            class_mask = labels == class_id
            class_voxels = voxel_centers[class_mask]
            class_probs = probabilities[class_mask]
            
            logger.info(f"\nProcessing class {class_id} ({self.class_names[class_id]}): "
                       f"{len(class_voxels)} voxels")
            
            # Cluster voxels
            cluster_labels = self.cluster_class_voxels(class_voxels, class_id)
            
            # Create instances from clusters
            instances = self.create_instances_from_clusters(
                class_voxels, cluster_labels, class_id, class_probs
            )
            
            logger.info(f"  Created {len(instances)} initial instances")
            
            # Merge nearby instances
            instances = self.merge_nearby_instances(instances)
            
            logger.info(f"  Final: {len(instances)} instances after merging")
            
            all_instances.extend(instances)
        
        logger.info(f"\nTotal instances: {len(all_instances)}")
        
        return all_instances


def merge_pipeline(
    fusion_result: Dict,
    voxel_size: float,
    class_names: List[str],
    config: Optional[Dict] = None
) -> List[Instance3D]:
    """
    High-level instance merging pipeline.
    
    Args:
        fusion_result: Result from LabelFusion.finalize()
        voxel_size: Voxel size in meters
        class_names: List of class names
        config: Configuration dictionary
        
    Returns:
        List of merged Instance3D objects
    """
    merger = InstanceMerger(voxel_size, class_names, config)
    
    instances = merger.merge_instances(
        fusion_result['voxel_centers'],
        fusion_result['labels'],
        fusion_result['probabilities']
    )
    
    return instances


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    np.random.seed(42)
    
    # Class 0: Two nearby clusters
    cluster1 = np.random.randn(50, 3) * 0.02 + [0, 0, 2]
    cluster2 = np.random.randn(50, 3) * 0.02 + [0.05, 0, 2]  # Close to cluster1
    cluster3 = np.random.randn(50, 3) * 0.02 + [0.5, 0, 2]   # Far away
    
    voxel_centers = np.vstack([cluster1, cluster2, cluster3])
    labels = np.zeros(len(voxel_centers), dtype=np.int32)
    
    # Probabilities
    probabilities = np.zeros((len(voxel_centers), 5))
    probabilities[:, 0] = 0.8  # Class 0
    probabilities[:, 4] = 0.2  # Background
    
    # Merge
    merger = InstanceMerger(
        voxel_size=0.01,
        class_names=['crack', 'spalling', 'efflorescence', 'exposed_rebar', 'background'],
        config={'dbscan_eps_voxel_mul': 3.0, 'iou_merge_thresh': 0.1}
    )
    
    instances = merger.merge_instances(voxel_centers, labels, probabilities)
    
    print(f"\nResults:")
    for inst in instances:
        print(f"  {inst.instance_id}: {inst.num_voxels} voxels, "
              f"conf={inst.mean_confidence:.3f}, views={inst.view_count}")
