"""
Unit Tests for YOLO + SFM 3D Fusion Pipeline
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calib_io import CameraCalibration, load_camera_info
from align_depth_to_rgb import backproject_depth, project_3d_to_image
from project_mask_to_A import MaskProjector
from fusion_3d import VoxelGrid, VoxelData, LabelFusion
from instance_merge import Instance3D, InstanceMerger
from measurement import InstanceMeasurement
from utils import normalize_vector, rotation_matrix_from_vectors, compute_statistics


class TestCalibIO:
    """Test camera calibration I/O"""
    
    def test_camera_calibration(self):
        K = np.eye(3)
        D = np.zeros(8)
        calib = CameraCalibration(640, 480, K, D)
        
        assert calib.width == 640
        assert calib.height == 480
        assert np.allclose(calib.K, K)


class TestAlignment:
    """Test depth-to-RGB alignment"""
    
    def test_backproject_depth(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        
        u = np.array([320, 420])
        v = np.array([240, 240])
        depth = np.array([2.0, 2.0])
        
        points_3d = backproject_depth(u, v, depth, K)
        
        assert points_3d.shape == (2, 3)
        assert np.allclose(points_3d[0], [0, 0, 2.0])
    
    def test_project_3d_to_image(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        
        points_3d = np.array([[0, 0, 2.0], [0.2, 0.1, 2.0]])
        
        coords, valid = project_3d_to_image(points_3d, K)
        
        assert coords.shape == (2, 2)
        assert np.allclose(coords[0], [320, 240])
        assert np.all(valid)


class TestMaskProjection:
    """Test YOLO mask projection"""
    
    def test_rasterize_polygon(self):
        K = np.eye(3) * 100
        K[2, 2] = 1
        D = np.zeros(8)
        R = np.eye(3)
        t = np.zeros((3, 1))
        
        projector = MaskProjector(K, D, R, t, ['crack', 'spalling'])
        
        polygon = [[10, 10], [50, 10], [50, 50], [10, 50]]
        mask = projector.rasterize_polygon(polygon, (100, 100))
        
        assert mask.shape == (100, 100)
        assert np.sum(mask) > 0


class TestFusion:
    """Test 3D label fusion"""
    
    def test_voxel_grid_creation(self):
        grid = VoxelGrid(voxel_size=0.01, num_classes=5)
        
        points = np.random.randn(100, 3)
        grid.accumulate(points, class_id=0, score=0.8, view_weight=1.0)
        
        assert len(grid.voxels) > 0
    
    def test_voxel_data_update(self):
        voxel = VoxelData(num_classes=5)
        
        voxel.update_logodds(class_id=0, score=0.8, weight=1.0, num_points=10)
        voxel.update_logodds(class_id=0, score=0.9, weight=1.0, num_points=5)
        
        probs = voxel.get_probabilities()
        
        assert probs.shape == (5,)
        assert np.isclose(np.sum(probs), 1.0)
        assert probs[0] > probs[1]  # Class 0 should have higher probability
    
    def test_label_fusion(self):
        fusion = LabelFusion(
            voxel_size=0.01,
            num_classes=5,
            class_names=['crack', 'spalling', 'efflorescence', 'exposed_rebar', 'background']
        )
        
        # Simulate projection
        points = np.random.randn(50, 3)
        projection = {
            'points_3d': points,
            'class': 'crack',
            'class_id': 0,
            'score': 0.8,
            'view_weight': 1.0,
            'num_points': len(points)
        }
        
        fusion.fuse_projection(projection)
        
        result = fusion.finalize(prob_thresh=0.5)
        
        assert 'voxel_centers' in result
        assert 'labels' in result
        assert 'probabilities' in result
        assert len(result['labels']) > 0


class TestInstanceMerge:
    """Test instance merging"""
    
    def test_instance_creation(self):
        points = np.random.randn(100, 3)
        probs = np.random.rand(100, 5)
        probs /= probs.sum(axis=1, keepdims=True)
        
        instance = Instance3D(
            instance_id='test_inst',
            class_id=0,
            class_name='crack',
            voxel_centers=points,
            probabilities=probs
        )
        
        assert instance.num_voxels == 100
        assert instance.bbox_min is not None
        assert instance.bbox_max is not None
    
    def test_instance_iou(self):
        points1 = np.random.randn(50, 3) * 0.1
        points2 = np.random.randn(50, 3) * 0.1 + [0.05, 0, 0]  # Overlapping
        
        probs = np.random.rand(50, 5)
        probs /= probs.sum(axis=1, keepdims=True)
        
        inst1 = Instance3D('i1', 0, 'crack', points1, probs)
        inst2 = Instance3D('i2', 0, 'crack', points2, probs)
        
        iou = inst1.compute_iou_3d(inst2, voxel_size=0.01)
        
        assert 0 <= iou <= 1
    
    def test_instance_merger(self):
        # Create two nearby clusters
        cluster1 = np.random.randn(30, 3) * 0.02
        cluster2 = np.random.randn(30, 3) * 0.02 + [0.05, 0, 0]
        
        points = np.vstack([cluster1, cluster2])
        labels = np.zeros(len(points), dtype=np.int32)
        probs = np.random.rand(len(points), 5)
        probs /= probs.sum(axis=1, keepdims=True)
        
        merger = InstanceMerger(
            voxel_size=0.01,
            class_names=['crack', 'spalling', 'efflorescence', 'exposed_rebar', 'background'],
            config={'dbscan_eps_voxel_mul': 5.0, 'iou_merge_thresh': 0.1}
        )
        
        instances = merger.merge_instances(points, labels, probs)
        
        assert len(instances) > 0


class TestMeasurement:
    """Test geometric measurements"""
    
    def test_crack_measurement(self):
        # Create linear crack
        t = np.linspace(0, 1, 100)
        points = np.column_stack([
            t * 0.5,
            np.zeros(100),
            np.ones(100) * 2.0
        ])
        
        probs = np.zeros((100, 5))
        probs[:, 0] = 0.8
        
        instance = Instance3D('crack_test', 0, 'crack', points, probs)
        
        measurer = InstanceMeasurement()
        measurements = measurer.measure_instance(instance, voxel_size=0.01)
        
        assert 'length_m' in measurements
        assert measurements['length_m'] > 0
        assert measurements['class_name'] == 'crack'
    
    def test_area_measurement(self):
        # Create planar defect
        x, y = np.meshgrid(np.linspace(0, 0.1, 20), np.linspace(0, 0.1, 20))
        points = np.column_stack([x.flatten(), y.flatten(), np.ones(400) * 2.0])
        
        probs = np.zeros((400, 5))
        probs[:, 1] = 0.8  # Spalling
        
        instance = Instance3D('spall_test', 1, 'spalling', points, probs)
        
        measurer = InstanceMeasurement()
        measurements = measurer.measure_instance(instance, voxel_size=0.005)
        
        assert 'area_m2' in measurements
        assert measurements['area_m2'] > 0


class TestUtils:
    """Test utility functions"""
    
    def test_normalize_vector(self):
        v = np.array([3, 4, 0])
        v_norm = normalize_vector(v)
        
        assert np.isclose(np.linalg.norm(v_norm), 1.0)
    
    def test_rotation_matrix(self):
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        
        R = rotation_matrix_from_vectors(v1, v2)
        
        v1_rotated = R @ v1
        
        assert np.allclose(v1_rotated, v2)
    
    def test_compute_statistics(self):
        values = np.array([1, 2, 3, 4, 5])
        stats = compute_statistics(values)
        
        assert stats['count'] == 5
        assert stats['mean'] == 3.0
        assert stats['median'] == 3.0


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
