"""
YOLO + SFM 3D Fusion Package
"""
__version__ = '1.0.0'

from .calib_io import load_camera_info, load_poses, CameraCalibration
from .align_depth_to_rgb import align_depth_to_rgb
from .project_mask_to_A import MaskProjector, project_all_masks
from .fusion_3d import VoxelGrid, LabelFusion
from .instance_merge import Instance3D, InstanceMerger, merge_pipeline
from .measurement import InstanceMeasurement, measure_all_instances
from .export_results import ResultExporter, export_all_results
from .utils import setup_logging, load_config, Timer

# COLMAP SFM (optional, requires COLMAP installation)
try:
    from .colmap_sfm import COLMAPRunner, run_colmap_sfm_auto
    _HAS_COLMAP = True
except ImportError:
    _HAS_COLMAP = False
    COLMAPRunner = None
    run_colmap_sfm_auto = None

__all__ = [
    'load_camera_info',
    'load_poses',
    'CameraCalibration',
    'align_depth_to_rgb',
    'MaskProjector',
    'project_all_masks',
    'VoxelGrid',
    'LabelFusion',
    'Instance3D',
    'InstanceMerger',
    'merge_pipeline',
    'InstanceMeasurement',
    'measure_all_instances',
    'ResultExporter',
    'export_all_results',
    'setup_logging',
    'load_config',
    'Timer',
    'COLMAPRunner',
    'run_colmap_sfm_auto'
]
