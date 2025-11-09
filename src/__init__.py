"""
YOLO + SFM Simple Pipeline Package
"""
__version__ = '2.0.0'

# Core utilities
from .utils import setup_logging, load_config, Timer

# Calibration and I/O
from .calib_io import load_camera_info, load_poses, CameraCalibration

# COLMAP
try:
    from .colmap_sfm import COLMAPRunner, run_colmap_sfm_auto
    from .colmap_io import read_cameras_binary, read_images_binary, read_points3D_binary
    _HAS_COLMAP = True
except ImportError:
    _HAS_COLMAP = False
    COLMAPRunner = None
    run_colmap_sfm_auto = None
    read_cameras_binary = None
    read_images_binary = None
    read_points3D_binary = None

# Simple Pipeline modules
try:
    from .pixel_calibration import calculate_pixel_scale_map, run_calibration
except ImportError:
    calculate_pixel_scale_map = None
    run_calibration = None

try:
    from .point_cloud_overlay import overlay_masks_on_pointcloud
except ImportError:
    overlay_masks_on_pointcloud = None

try:
    from .measure_cracks_simple import measure_crack_in_image, run_measurement
except ImportError:
    measure_crack_in_image = None
    run_measurement = None

__all__ = [
    # Utilities
    'setup_logging',
    'load_config',
    'Timer',
    # Calibration
    'load_camera_info',
    'load_poses',
    'CameraCalibration',
    # COLMAP
    'COLMAPRunner',
    'run_colmap_sfm_auto',
    'read_cameras_binary',
    'read_images_binary',
    'read_points3D_binary',
    # Simple Pipeline
    'calculate_pixel_scale_map',
    'run_calibration',
    'overlay_masks_on_pointcloud',
    'measure_crack_in_image',
    'run_measurement',
]
