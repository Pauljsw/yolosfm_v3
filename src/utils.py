"""
Utility Functions
Common utilities for visualization, logging, and data processing.
"""
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict, output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def list_files(directory: str, extension: str = '*') -> List[str]:
    """
    List files in directory with given extension.

    Args:
        directory: Directory path
        extension: File extension (e.g., '.png', '*' for all)

    Returns:
        List of file paths
    """
    path = Path(directory)

    if not path.exists():
        return []

    if extension == '*':
        pattern = '*'
    else:
        if not extension.startswith('.'):
            extension = '.' + extension
        pattern = f'*{extension}'

    files = sorted(path.glob(pattern))
    return [str(f) for f in files]


def parse_rgb_dpt_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse RGB/Depth filename pattern: camera_RGB_X_Y.png or camera_DPT_X_Y.png

    Args:
        filename: Filename to parse

    Returns:
        (type, id1, id2) where type is 'RGB' or 'DPT', or None if pattern doesn't match
    """
    import re

    stem = Path(filename).stem
    match = re.match(r'camera_(RGB|DPT)_(\d+)_(\d+)', stem)

    if match:
        return match.group(1), match.group(2), match.group(3)

    return None


def find_rgb_depth_pairs(rgb_dir: str, depth_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find matching RGB-Depth image pairs based on filename pattern.

    Args:
        rgb_dir: RGB images directory
        depth_dir: Depth images directory

    Returns:
        List of (rgb_path, depth_path, pair_id) tuples
    """
    rgb_files = list_files(rgb_dir, '.png')
    depth_files = list_files(depth_dir, '.png')

    # Parse RGB files
    rgb_map = {}  # (id1, id2) -> path
    for rgb_path in rgb_files:
        parsed = parse_rgb_dpt_filename(rgb_path)
        if parsed and parsed[0] == 'RGB':
            _, id1, id2 = parsed
            rgb_map[(id1, id2)] = rgb_path

    # Parse Depth files and match
    pairs = []
    for depth_path in depth_files:
        parsed = parse_rgb_dpt_filename(depth_path)
        if parsed and parsed[0] == 'DPT':
            _, id1, id2 = parsed
            if (id1, id2) in rgb_map:
                rgb_path = rgb_map[(id1, id2)]
                pair_id = f"{id1}_{id2}"
                pairs.append((rgb_path, depth_path, pair_id))

    logging.info(f"Found {len(pairs)} RGB-Depth pairs")
    return pairs


def detect_depth_unit(depth_img: np.ndarray) -> str:
    """
    Auto-detect depth image unit (m or mm) based on value range.

    Args:
        depth_img: Depth image array

    Returns:
        'm' or 'mm'
    """
    valid_depths = depth_img[depth_img > 0]

    if len(valid_depths) == 0:
        logging.warning("No valid depth values found, assuming mm")
        return 'mm'

    median_depth = np.median(valid_depths)
    max_depth = np.max(valid_depths)

    # Heuristic: if median > 100, likely mm; if < 100, likely m
    # Typical indoor scenes: 1-10m (1000-10000mm)
    if median_depth > 100:
        return 'mm'
    else:
        return 'm'


def compute_statistics(values: np.ndarray) -> Dict:
    """
    Compute basic statistics for array.
    
    Args:
        values: Numpy array
        
    Returns:
        Dictionary with statistics
    """
    if len(values) == 0:
        return {
            'count': 0,
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'median': 0
        }
    
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values))
    }


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length"""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix that rotates vec1 to vec2.
    
    Args:
        vec1: Source vector (3,)
        vec2: Target vector (3,)
        
    Returns:
        3x3 rotation matrix
    """
    vec1 = normalize_vector(vec1)
    vec2 = normalize_vector(vec2)
    
    v = np.cross(vec1, vec2)
    c = np.dot(vec1, vec2)
    s = np.linalg.norm(v)
    
    if s < 1e-8:
        # Vectors are parallel
        return np.eye(3) if c > 0 else -np.eye(3)
    
    # Skew-symmetric cross-product matrix
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    
    return R


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transform points using rotation and translation.
    
    Args:
        points: Nx3 points
        R: 3x3 rotation matrix
        t: 3x1 or (3,) translation vector
        
    Returns:
        Transformed Nx3 points
    """
    if t.ndim == 2:
        t = t.flatten()
    
    return (R @ points.T).T + t


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box.
    
    Args:
        points: Nx3 points
        
    Returns:
        Tuple of (min_bound, max_bound)
    """
    if len(points) == 0:
        return np.zeros(3), np.zeros(3)
    
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    return min_bound, max_bound


def downsample_points(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Downsample point cloud using voxel grid.
    
    Args:
        points: Nx3 points
        voxel_size: Voxel size for downsampling
        
    Returns:
        Downsampled points
    """
    if len(points) == 0:
        return points
    
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    
    # Get unique voxels and their first occurrence
    unique_voxels, first_indices = np.unique(voxel_indices, axis=0, return_index=True)
    
    # Return points at first occurrence of each voxel
    downsampled = points[first_indices]
    
    return downsampled


def sample_points_uniform(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Uniformly sample points.
    
    Args:
        points: Nx3 points
        n_samples: Number of samples
        
    Returns:
        Sampled points
    """
    if len(points) <= n_samples:
        return points
    
    indices = np.random.choice(len(points), n_samples, replace=False)
    return points[indices]


def compute_normal_from_points(points: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Estimate normal vector from point neighborhood.
    
    Args:
        points: Nx3 points
        k: Number of neighbors
        
    Returns:
        Normal vector (3,)
    """
    from sklearn.decomposition import PCA
    
    if len(points) < 3:
        return np.array([0, 0, 1])
    
    # Use PCA to find normal
    pca = PCA(n_components=3)
    pca.fit(points - np.mean(points, axis=0))
    
    # Normal is the direction with smallest variance
    normal = pca.components_[-1]
    
    return normalize_vector(normal)


def visualize_point_cloud_matplotlib(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = 'Point Cloud',
    save_path: Optional[str] = None
):
    """
    Visualize point cloud using matplotlib.
    
    Args:
        points: Nx3 points
        colors: Nx3 RGB colors (0-255) or None
        title: Plot title
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        colors = colors / 255.0  # Normalize to 0-1
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1, alpha=0.6)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=points[:, 2], s=1, cmap='viridis', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                         points[:, 1].max() - points[:, 1].min(),
                         points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def progress_bar(current: int, total: int, bar_length: int = 50):
    """
    Display progress bar.
    
    Args:
        current: Current iteration
        total: Total iterations
        bar_length: Length of progress bar
    """
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    
    print(f'\rProgress: [{arrow}{spaces}] {int(percent * 100)}% ({current}/{total})', end='', flush=True)
    
    if current == total:
        print()  # New line when complete


class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name: str = 'Operation'):
        self.name = name
        self.start_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {self.name}...")
        return self
    
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} completed in {elapsed:.2f} seconds")


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")
    
    # Test statistics
    values = np.random.randn(100)
    stats = compute_statistics(values)
    print(f"Statistics: {stats}")
    
    # Test rotation
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    R = rotation_matrix_from_vectors(v1, v2)
    print(f"Rotation matrix:\n{R}")
    print(f"Rotated v1: {R @ v1}")
    
    # Test transform
    points = np.random.randn(10, 3)
    R = np.eye(3)
    t = np.array([1, 2, 3])
    transformed = transform_points(points, R, t)
    print(f"Transformed shape: {transformed.shape}")
    
    print("Utility tests passed!")
