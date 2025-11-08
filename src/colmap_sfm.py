"""
COLMAP SFM Wrapper
Automates Structure from Motion using COLMAP.
"""
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import shutil

logger = logging.getLogger(__name__)


class COLMAPRunner:
    """Wrapper for COLMAP SFM pipeline"""
    
    def __init__(self, colmap_executable: str = 'colmap'):
        """
        Initialize COLMAP runner.
        
        Args:
            colmap_executable: Path to COLMAP executable
        """
        self.colmap_exe = colmap_executable
        self._check_colmap_installed()
    
    def _check_colmap_installed(self):
        """Check if COLMAP is installed"""
        try:
            result = subprocess.run(
                [self.colmap_exe, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"COLMAP found: {result.stdout.strip()}")
            else:
                logger.warning("COLMAP executable found but version check failed")
        except FileNotFoundError:
            logger.error(f"COLMAP not found at: {self.colmap_exe}")
            logger.error("Install COLMAP: https://colmap.github.io/install.html")
            raise RuntimeError("COLMAP not installed")
        except Exception as e:
            logger.warning(f"Could not verify COLMAP installation: {e}")
    
    def run_sfm_pipeline(
        self,
        image_dir: str,
        output_dir: str,
        camera_model: str = 'OPENCV',
        use_gpu: bool = True,
        quality: str = 'high'
    ) -> str:
        """
        Run complete COLMAP SFM pipeline.
        
        Args:
            image_dir: Directory containing RGB images
            output_dir: Output directory for COLMAP results
            camera_model: Camera model (OPENCV, PINHOLE, RADIAL, etc.)
            use_gpu: Whether to use GPU acceleration
            quality: 'low', 'medium', 'high', 'extreme'
            
        Returns:
            Path to sparse reconstruction directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        database_path = output_path / 'database.db'
        sparse_dir = output_path / 'sparse'
        sparse_dir.mkdir(exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("Starting COLMAP SFM Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Feature Extraction
        logger.info("Step 1/4: Feature Extraction")
        self._run_feature_extraction(
            database_path, image_dir, camera_model, use_gpu, quality
        )
        
        # Step 2: Feature Matching
        logger.info("Step 2/4: Feature Matching")
        self._run_feature_matching(database_path, use_gpu, quality)
        
        # Step 3: Sparse Reconstruction
        logger.info("Step 3/4: Sparse Reconstruction")
        self._run_mapper(database_path, image_dir, sparse_dir)
        
        # Step 4: Model Conversion
        logger.info("Step 4/4: Converting to standard format")
        model_dir = sparse_dir / '0'  # COLMAP creates numbered models
        
        if not model_dir.exists():
            raise RuntimeError(f"Reconstruction failed: {model_dir} not found")
        
        logger.info(f"SFM reconstruction complete: {model_dir}")
        
        return str(model_dir)
    
    def _run_feature_extraction(
        self,
        database_path: Path,
        image_dir: str,
        camera_model: str,
        use_gpu: bool,
        quality: str
    ):
        """Extract features from images"""
        quality_settings = {
            'low': {'max_image_size': 1600, 'max_num_features': 4096},
            'medium': {'max_image_size': 2400, 'max_num_features': 8192},
            'high': {'max_image_size': 3200, 'max_num_features': 16384},
            'extreme': {'max_image_size': 4800, 'max_num_features': 32768}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        cmd = [
            self.colmap_exe, 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', image_dir,
            '--ImageReader.camera_model', camera_model,
            '--ImageReader.single_camera', '1',  # Assume same camera for all images
            '--SiftExtraction.max_image_size', str(settings['max_image_size']),
            '--SiftExtraction.max_num_features', str(settings['max_num_features']),
        ]
        
        if use_gpu:
            cmd.extend(['--SiftExtraction.use_gpu', '1'])
        
        self._run_command(cmd, "Feature extraction failed")
    
    def _run_feature_matching(
        self,
        database_path: Path,
        use_gpu: bool,
        quality: str
    ):
        """Match features between images"""
        # Use exhaustive matching for small datasets, sequential for large
        cmd = [
            self.colmap_exe, 'exhaustive_matcher',
            '--database_path', str(database_path),
        ]
        
        if use_gpu:
            cmd.extend(['--SiftMatching.use_gpu', '1'])
        
        # For larger datasets, consider sequential or spatial matching
        # cmd = [self.colmap_exe, 'sequential_matcher', ...]
        
        self._run_command(cmd, "Feature matching failed")
    
    def _run_mapper(
        self,
        database_path: Path,
        image_dir: str,
        output_dir: Path
    ):
        """Run sparse reconstruction"""
        cmd = [
            self.colmap_exe, 'mapper',
            '--database_path', str(database_path),
            '--image_path', image_dir,
            '--output_path', str(output_dir),
            '--Mapper.ba_refine_focal_length', '1',
            '--Mapper.ba_refine_extra_params', '1',
        ]
        
        self._run_command(cmd, "Sparse reconstruction failed")
    
    def _run_command(self, cmd: list, error_msg: str):
        """Run COLMAP command"""
        logger.debug(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"{error_msg}: {result.stderr}")
            
            logger.debug(f"Command successful")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{error_msg}: Command timeout")
        except Exception as e:
            raise RuntimeError(f"{error_msg}: {e}")
    
    def convert_to_poses_json(
        self,
        sparse_model_dir: str,
        output_json: str,
        image_names: Optional[list] = None
    ) -> Dict:
        """
        Convert COLMAP sparse model to poses.json format.
        
        Args:
            sparse_model_dir: Directory containing COLMAP sparse model
            output_json: Output JSON file path
            image_names: Optional list of image names to include
            
        Returns:
            Dictionary of poses
        """
        logger.info(f"Converting COLMAP model to poses.json: {sparse_model_dir}")
        
        model_path = Path(sparse_model_dir)
        
        # Read COLMAP files
        cameras = self._read_cameras_txt(model_path / 'cameras.txt')
        images = self._read_images_txt(model_path / 'images.txt')
        
        # Convert to our format
        poses = {}
        
        for img_id, img_data in images.items():
            img_name = img_data['name']
            
            # Skip if not in requested list
            if image_names is not None and img_name not in image_names:
                continue
            
            # Get camera
            cam_id = img_data['camera_id']
            camera = cameras.get(cam_id)
            
            if camera is None:
                logger.warning(f"Camera {cam_id} not found for image {img_name}")
                continue
            
            # COLMAP uses quaternion (qw, qx, qy, qz) and translation
            qvec = img_data['qvec']  # [qw, qx, qy, qz]
            tvec = img_data['tvec']  # [tx, ty, tz]
            
            # Convert quaternion to rotation matrix
            R = self._qvec_to_rotmat(qvec)
            
            # COLMAP convention: world-to-camera
            # We need camera-to-world for our pipeline
            # R_c2w = R_w2c^T, t_c2w = -R_c2w @ t_w2c
            R_c2w = R.T
            t_c2w = -R_c2w @ tvec
            
            # Build intrinsic matrix
            K = self._build_intrinsic_matrix(camera)
            
            poses[img_name] = {
                'filename': img_name,
                'R': R_c2w.tolist(),
                't': t_c2w.reshape(3, 1).tolist(),
                'K': K.tolist()
            }
        
        logger.info(f"Converted {len(poses)} image poses")
        
        # Save to JSON
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(poses, f, indent=2)
        
        logger.info(f"Poses saved to: {output_json}")
        
        return poses
    
    def _read_cameras_txt(self, cameras_file: Path) -> Dict:
        """
        Read COLMAP cameras file (supports both .txt and .bin formats).

        Args:
            cameras_file: Path to cameras.txt (will also try cameras.bin)

        Returns:
            Dictionary of camera data
        """
        import struct

        # Try binary format first
        cameras_bin = cameras_file.parent / 'cameras.bin'
        cameras_txt = cameras_file

        if cameras_bin.exists():
            logger.debug(f"Reading COLMAP cameras (binary): {cameras_bin}")
            cameras = {}

            # COLMAP camera model ID to name mapping
            CAMERA_MODEL_NAMES = {
                0: 'SIMPLE_PINHOLE',
                1: 'PINHOLE',
                2: 'SIMPLE_RADIAL',
                3: 'RADIAL',
                4: 'OPENCV',
                5: 'OPENCV_FISHEYE',
                6: 'FULL_OPENCV',
                7: 'FOV',
                8: 'SIMPLE_RADIAL_FISHEYE',
                9: 'RADIAL_FISHEYE',
                10: 'THIN_PRISM_FISHEYE'
            }

            with open(cameras_bin, 'rb') as f:
                num_cameras = struct.unpack('Q', f.read(8))[0]

                for _ in range(num_cameras):
                    cam_id = struct.unpack('I', f.read(4))[0]
                    model_id = struct.unpack('i', f.read(4))[0]
                    width = struct.unpack('Q', f.read(8))[0]
                    height = struct.unpack('Q', f.read(8))[0]

                    model_name = CAMERA_MODEL_NAMES.get(model_id, f'MODEL_{model_id}')

                    # Number of params depends on model
                    num_params_map = {
                        'SIMPLE_PINHOLE': 3,
                        'PINHOLE': 4,
                        'SIMPLE_RADIAL': 4,
                        'RADIAL': 5,
                        'OPENCV': 8,
                        'OPENCV_FISHEYE': 8,
                        'FULL_OPENCV': 12,
                        'FOV': 5,
                        'SIMPLE_RADIAL_FISHEYE': 4,
                        'RADIAL_FISHEYE': 5,
                        'THIN_PRISM_FISHEYE': 12
                    }

                    num_params = num_params_map.get(model_name, 8)  # default to 8
                    params = struct.unpack(f'{num_params}d', f.read(8 * num_params))

                    cameras[cam_id] = {
                        'model': model_name,
                        'width': int(width),
                        'height': int(height),
                        'params': list(params)
                    }

            logger.debug(f"Read {len(cameras)} cameras from binary")
            return cameras

        # Fallback to text format
        elif cameras_txt.exists():
            logger.debug(f"Reading COLMAP cameras (text): {cameras_txt}")
            cameras = {}

            with open(cameras_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    cam_id = int(parts[0])
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = [float(p) for p in parts[4:]]

                    cameras[cam_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'params': params
                    }

            logger.debug(f"Read {len(cameras)} cameras from text")
            return cameras

        else:
            raise FileNotFoundError(
                f"COLMAP cameras file not found. Tried: {cameras_bin}, {cameras_txt}"
            )
    
    def _read_images_txt(self, images_file: Path) -> Dict:
        """
        Read COLMAP images file (supports both .txt and .bin formats).

        Args:
            images_file: Path to images.txt (will also try images.bin)

        Returns:
            Dictionary of image data
        """
        import struct

        # Try binary format first
        images_bin = images_file.parent / 'images.bin'
        images_txt = images_file

        if images_bin.exists():
            logger.debug(f"Reading COLMAP images (binary): {images_bin}")
            images = {}

            with open(images_bin, 'rb') as f:
                num_images = struct.unpack('Q', f.read(8))[0]

                for _ in range(num_images):
                    img_id = struct.unpack('I', f.read(4))[0]

                    # Quaternion (qw, qx, qy, qz)
                    qvec = struct.unpack('dddd', f.read(32))

                    # Translation (tx, ty, tz)
                    tvec = struct.unpack('ddd', f.read(24))

                    camera_id = struct.unpack('I', f.read(4))[0]

                    # Image name (null-terminated string)
                    name_chars = []
                    while True:
                        char = f.read(1)
                        if char == b'\x00':
                            break
                        name_chars.append(char)
                    name = b''.join(name_chars).decode('utf-8')

                    # Skip points2D data
                    num_points2D = struct.unpack('Q', f.read(8))[0]
                    # Each point2D: x, y (2 doubles) + point3D_id (uint64) = 24 bytes
                    f.read(num_points2D * 24)

                    images[img_id] = {
                        'name': name,
                        'camera_id': camera_id,
                        'qvec': np.array(qvec),
                        'tvec': np.array(tvec)
                    }

            logger.debug(f"Read {len(images)} images from binary")
            return images

        # Fallback to text format
        elif images_txt.exists():
            logger.debug(f"Reading COLMAP images (text): {images_txt}")
            images = {}

            with open(images_txt, 'r') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                i += 1

                if not line or line.startswith('#'):
                    continue

                # Image line
                parts = line.split()
                img_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                name = parts[9]

                # Skip points2D line
                if i < len(lines):
                    i += 1

                images[img_id] = {
                    'name': name,
                    'camera_id': camera_id,
                    'qvec': np.array([qw, qx, qy, qz]),
                    'tvec': np.array([tx, ty, tz])
                }

            logger.debug(f"Read {len(images)} images from text")
            return images

        else:
            raise FileNotFoundError(
                f"COLMAP images file not found. Tried: {images_bin}, {images_txt}"
            )
    
    def _qvec_to_rotmat(self, qvec: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        qw, qx, qy, qz = qvec
        
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return R
    
    def _build_intrinsic_matrix(self, camera: Dict) -> np.ndarray:
        """Build intrinsic matrix from COLMAP camera parameters"""
        model = camera['model']
        params = camera['params']
        width = camera['width']
        height = camera['height']
        
        if model == 'PINHOLE':
            # params: fx, fy, cx, cy
            fx, fy, cx, cy = params
        elif model == 'SIMPLE_PINHOLE':
            # params: f, cx, cy
            f, cx, cy = params
            fx = fy = f
        elif model == 'OPENCV' or model == 'RADIAL':
            # params: fx, fy, cx, cy, k1, k2, p1, p2 [, k3, k4, k5, k6]
            fx, fy, cx, cy = params[:4]
        elif model == 'SIMPLE_RADIAL':
            # params: f, cx, cy, k
            f, cx, cy, k = params
            fx = fy = f
        else:
            logger.warning(f"Unknown camera model: {model}, using default")
            fx = fy = max(width, height)
            cx, cy = width / 2, height / 2
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return K
    
    def extract_dense_point_cloud(
        self,
        sparse_model_dir: str,
        image_dir: str,
        output_dir: str,
        max_image_size: int = 2000
    ) -> str:
        """
        Run dense reconstruction (optional, for reference point cloud).
        
        Args:
            sparse_model_dir: Sparse model directory
            image_dir: Image directory
            output_dir: Output directory for dense reconstruction
            max_image_size: Maximum image size for dense reconstruction
            
        Returns:
            Path to dense point cloud
        """
        logger.info("Running dense reconstruction (this may take a while)...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dense_dir = output_path / 'dense'
        dense_dir.mkdir(exist_ok=True)
        
        # Step 1: Undistort images
        logger.info("Step 1/3: Image undistortion")
        cmd = [
            self.colmap_exe, 'image_undistorter',
            '--image_path', image_dir,
            '--input_path', sparse_model_dir,
            '--output_path', str(dense_dir),
            '--output_type', 'COLMAP',
            '--max_image_size', str(max_image_size)
        ]
        self._run_command(cmd, "Image undistortion failed")
        
        # Step 2: Patch match stereo
        logger.info("Step 2/3: Patch match stereo")
        cmd = [
            self.colmap_exe, 'patch_match_stereo',
            '--workspace_path', str(dense_dir),
            '--workspace_format', 'COLMAP',
            '--PatchMatchStereo.geom_consistency', 'true'
        ]
        self._run_command(cmd, "Patch match stereo failed")
        
        # Step 3: Stereo fusion
        logger.info("Step 3/3: Stereo fusion")
        cmd = [
            self.colmap_exe, 'stereo_fusion',
            '--workspace_path', str(dense_dir),
            '--workspace_format', 'COLMAP',
            '--input_type', 'geometric',
            '--output_path', str(dense_dir / 'fused.ply')
        ]
        self._run_command(cmd, "Stereo fusion failed")
        
        output_ply = dense_dir / 'fused.ply'
        logger.info(f"Dense point cloud saved: {output_ply}")
        
        return str(output_ply)


def run_colmap_sfm_auto(
    image_dir: str,
    output_dir: str,
    poses_json_output: str,
    camera_model: str = 'OPENCV',
    quality: str = 'high',
    dense: bool = False
) -> Dict:
    """
    Automatic COLMAP SFM pipeline.
    
    Args:
        image_dir: Directory with RGB images
        output_dir: Output directory for COLMAP
        poses_json_output: Output path for poses.json
        camera_model: COLMAP camera model
        quality: Quality setting ('low', 'medium', 'high', 'extreme')
        dense: Whether to run dense reconstruction
        
    Returns:
        Dictionary of poses
    """
    runner = COLMAPRunner()
    
    # Run sparse reconstruction
    sparse_model_dir = runner.run_sfm_pipeline(
        image_dir=image_dir,
        output_dir=output_dir,
        camera_model=camera_model,
        quality=quality
    )
    
    # Convert to poses.json
    poses = runner.convert_to_poses_json(
        sparse_model_dir=sparse_model_dir,
        output_json=poses_json_output
    )
    
    # Optional: Dense reconstruction
    if dense:
        runner.extract_dense_point_cloud(
            sparse_model_dir=sparse_model_dir,
            image_dir=image_dir,
            output_dir=output_dir
        )
    
    return poses


if __name__ == '__main__':
    # Test
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Run COLMAP SFM')
    parser.add_argument('--image-dir', required=True, help='Directory with images')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--poses-json', default='poses.json', help='Output poses.json')
    parser.add_argument('--camera-model', default='OPENCV', help='Camera model')
    parser.add_argument('--quality', default='high', choices=['low', 'medium', 'high', 'extreme'])
    parser.add_argument('--dense', action='store_true', help='Run dense reconstruction')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        poses = run_colmap_sfm_auto(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            poses_json_output=args.poses_json,
            camera_model=args.camera_model,
            quality=args.quality,
            dense=args.dense
        )
        
        print(f"\n✅ SFM complete!")
        print(f"   Poses: {args.poses_json}")
        print(f"   Reconstructed {len(poses)} images")
        
    except Exception as e:
        print(f"\n❌ SFM failed: {e}")
        sys.exit(1)
