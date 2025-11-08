"""
Convert COLMAP SFM output to pipeline format
Converts COLMAP's cameras.txt, images.txt to poses.json
"""
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict


def read_colmap_cameras(cameras_file):
    """
    Read COLMAP cameras.txt
    
    Format:
    # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    """
    cameras = {}
    
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            
            # Convert to K matrix
            if model == 'PINHOLE':
                # PINHOLE: fx, fy, cx, cy
                fx, fy, cx, cy = params
                K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                D = [0, 0, 0, 0, 0, 0, 0, 0]  # No distortion
            
            elif model == 'RADIAL' or model == 'SIMPLE_RADIAL':
                # SIMPLE_RADIAL: f, cx, cy, k1
                # RADIAL: f, cx, cy, k1, k2
                if len(params) == 4:
                    f, cx, cy, k1 = params
                    D = [k1, 0, 0, 0, 0]
                else:
                    f, cx, cy, k1, k2 = params
                    D = [k1, k2, 0, 0, 0]
                
                K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
            
            elif model == 'OPENCV':
                # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
                fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
                K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                D = [k1, k2, p1, p2, 0]  # Standard OpenCV
            
            else:
                print(f"Warning: Unsupported camera model {model}, using PINHOLE approximation")
                fx, fy, cx, cy = params[:4]
                K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                D = [0, 0, 0, 0, 0, 0, 0, 0]
            
            cameras[camera_id] = {
                'width': width,
                'height': height,
                'K': K,
                'D': D,
                'model': model
            }
    
    return cameras


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix"""
    # Normalize
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def read_colmap_images(images_file, cameras):
    """
    Read COLMAP images.txt
    
    Format:
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    # POINTS2D[] as (X, Y, POINT3D_ID)
    """
    poses = {}
    
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        if not line or line.startswith('#'):
            continue
        
        # Image line
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]
        
        # Skip points2D line
        if i < len(lines):
            i += 1
        
        # Convert COLMAP convention to OpenCV convention
        # COLMAP: world-to-camera transformation
        # We need: camera-to-world (inverse)
        
        R_colmap = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t_colmap = np.array([tx, ty, tz])
        
        # Camera-to-world
        R = R_colmap.T
        t = -R @ t_colmap
        
        poses[image_name] = {
            'filename': image_name,
            'R': R.tolist(),
            't': t.tolist(),
            'K': cameras[camera_id]['K'],
            'camera_id': camera_id
        }
    
    return poses


def convert_colmap_to_poses(
    colmap_dir: str,
    output_file: str = 'poses.json',
    cameras_file: str = 'cameras.txt',
    images_file: str = 'images.txt'
):
    """
    Convert COLMAP output to poses.json format
    
    Args:
        colmap_dir: Directory containing COLMAP output
        output_file: Output JSON file name
        cameras_file: Name of cameras file
        images_file: Name of images file
    """
    colmap_path = Path(colmap_dir)
    
    # Read cameras
    cameras_path = colmap_path / cameras_file
    if not cameras_path.exists():
        raise FileNotFoundError(f"Cameras file not found: {cameras_path}")
    
    print(f"Reading cameras from {cameras_path}")
    cameras = read_colmap_cameras(cameras_path)
    print(f"Found {len(cameras)} cameras")
    
    # Read images
    images_path = colmap_path / images_file
    if not images_path.exists():
        raise FileNotFoundError(f"Images file not found: {images_path}")
    
    print(f"Reading images from {images_path}")
    poses = read_colmap_images(images_path, cameras)
    print(f"Found {len(poses)} images")
    
    # Save poses
    output_path = colmap_path / output_file
    with open(output_path, 'w') as f:
        json.dump(poses, f, indent=2)
    
    print(f"Saved poses to {output_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Images: {len(poses)}")
    print(f"  Cameras: {len(cameras)}")
    print(f"\nCamera details:")
    for cam_id, cam_data in cameras.items():
        print(f"  Camera {cam_id}: {cam_data['width']}x{cam_data['height']}, "
              f"model={cam_data['model']}")
    
    return poses


def extract_point_cloud(
    colmap_dir: str,
    output_file: str = 'points3D.ply',
    points_file: str = 'points3D.txt'
):
    """
    Extract point cloud from COLMAP and save as PLY
    
    Args:
        colmap_dir: Directory containing COLMAP output
        output_file: Output PLY file name
        points_file: Name of points3D file
    """
    colmap_path = Path(colmap_dir)
    points_path = colmap_path / points_file
    
    if not points_path.exists():
        print(f"Warning: Points file not found: {points_path}")
        return
    
    print(f"Reading 3D points from {points_path}")
    
    points = []
    colors = []
    
    with open(points_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            point_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            
            points.append([x, y, z])
            colors.append([r, g, b])
    
    points = np.array(points)
    colors = np.array(colors)
    
    print(f"Found {len(points)} 3D points")
    
    # Save as PLY
    output_path = colmap_path / output_file
    with open(output_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f'{x} {y} {z} {r} {g} {b}\n')
    
    print(f"Saved point cloud to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert COLMAP SFM output to pipeline format'
    )
    parser.add_argument(
        'colmap_dir',
        type=str,
        help='Directory containing COLMAP output (cameras.txt, images.txt, points3D.txt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='poses.json',
        help='Output poses file name (default: poses.json)'
    )
    parser.add_argument(
        '--cameras',
        type=str,
        default='cameras.txt',
        help='Cameras file name (default: cameras.txt)'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='images.txt',
        help='Images file name (default: images.txt)'
    )
    parser.add_argument(
        '--extract-points',
        action='store_true',
        help='Also extract 3D point cloud as PLY'
    )
    parser.add_argument(
        '--points',
        type=str,
        default='points3D.txt',
        help='Points3D file name (default: points3D.txt)'
    )
    
    args = parser.parse_args()
    
    try:
        # Convert poses
        poses = convert_colmap_to_poses(
            args.colmap_dir,
            args.output,
            args.cameras,
            args.images
        )
        
        # Extract point cloud if requested
        if args.extract_points:
            extract_point_cloud(
                args.colmap_dir,
                'A_cloud.ply',
                args.points
            )
        
        print("\n✅ Conversion complete!")
        print(f"\nNext steps:")
        print(f"1. Copy {args.output} to your pipeline's data/sfm/ directory")
        print(f"2. Run the pipeline: python -m src.pipeline full --config configs/default.yaml")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
