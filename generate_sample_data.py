"""
Generate Sample Data for Testing
Creates synthetic RGB images, depth maps, YOLO masks, and SFM poses.
"""
import numpy as np
import json
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(output_dir: str = 'data', num_images: int = 5):
    """
    Generate synthetic test data.
    
    Args:
        output_dir: Output directory for generated data
        num_images: Number of images to generate
    """
    output_path = Path(output_dir)
    
    # Create directories
    rgb_dir = output_path / 'rgb'
    depth_dir = output_path / 'depth'
    masks_dir = output_path / 'yolo_masks'
    sfm_dir = output_path / 'sfm'
    
    for d in [rgb_dir, depth_dir, masks_dir, sfm_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {num_images} sample images...")
    
    # Camera parameters
    rgb_width, rgb_height = 3840, 2160
    depth_width, depth_height = 512, 512
    
    # Generate poses (circular trajectory around origin)
    poses = {}
    radius = 3.0  # meters
    
    for i in range(num_images):
        image_id = f"{i:06d}"
        
        # Generate RGB image (simple gradient with pattern)
        rgb_img = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
        
        # Create gradient
        for y in range(rgb_height):
            rgb_img[y, :, 0] = int(255 * y / rgb_height)  # Red gradient
        for x in range(rgb_width):
            rgb_img[:, x, 1] = int(255 * x / rgb_width)   # Green gradient
        rgb_img[:, :, 2] = 100  # Constant blue
        
        # Add some texture (simulated cracks)
        for _ in range(5):
            # Random crack line
            pt1 = (np.random.randint(0, rgb_width), np.random.randint(0, rgb_height))
            pt2 = (np.random.randint(0, rgb_width), np.random.randint(0, rgb_height))
            cv2.line(rgb_img, pt1, pt2, (0, 0, 0), thickness=np.random.randint(5, 15))
        
        # Save RGB
        rgb_path = rgb_dir / f"{image_id}.png"
        cv2.imwrite(str(rgb_path), rgb_img)
        
        # Generate depth image (planar surface with noise)
        depth_img = np.ones((depth_height, depth_width), dtype=np.float32) * 2.0  # 2 meters
        
        # Add some variation (simulate uneven surface)
        y_coords, x_coords = np.ogrid[:depth_height, :depth_width]
        depth_img += 0.1 * np.sin(x_coords / 50) * np.cos(y_coords / 50)
        
        # Add noise
        depth_img += np.random.randn(depth_height, depth_width) * 0.01
        
        # Add some holes (invalid depth)
        num_holes = 10
        for _ in range(num_holes):
            cx, cy = np.random.randint(50, depth_width-50), np.random.randint(50, depth_height-50)
            r = np.random.randint(5, 20)
            cv2.circle(depth_img, (cx, cy), r, 0, -1)
        
        # Save depth (as uint16 in mm)
        depth_mm = (depth_img * 1000).astype(np.uint16)
        depth_path = depth_dir / f"{image_id}.png"
        cv2.imwrite(str(depth_path), depth_mm)
        
        # Generate YOLO masks
        masks = []
        
        # Generate 2-3 crack masks per image
        num_cracks = np.random.randint(2, 4)
        for j in range(num_cracks):
            # Random crack polygon (linear)
            x1, y1 = np.random.randint(500, rgb_width-500), np.random.randint(500, rgb_height-500)
            x2, y2 = x1 + np.random.randint(-800, 800), y1 + np.random.randint(-400, 400)
            
            # Create polygon around line (width ~20-40 pixels)
            width = np.random.randint(20, 40)
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            dx, dy = dx / length, dy / length
            
            # Perpendicular direction
            px, py = -dy * width, dx * width
            
            polygon = [
                [float(x1 + px), float(y1 + py)],
                [float(x2 + px), float(y2 + py)],
                [float(x2 - px), float(y2 - py)],
                [float(x1 - px), float(y1 - py)]
            ]
            
            # Clip to image bounds
            polygon = [[max(0, min(rgb_width-1, x)), max(0, min(rgb_height-1, y))] 
                      for x, y in polygon]
            
            mask = {
                'class': 'crack',
                'score': float(np.random.uniform(0.7, 0.95)),
                'polygon': polygon,
                'instance_id': f'i_{image_id}_{j}'
            }
            masks.append(mask)
        
        # Generate 1-2 spalling masks
        num_spalls = np.random.randint(1, 3)
        for j in range(num_spalls):
            # Random circular/elliptical spall
            cx, cy = np.random.randint(500, rgb_width-500), np.random.randint(500, rgb_height-500)
            rx, ry = np.random.randint(100, 300), np.random.randint(100, 300)
            
            # Create ellipse polygon
            num_pts = 20
            angles = np.linspace(0, 2*np.pi, num_pts)
            polygon = [[float(cx + rx*np.cos(a)), float(cy + ry*np.sin(a))] 
                      for a in angles]
            
            mask = {
                'class': 'spalling',
                'score': float(np.random.uniform(0.75, 0.95)),
                'polygon': polygon,
                'instance_id': f'i_{image_id}_s{j}'
            }
            masks.append(mask)
        
        # Save masks
        masks_data = {
            'image_id': image_id,
            'masks': masks
        }
        masks_path = masks_dir / f"{image_id}.json"
        with open(masks_path, 'w') as f:
            json.dump(masks_data, f, indent=2)
        
        # Generate pose (circular trajectory)
        angle = i * (2 * np.pi / num_images)
        
        # Camera position
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = 0.5  # Slightly above center
        
        # Look at origin
        target = np.array([0, 0, 0])
        position = np.array([cam_x, cam_y, cam_z])
        up = np.array([0, 0, 1])
        
        # Compute rotation matrix (world to camera)
        z_axis = target - position
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        R_world_to_cam = np.column_stack([x_axis, y_axis, z_axis])
        
        # For OpenCV convention: camera to world
        R = R_world_to_cam.T
        t = -R @ position
        
        poses[f"{image_id}.png"] = {
            'filename': f"{image_id}.png",
            'R': R.tolist(),
            't': t.tolist(),
            'K': [
                [2800.0, 0.0, 1920.0],
                [0.0, 2800.0, 1080.0],
                [0.0, 0.0, 1.0]
            ]
        }
        
        logger.info(f"Generated image {image_id}: {len(masks)} masks")
    
    # Save poses
    poses_path = sfm_dir / 'poses.json'
    with open(poses_path, 'w') as f:
        json.dump(poses, f, indent=2)
    
    logger.info(f"Sample data generated successfully in {output_dir}")
    logger.info(f"  RGB images: {num_images}")
    logger.info(f"  Depth maps: {num_images}")
    logger.info(f"  YOLO masks: {num_images}")
    logger.info(f"  Poses: {num_images}")
    
    return output_path


def generate_sample_calibration(output_dir: str = 'calib'):
    """Generate sample camera calibration files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # RGB camera (3840x2160)
    rgb_calib = {
        'width': 3840,
        'height': 2160,
        'K': [
            [2800.0, 0.0, 1920.0],
            [0.0, 2800.0, 1080.0],
            [0.0, 0.0, 1.0]
        ],
        'D': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'distortion_model': 'rational_polynomial'
    }
    
    # Depth camera (512x512)
    depth_calib = {
        'width': 512,
        'height': 512,
        'K': [
            [365.0, 0.0, 256.0],
            [0.0, 365.0, 256.0],
            [0.0, 0.0, 1.0]
        ],
        'D': [0.0, 0.0, 0.0, 0.0, 0.0],
        'distortion_model': 'radial_tangential'
    }
    
    with open(output_path / 'rgb_camera_info.json', 'w') as f:
        json.dump(rgb_calib, f, indent=2)
    
    with open(output_path / 'depth_camera_info.json', 'w') as f:
        json.dump(depth_calib, f, indent=2)
    
    logger.info(f"Sample calibration files created in {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample data for testing')
    parser.add_argument('--num-images', type=int, default=5, help='Number of images to generate')
    parser.add_argument('--data-dir', type=str, default='data', help='Output data directory')
    parser.add_argument('--calib-dir', type=str, default='calib', help='Output calibration directory')
    
    args = parser.parse_args()
    
    # Generate calibration
    generate_sample_calibration(args.calib_dir)
    
    # Generate data
    generate_sample_data(args.data_dir, args.num_images)
    
    print("\nSample data generation complete!")
    print(f"Data directory: {args.data_dir}")
    print(f"Calibration directory: {args.calib_dir}")
    print("\nRun pipeline with:")
    print("  python -m src.pipeline full --config configs/default.yaml")
