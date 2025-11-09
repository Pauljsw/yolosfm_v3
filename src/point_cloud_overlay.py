"""
Point Cloud Mask Overlay

Overlays YOLO crack masks onto SFM sparse point cloud.
Uses COLMAP track information to map 2D pixels to 3D points.

Usage:
    python -m src.point_cloud_overlay \
        --sparse-dir data/sfm/sparse/0 \
        --masks-dir data/yolo_masks \
        --output outputs/sfm_masked_cloud.ply
"""
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from .colmap_io import read_points3D_binary, read_images_binary, read_cameras_binary

logger = logging.getLogger(__name__)


def load_yolo_mask(mask_path: Path) -> Dict:
    """Load YOLO mask JSON"""
    with open(mask_path, 'r') as f:
        return json.load(f)


def is_pixel_in_crack_mask(mask_json: Dict, pixel_xy: Tuple[float, float]) -> bool:
    """
    Check if pixel is inside any crack polygon.

    Args:
        mask_json: YOLO mask JSON data
        pixel_xy: Pixel coordinates (u, v)

    Returns:
        True if pixel is in crack mask
    """
    if 'masks' not in mask_json:
        return False

    point = ShapelyPoint(pixel_xy)

    for mask in mask_json['masks']:
        if mask['class'] == 'crack':
            polygon_coords = mask['polygon']
            if len(polygon_coords) < 3:
                continue

            try:
                poly = Polygon(polygon_coords)
                if poly.is_valid and poly.contains(point):
                    return True
            except Exception as e:
                logger.debug(f"Invalid polygon: {e}")
                continue

    return False


def overlay_masks_on_pointcloud(
    sparse_dir: str,
    masks_dir: str,
    output_ply: str,
    crack_color: Tuple[int, int, int] = (255, 0, 0),
    min_track_length: int = 2
):
    """
    Overlay YOLO masks on SFM point cloud.

    Args:
        sparse_dir: COLMAP sparse/0 directory
        masks_dir: YOLO masks directory
        output_ply: Output PLY path
        crack_color: RGB color for crack points (default: red)
        min_track_length: Minimum track length to include point
    """
    logger.info("=" * 80)
    logger.info("Point Cloud Mask Overlay")
    logger.info("=" * 80)

    sparse_path = Path(sparse_dir)
    masks_path = Path(masks_dir)

    # Read COLMAP data
    logger.info(f"Reading COLMAP data from: {sparse_dir}")
    points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))
    images = read_images_binary(str(sparse_path / "images.bin"))
    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))

    logger.info(f"  3D Points: {len(points3D)}")
    logger.info(f"  Images: {len(images)}")
    logger.info(f"  Cameras: {len(cameras)}")

    # Build image name to ID mapping
    image_name_to_id = {img.name: img_id for img_id, img in images.items()}

    # Load all masks
    logger.info(f"Loading YOLO masks from: {masks_dir}")
    masks = {}
    for mask_file in masks_path.glob("*.json"):
        try:
            mask_data = load_yolo_mask(mask_file)
            # Match image name (without extension)
            img_stem = mask_file.stem
            masks[img_stem] = mask_data
        except Exception as e:
            logger.warning(f"Failed to load mask {mask_file}: {e}")

    logger.info(f"  Loaded {len(masks)} masks")

    # Process each 3D point
    logger.info("Processing 3D points...")

    xyz_list = []
    rgb_list = []
    crack_count = 0
    skipped_count = 0

    for point_id, point in points3D.items():
        xyz = point.xyz
        original_rgb = point.rgb

        # Skip points with short tracks (likely noise)
        if len(point.image_ids) < min_track_length:
            skipped_count += 1
            continue

        # Check if any observation is in crack mask
        is_crack = False

        for img_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
            if img_id not in images:
                continue

            image = images[img_id]
            image_name = image.name
            image_stem = Path(image_name).stem

            # Get pixel coordinates
            if point2D_idx >= len(image.xys):
                continue

            pixel_xy = image.xys[point2D_idx]

            # Check mask
            if image_stem in masks:
                if is_pixel_in_crack_mask(masks[image_stem], pixel_xy):
                    is_crack = True
                    break

        # Assign color
        if is_crack:
            rgb = crack_color
            crack_count += 1
        else:
            rgb = original_rgb

        xyz_list.append(xyz)
        rgb_list.append(rgb)

    logger.info(f"  Total points: {len(xyz_list)}")
    logger.info(f"  Crack points: {crack_count} ({crack_count/len(xyz_list)*100:.1f}%)")
    logger.info(f"  Skipped (short track): {skipped_count}")

    # Save PLY
    logger.info(f"Saving point cloud to: {output_ply}")
    save_ply(output_ply, xyz_list, rgb_list)

    logger.info("=" * 80)
    logger.info("Point cloud overlay complete!")
    logger.info("=" * 80)

    return {
        'total_points': len(xyz_list),
        'crack_points': crack_count,
        'skipped_points': skipped_count
    }


def save_ply(filename: str, xyz: List[np.ndarray], rgb: List[np.ndarray]):
    """
    Save point cloud as binary PLY.

    Args:
        filename: Output PLY path
        xyz: List of 3D coordinates
        rgb: List of RGB colors (uint8)
    """
    xyz = np.array(xyz)
    rgb = np.array(rgb, dtype=np.uint8)

    assert xyz.shape[0] == rgb.shape[0]
    assert xyz.shape[1] == 3
    assert rgb.shape[1] == 3

    # Create output directory
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write binary PLY
    with open(filename, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(xyz)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))

        # Vertices
        for i in range(len(xyz)):
            # xyz as float32
            f.write(xyz[i].astype(np.float32).tobytes())
            # rgb as uint8
            f.write(rgb[i].tobytes())

    logger.info(f"Saved {len(xyz)} points to {filename}")


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Overlay YOLO masks on SFM point cloud')
    parser.add_argument('--sparse-dir', required=True,
                       help='COLMAP sparse directory (e.g., data/sfm/sparse/0)')
    parser.add_argument('--masks-dir', required=True,
                       help='YOLO masks directory')
    parser.add_argument('--output', required=True,
                       help='Output PLY path')
    parser.add_argument('--crack-color', type=int, nargs=3, default=[255, 0, 0],
                       help='RGB color for crack points (default: 255 0 0)')
    parser.add_argument('--min-track-length', type=int, default=2,
                       help='Minimum track length (default: 2)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        result = overlay_masks_on_pointcloud(
            args.sparse_dir,
            args.masks_dir,
            args.output,
            tuple(args.crack_color),
            args.min_track_length
        )

        print(f"\n✅ Point cloud overlay complete!")
        print(f"   Total points: {result['total_points']}")
        print(f"   Crack points: {result['crack_points']}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Overlay failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
