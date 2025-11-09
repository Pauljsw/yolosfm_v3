"""
Pixel-to-MM Calibration

Calculates pixel size in millimeters for each RGB image using aligned depth.
Considers camera distortion by sampling multiple locations.

Usage:
    python -m src.pixel_calibration \
        --rgb-dir data/rgb \
        --depth-dir data/depth \
        --calib calib/rgb_camera_info.json \
        --output calibration/pixel_scales.json
"""
import numpy as np
import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.interpolate import griddata

from .calib_io import load_camera_info
from .utils import find_rgb_depth_pairs

logger = logging.getLogger(__name__)


def calculate_pixel_scale_map(
    depth_img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray = None,
    num_samples: int = 9
) -> Tuple[np.ndarray, Dict]:
    """
    Calculate pixel-to-mm scale map for an image.

    Samples depth at multiple locations (considering distortion),
    calculates pixel size, and interpolates to full image.

    Args:
        depth_img: Depth image in meters (H, W)
        K: Camera intrinsic matrix (3x3)
        D: Distortion coefficients (optional)
        num_samples: Number of sample points (3x3 grid)

    Returns:
        (scale_map, stats):
            scale_map: (H, W) array of mm/pixel
            stats: Statistics dictionary
    """
    h, w = depth_img.shape
    fx = K[0, 0]
    fy = K[1, 1]

    # Define sampling grid (avoid edges where distortion is extreme)
    margin = int(min(w, h) * 0.1)  # 10% margin
    sample_positions = []

    # 3x3 grid
    for i in range(3):
        for j in range(3):
            u = int(margin + (w - 2*margin) * j / 2)
            v = int(margin + (h - 2*margin) * i / 2)
            sample_positions.append((u, v))

    # Calculate pixel scale at each sample
    scales = []
    valid_positions = []

    for (u, v) in sample_positions:
        d = depth_img[v, u]

        # Valid depth check
        if d < 0.1 or d > 10.0 or np.isnan(d) or np.isinf(d):
            continue

        # Pixel size in mm (using fx for horizontal, could use fy for vertical)
        pixel_mm = (d / fx) * 1000.0

        scales.append(pixel_mm)
        valid_positions.append([u, v])

    if len(scales) == 0:
        logger.warning("No valid depth samples found, using default 1.0 mm/px")
        return np.ones((h, w), dtype=np.float32), {'valid_samples': 0}

    scales = np.array(scales)
    valid_positions = np.array(valid_positions)

    # Interpolate to full image
    grid_u, grid_v = np.meshgrid(np.arange(w), np.arange(h))

    scale_map = griddata(
        valid_positions,
        scales,
        (grid_u, grid_v),
        method='cubic',
        fill_value=np.median(scales)
    )

    # Clip to reasonable range
    scale_map = np.clip(scale_map, 0.1, 10.0)

    # Statistics
    stats = {
        'valid_samples': len(scales),
        'min_scale_mm': float(np.min(scales)),
        'max_scale_mm': float(np.max(scales)),
        'mean_scale_mm': float(np.mean(scales)),
        'median_scale_mm': float(np.median(scales)),
        'std_scale_mm': float(np.std(scales)),
        'sample_positions': valid_positions.tolist(),
        'sample_scales': scales.tolist()
    }

    return scale_map, stats


def run_pixel_calibration(
    rgb_dir: str,
    depth_dir: str,
    calib_path: str,
    output_json: str,
    save_maps: bool = False,
    map_dir: str = None
):
    """
    Run pixel calibration for all RGB-Depth pairs.

    Args:
        rgb_dir: RGB images directory
        depth_dir: Depth images directory (mm unit)
        calib_path: RGB camera calibration JSON
        output_json: Output JSON path for scales
        save_maps: Save individual scale maps as .npy
        map_dir: Directory for scale maps (if save_maps=True)
    """
    logger.info("=" * 80)
    logger.info("Pixel-to-MM Calibration")
    logger.info("=" * 80)

    # Load calibration
    calib = load_camera_info(calib_path)
    logger.info(f"Loaded camera: {calib.width}×{calib.height}, fx={calib.K[0,0]:.1f}")

    # Find RGB-Depth pairs
    pairs = find_rgb_depth_pairs(rgb_dir, depth_dir)
    logger.info(f"Found {len(pairs)} RGB-Depth pairs")

    if len(pairs) == 0:
        raise ValueError("No RGB-Depth pairs found!")

    # Output directory
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_maps:
        map_path = Path(map_dir) if map_dir else output_path.parent / "scale_maps"
        map_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving scale maps to: {map_path}")

    # Process each pair
    calibration_data = {}

    for idx, (rgb_path, depth_path, pair_id) in enumerate(pairs):
        logger.info(f"Processing [{idx+1}/{len(pairs)}]: {pair_id}")

        # Load depth (assume mm unit)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            logger.warning(f"  Failed to load depth: {depth_path}")
            continue

        depth = depth.astype(np.float32) / 1000.0  # mm → meters

        # Calculate scale map
        try:
            scale_map, stats = calculate_pixel_scale_map(
                depth, calib.K, calib.D
            )

            calibration_data[pair_id] = stats

            logger.info(f"  Mean: {stats['mean_scale_mm']:.3f} mm/px, "
                       f"Range: [{stats['min_scale_mm']:.3f}, {stats['max_scale_mm']:.3f}]")

            # Save map
            if save_maps:
                map_file = map_path / f"{pair_id}.npy"
                np.save(map_file, scale_map)

        except Exception as e:
            logger.error(f"  Failed to calibrate {pair_id}: {e}")
            continue

    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"Calibration complete! Saved to: {output_path}")
    logger.info(f"Processed: {len(calibration_data)}/{len(pairs)} images")
    logger.info("=" * 80)

    return calibration_data


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Pixel-to-MM calibration')
    parser.add_argument('--rgb-dir', required=True, help='RGB images directory')
    parser.add_argument('--depth-dir', required=True, help='Depth images directory')
    parser.add_argument('--calib', required=True, help='RGB camera calibration JSON')
    parser.add_argument('--output', default='calibration/pixel_scales.json',
                       help='Output JSON path')
    parser.add_argument('--save-maps', action='store_true',
                       help='Save individual scale maps as .npy')
    parser.add_argument('--map-dir', default=None,
                       help='Directory for scale maps (default: same as output)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        run_pixel_calibration(
            args.rgb_dir,
            args.depth_dir,
            args.calib,
            args.output,
            args.save_maps,
            args.map_dir
        )

        print("\n✅ Pixel calibration complete!")

    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
