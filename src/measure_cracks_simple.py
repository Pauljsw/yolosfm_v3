"""
Simple Crack Measurement (2D + Pixel-to-MM)

Measures crack length and width in each image using:
1. 2D skeletonization and MST for length
2. Perpendicular scanning for width
3. Pixel-to-MM calibration for absolute scale

Multi-view aggregation using median.

Usage:
    python -m src.measure_cracks_simple \
        --masks-dir data/yolo_masks \
        --pixel-scales calibration/pixel_scales.json \
        --output outputs/measurements.csv
"""
import numpy as np
import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import csv

from skimage.morphology import skeletonize
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

logger = logging.getLogger(__name__)


def rasterize_polygon(polygon: List[List[float]], img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Rasterize polygon to binary mask.

    Args:
        polygon: List of (x, y) points
        img_shape: (height, width)

    Returns:
        Binary mask (H, W)
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Preprocess mask for measurement (paper methodology).

    Args:
        mask: Binary mask (uint8)

    Returns:
        Preprocessed mask
    """
    # Median blur
    mask = cv2.medianBlur(mask, ksize=3)

    # Bilateral filter
    mask = cv2.bilateralFilter(mask, d=5, sigmaColor=75, sigmaSpace=75)

    # Gamma correction
    gamma = 0.7
    mask = np.power(mask / 255.0, gamma) * 255
    mask = mask.astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mask = clahe.apply(mask)

    # Adaptive threshold
    mask = cv2.adaptiveThreshold(
        mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )

    return mask


def measure_skeleton_length_mst(skeleton: np.ndarray) -> float:
    """
    Measure skeleton length using MST.

    Args:
        skeleton: Binary skeleton (uint8)

    Returns:
        Total length in pixels
    """
    # Get skeleton points
    coords = np.column_stack(np.where(skeleton > 0))

    if len(coords) < 2:
        return 0.0

    # Compute distance matrix
    dist = distance_matrix(coords, coords)

    # MST
    mst = minimum_spanning_tree(dist)

    # Total length
    total_length = mst.sum()

    return total_length


def measure_perpendicular_width(
    skeleton: np.ndarray,
    mask: np.ndarray,
    num_samples: int = 10
) -> float:
    """
    Measure average crack width using perpendicular scanning.

    Args:
        skeleton: Binary skeleton
        mask: Binary mask
        num_samples: Number of points to sample

    Returns:
        Average width in pixels
    """
    coords = np.column_stack(np.where(skeleton > 0))

    if len(coords) < num_samples:
        num_samples = len(coords)

    if num_samples == 0:
        return 0.0

    # Sample points along skeleton
    indices = np.linspace(0, len(coords)-1, num_samples, dtype=int)
    sample_coords = coords[indices]

    widths = []

    for y, x in sample_coords:
        # Simple perpendicular scan (horizontal for simplicity)
        # More advanced: estimate local tangent direction

        # Scan left
        left_dist = 0
        for dx in range(1, 50):
            if x - dx < 0 or mask[y, x-dx] == 0:
                left_dist = dx - 1
                break

        # Scan right
        right_dist = 0
        for dx in range(1, 50):
            if x + dx >= mask.shape[1] or mask[y, x+dx] == 0:
                right_dist = dx - 1
                break

        width = left_dist + right_dist
        if width > 0:
            widths.append(width)

    if len(widths) == 0:
        return 0.0

    return np.median(widths)


def measure_crack_in_image(
    mask_json: Dict,
    pixel_scale: float,
    img_shape: Tuple[int, int] = (2160, 3840)
) -> List[Dict]:
    """
    Measure all cracks in one image.

    Args:
        mask_json: YOLO mask JSON
        pixel_scale: Pixel-to-mm scale (mm/pixel)
        img_shape: Image shape (H, W)

    Returns:
        List of measurements
    """
    if 'masks' not in mask_json:
        return []

    measurements = []

    for idx, mask_data in enumerate(mask_json['masks']):
        if mask_data['class'] != 'crack':
            continue

        polygon = mask_data['polygon']
        confidence = mask_data.get('score', 1.0)

        # Rasterize
        mask = rasterize_polygon(polygon, img_shape)

        if mask.sum() == 0:
            continue

        # Preprocess
        mask_processed = preprocess_mask(mask)

        # Skeletonize
        skeleton = skeletonize(mask_processed > 0).astype(np.uint8) * 255

        # Measure length
        length_px = measure_skeleton_length_mst(skeleton)

        # Measure width
        width_px = measure_perpendicular_width(skeleton, mask_processed)

        # Convert to mm
        length_mm = length_px * pixel_scale
        width_mm = width_px * pixel_scale

        measurements.append({
            'mask_idx': idx,
            'length_px': length_px,
            'width_px': width_px,
            'length_mm': length_mm,
            'width_mm': width_mm,
            'confidence': confidence,
            'pixel_scale_mm': pixel_scale
        })

    return measurements


def run_measurement(
    masks_dir: str,
    pixel_scales_json: str,
    output_csv: str,
    img_shape: Tuple[int, int] = (2160, 3840),
    min_length_mm: float = 10.0
):
    """
    Run measurement for all images.

    Args:
        masks_dir: YOLO masks directory
        pixel_scales_json: Pixel scales JSON
        output_csv: Output CSV path
        img_shape: Image shape (H, W)
        min_length_mm: Minimum crack length in mm
    """
    logger.info("=" * 80)
    logger.info("Crack Measurement (2D + Pixel-to-MM)")
    logger.info("=" * 80)

    # Load pixel scales
    with open(pixel_scales_json, 'r') as f:
        pixel_scales = json.load(f)

    logger.info(f"Loaded pixel scales for {len(pixel_scales)} images")

    # Load masks
    masks_path = Path(masks_dir)
    all_measurements = []
    image_count = 0

    for mask_file in sorted(masks_path.glob("*.json")):
        image_id = mask_file.stem

        # Load mask
        with open(mask_file, 'r') as f:
            mask_json = json.load(f)

        # Get pixel scale (use mean if available)
        if image_id in pixel_scales:
            pixel_scale = pixel_scales[image_id].get('mean_scale_mm', 1.0)
        else:
            logger.warning(f"No pixel scale for {image_id}, using default 1.0 mm/px")
            pixel_scale = 1.0

        # Measure
        measurements = measure_crack_in_image(mask_json, pixel_scale, img_shape)

        for meas in measurements:
            meas['image_id'] = image_id
            all_measurements.append(meas)

        if len(measurements) > 0:
            image_count += 1
            logger.info(f"  {image_id}: {len(measurements)} cracks")

    logger.info(f"Total images processed: {image_count}")
    logger.info(f"Total cracks detected: {len(all_measurements)}")

    # Filter by minimum length
    filtered = [m for m in all_measurements if m['length_mm'] >= min_length_mm]
    logger.info(f"After min length filter ({min_length_mm}mm): {len(filtered)} cracks")

    # Save CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'mask_idx', 'length_mm', 'width_mm',
                     'length_px', 'width_px', 'confidence', 'pixel_scale_mm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for meas in filtered:
            writer.writerow(meas)

    logger.info("=" * 80)
    logger.info(f"Measurement complete! Saved to: {output_csv}")
    logger.info("=" * 80)

    return filtered


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Simple crack measurement')
    parser.add_argument('--masks-dir', required=True, help='YOLO masks directory')
    parser.add_argument('--pixel-scales', required=True,
                       help='Pixel scales JSON from calibration')
    parser.add_argument('--output', default='outputs/measurements.csv',
                       help='Output CSV path')
    parser.add_argument('--img-shape', type=int, nargs=2, default=[2160, 3840],
                       help='Image shape (H W), default: 2160 3840')
    parser.add_argument('--min-length-mm', type=float, default=10.0,
                       help='Minimum crack length in mm (default: 10.0)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        measurements = run_measurement(
            args.masks_dir,
            args.pixel_scales,
            args.output,
            tuple(args.img_shape),
            args.min_length_mm
        )

        print(f"\n✅ Measurement complete!")
        print(f"   Total cracks: {len(measurements)}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Measurement failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
