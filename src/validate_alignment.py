"""
Alignment Validation Tool

Validates the quality of RGB-Depth alignment by:
1. Visual validation: RGB + Depth overlay, edge comparison
2. Quantitative validation: Reprojection error measurement

Usage:
    python -m src.validate_alignment --rgb-dir data/rgb --depth-dir data/depth \
                                     --aligned-dir outputs/aligned_depth \
                                     --calib-rgb calib/rgb_camera_info.json \
                                     --calib-depth calib/depth_camera_info.json \
                                     --num-samples 5
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import argparse

logger = logging.getLogger(__name__)


def load_image_pair(rgb_path: str, depth_orig_path: str, aligned_depth_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load RGB, original depth, and aligned depth images"""
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(f"RGB not found: {rgb_path}")

    depth_orig = cv2.imread(depth_orig_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if depth_orig is None:
        raise FileNotFoundError(f"Original depth not found: {depth_orig_path}")

    aligned_depth = cv2.imread(aligned_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if aligned_depth is None:
        raise FileNotFoundError(f"Aligned depth not found: {aligned_depth_path}")

    # Convert depth from mm to meters if needed
    if depth_orig.max() > 100:
        depth_orig = depth_orig / 1000.0
    if aligned_depth.max() > 100:
        aligned_depth = aligned_depth / 1000.0

    return rgb, depth_orig, aligned_depth


def visual_validation(rgb: np.ndarray, aligned_depth: np.ndarray, output_path: str = None) -> Dict:
    """
    Visual validation: RGB + Depth overlay and edge comparison

    Returns:
        Dictionary with validation metrics
    """
    # 1. Create depth heatmap
    depth_valid = aligned_depth > 0
    if depth_valid.sum() == 0:
        logger.warning("No valid depth for visualization")
        return {'edge_overlap': 0.0, 'coverage': 0.0}

    depth_normalized = np.zeros_like(aligned_depth)
    depth_normalized[depth_valid] = (aligned_depth[depth_valid] - aligned_depth[depth_valid].min()) / \
                                     (aligned_depth[depth_valid].max() - aligned_depth[depth_valid].min())

    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_colored[~depth_valid] = 0  # Black for invalid

    # 2. Overlay
    overlay = cv2.addWeighted(rgb, 0.6, depth_colored, 0.4, 0)

    # 3. Edge comparison
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rgb_edges = cv2.Canny(rgb_gray, 50, 150)

    depth_8u = (depth_normalized * 255).astype(np.uint8)
    depth_edges = cv2.Canny(depth_8u, 50, 150)

    # Edge overlap
    overlap = cv2.bitwise_and(rgb_edges, depth_edges)
    edge_overlap_ratio = overlap.sum() / max(depth_edges.sum(), 1)

    # Coverage
    coverage = depth_valid.sum() / aligned_depth.size

    # 4. Visualize
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')

    im = axes[0, 1].imshow(aligned_depth, cmap='jet')
    axes[0, 1].set_title(f'Aligned Depth (Coverage: {coverage:.1%})')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], label='Depth (m)')

    axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('RGB + Depth Overlay')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(rgb_edges, cmap='gray')
    axes[1, 0].set_title('RGB Edges')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(depth_edges, cmap='gray')
    axes[1, 1].set_title('Depth Edges')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(overlap, cmap='gray')
    axes[1, 2].set_title(f'Edge Overlap: {edge_overlap_ratio:.1%}')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visual validation: {output_path}")
    else:
        plt.show()

    plt.close()

    return {
        'edge_overlap': float(edge_overlap_ratio),
        'coverage': float(coverage)
    }


def backproject_depth(u: np.ndarray, v: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Backproject depth pixels to 3D camera coordinates"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return np.stack([x, y, z], axis=-1)


def project_3d_to_image(points_3d: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to image coordinates"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    z = points_3d[:, 2]
    valid = z > 0

    u = np.zeros(len(points_3d))
    v = np.zeros(len(points_3d))

    u[valid] = (points_3d[valid, 0] * fx / z[valid]) + cx
    v[valid] = (points_3d[valid, 1] * fy / z[valid]) + cy

    return np.stack([u, v], axis=-1), valid


def quantitative_validation(
    depth_orig: np.ndarray,
    aligned_depth: np.ndarray,
    depth_K: np.ndarray,
    rgb_K: np.ndarray,
    num_samples: int = 1000
) -> Dict:
    """
    Quantitative validation: Reprojection error measurement

    Compares simple resize alignment vs full geometric alignment

    Returns:
        Dictionary with error statistics
    """
    h_depth, w_depth = depth_orig.shape
    h_rgb, w_rgb = aligned_depth.shape

    # Sample valid depth pixels
    y_d, x_d = np.where(depth_orig > 0)

    if len(x_d) == 0:
        logger.warning("No valid depth pixels for validation")
        return {'mean_error': 0, 'median_error': 0, 'max_error': 0}

    # Random sample
    n_samples = min(num_samples, len(x_d))
    indices = np.random.choice(len(x_d), n_samples, replace=False)
    x_d_sample = x_d[indices].astype(np.float32)
    y_d_sample = y_d[indices].astype(np.float32)
    depth_sample = depth_orig[y_d[indices], x_d[indices]]

    # Method 1: Simple resize (what we're using)
    scale_x = w_rgb / w_depth
    scale_y = h_rgb / h_depth
    x_rgb_simple = x_d_sample * scale_x
    y_rgb_simple = y_d_sample * scale_y

    # Method 2: Full geometric (ground truth)
    points_3d = backproject_depth(x_d_sample, y_d_sample, depth_sample, depth_K)
    rgb_coords, valid = project_3d_to_image(points_3d, rgb_K)
    x_rgb_full = rgb_coords[:, 0]
    y_rgb_full = rgb_coords[:, 1]

    # Compute reprojection error (only for valid projections)
    errors = np.sqrt((x_rgb_simple[valid] - x_rgb_full[valid])**2 +
                     (y_rgb_simple[valid] - y_rgb_full[valid])**2)

    if len(errors) == 0:
        logger.warning("No valid projections for error calculation")
        return {'mean_error': 0, 'median_error': 0, 'max_error': 0, 'valid_ratio': 0}

    # Statistics
    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))
    std_error = float(np.std(errors))
    max_error = float(np.max(errors))
    p95_error = float(np.percentile(errors, 95))
    valid_ratio = valid.sum() / len(valid)

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(mean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}px')
    plt.axvline(median_error, color='g', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}px')
    plt.xlabel('Reprojection Error (pixels)')
    plt.ylabel('Count')
    plt.title(f'Reprojection Error Distribution (n={len(errors)})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = 'outputs/alignment_validation/reprojection_error_histogram.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved error histogram: {output_path}")
    plt.close()

    return {
        'mean_error': mean_error,
        'median_error': median_error,
        'std_error': std_error,
        'max_error': max_error,
        'p95_error': p95_error,
        'valid_ratio': float(valid_ratio),
        'num_samples': len(errors)
    }


def validate_alignment(
    rgb_dir: str,
    depth_dir: str,
    aligned_dir: str,
    calib_rgb_path: str,
    calib_depth_path: str,
    num_samples: int = 5,
    output_dir: str = 'outputs/alignment_validation'
):
    """
    Complete alignment validation pipeline

    Args:
        rgb_dir: RGB images directory
        depth_dir: Original depth images directory
        aligned_dir: Aligned depth images directory
        calib_rgb_path: RGB camera calibration JSON
        calib_depth_path: Depth camera calibration JSON
        num_samples: Number of image pairs to validate
        output_dir: Output directory for validation results
    """
    from .calib_io import load_camera_info

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load calibrations
    rgb_calib = load_camera_info(calib_rgb_path)
    depth_calib = load_camera_info(calib_depth_path)

    logger.info("=" * 80)
    logger.info("Alignment Validation")
    logger.info("=" * 80)
    logger.info(f"RGB calibration: {rgb_calib}")
    logger.info(f"Depth calibration: {depth_calib}")

    # Find aligned depth files
    aligned_files = sorted(list(Path(aligned_dir).glob('*.png')))

    if not aligned_files:
        raise FileNotFoundError(f"No aligned depth files found in {aligned_dir}")

    logger.info(f"Found {len(aligned_files)} aligned depth files")

    # Sample files
    n_samples = min(num_samples, len(aligned_files))
    sample_indices = np.linspace(0, len(aligned_files) - 1, n_samples, dtype=int)
    sample_files = [aligned_files[i] for i in sample_indices]

    # Results
    visual_results = []
    quant_results = []

    for i, aligned_path in enumerate(sample_files):
        frame_id = aligned_path.stem
        logger.info(f"\nValidating [{i+1}/{n_samples}]: {frame_id}")

        # Find corresponding files
        # Depth: camera_DPT_XXX → camera_DPT_XXX.png
        # RGB: camera_DPT_XXX → camera_RGB_XXX.png
        rgb_filename = frame_id.replace('camera_DPT_', 'camera_RGB_') + '.png'
        depth_filename = frame_id + '.png'

        rgb_path = Path(rgb_dir) / rgb_filename
        depth_path = Path(depth_dir) / depth_filename

        if not rgb_path.exists():
            logger.warning(f"RGB not found: {rgb_path}, skipping")
            continue
        if not depth_path.exists():
            logger.warning(f"Depth not found: {depth_path}, skipping")
            continue

        try:
            # Load images
            rgb, depth_orig, aligned_depth = load_image_pair(
                str(rgb_path), str(depth_path), str(aligned_path)
            )

            # Visual validation
            visual_output = output_path / f'visual_{frame_id}.png'
            visual_result = visual_validation(rgb, aligned_depth, str(visual_output))
            visual_result['frame_id'] = frame_id
            visual_results.append(visual_result)

            logger.info(f"  Coverage: {visual_result['coverage']:.1%}")
            logger.info(f"  Edge overlap: {visual_result['edge_overlap']:.1%}")

            # Quantitative validation (only first sample for speed)
            if i == 0:
                quant_result = quantitative_validation(
                    depth_orig, aligned_depth,
                    depth_calib.K, rgb_calib.K,
                    num_samples=1000
                )
                quant_result['frame_id'] = frame_id
                quant_results.append(quant_result)

                logger.info(f"  Reprojection error:")
                logger.info(f"    Mean: {quant_result['mean_error']:.2f} pixels")
                logger.info(f"    Median: {quant_result['median_error']:.2f} pixels")
                logger.info(f"    95th percentile: {quant_result['p95_error']:.2f} pixels")

        except Exception as e:
            logger.error(f"Validation failed for {frame_id}: {e}", exc_info=True)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)

    if visual_results:
        avg_coverage = np.mean([r['coverage'] for r in visual_results])
        avg_edge_overlap = np.mean([r['edge_overlap'] for r in visual_results])

        logger.info(f"Average coverage: {avg_coverage:.1%}")
        logger.info(f"Average edge overlap: {avg_edge_overlap:.1%}")

        # Interpretation
        logger.info("\nInterpretation:")
        if avg_coverage > 0.7:
            logger.info("  ✅ Coverage is good (>70%)")
        elif avg_coverage > 0.5:
            logger.info("  ⚠️  Coverage is moderate (50-70%)")
        else:
            logger.info("  ❌ Coverage is low (<50%)")

        if avg_edge_overlap > 0.6:
            logger.info("  ✅ Edge alignment is good (>60%)")
        elif avg_edge_overlap > 0.4:
            logger.info("  ⚠️  Edge alignment is moderate (40-60%)")
        else:
            logger.info("  ❌ Edge alignment is poor (<40%)")

    if quant_results:
        quant = quant_results[0]
        logger.info(f"\nReprojection Error:")
        logger.info(f"  Mean: {quant['mean_error']:.2f} pixels")
        logger.info(f"  Median: {quant['median_error']:.2f} pixels")
        logger.info(f"  95th percentile: {quant['p95_error']:.2f} pixels")

        # Interpretation
        logger.info("\nInterpretation:")
        if quant['mean_error'] < 5:
            logger.info("  ✅ Reprojection error is excellent (<5px)")
        elif quant['mean_error'] < 10:
            logger.info("  ✅ Reprojection error is good (5-10px)")
        elif quant['mean_error'] < 20:
            logger.info("  ⚠️  Reprojection error is moderate (10-20px)")
        else:
            logger.info("  ❌ Reprojection error is high (>20px) - consider full geometric alignment")

    logger.info("\n" + "=" * 80)
    logger.info(f"Validation complete! Results saved to: {output_dir}")
    logger.info("=" * 80)

    return visual_results, quant_results


if __name__ == '__main__':
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Validate RGB-Depth alignment quality')
    parser.add_argument('--rgb-dir', required=True, help='RGB images directory')
    parser.add_argument('--depth-dir', required=True, help='Original depth images directory')
    parser.add_argument('--aligned-dir', required=True, help='Aligned depth images directory')
    parser.add_argument('--calib-rgb', required=True, help='RGB camera calibration JSON')
    parser.add_argument('--calib-depth', required=True, help='Depth camera calibration JSON')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to validate')
    parser.add_argument('--output-dir', default='outputs/alignment_validation', help='Output directory')

    args = parser.parse_args()

    setup_logging('INFO')

    try:
        validate_alignment(
            args.rgb_dir,
            args.depth_dir,
            args.aligned_dir,
            args.calib_rgb,
            args.calib_depth,
            args.num_samples,
            args.output_dir
        )
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        exit(1)
