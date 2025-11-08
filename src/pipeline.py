"""Main pipeline entrypoint for the YOLO + SFM fusion workflow."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Import modules
from .calib_io import load_camera_info, load_poses
from .align_depth_to_rgb import align_depth_to_rgb
from .project_mask_to_A import project_all_masks
from .fusion_3d import LabelFusion
from .instance_merge import merge_pipeline
from .measurement import measure_all_instances
from .export_results import export_all_results
from .utils import setup_logging, load_config, Timer, ensure_dir, list_files, find_rgb_depth_pairs
from .depth_tsdf_reconstruction import run_depth_reconstruction
from .sfm_scale_alignment import run_sfm_scale_alignment

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration YAML
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.setup_paths()

        # Camera calibrations (reloaded for each stage as needed)
        self.rgb_calib = None
        self.depth_calib = None
        self.reload_calibrations()

        # Pose bookkeeping
        self.poses: Dict[str, Dict] = {}
        self.pose_index: Dict[str, str] = {}
        self._load_poses(initial=True)

        # Class and colour metadata
        self.class_names: List[str] = []
        self.class_id_map: Dict[str, int] = {}
        self.colors: Dict[str, List[int]] = {}
        self._load_class_metadata()

        logger.info("Pipeline initialised with %d poses", len(self.poses))
        logger.info("Configured classes: %s", self.class_names)

    def reload_calibrations(self) -> None:
        """Reload camera calibration files from disk."""

        paths = self.config['paths']
        self.rgb_calib = load_camera_info(paths['calib_rgb'])
        self.depth_calib = load_camera_info(paths['calib_depth'])
        logger.debug("Loaded RGB calib %s and depth calib %s", paths['calib_rgb'], paths['calib_depth'])

    def _load_poses(self, *, initial: bool = False) -> None:
        """Load SFM poses if available and build a lookup index."""

        sfm_dir = Path(self.config['paths']['sfm_dir'])

        # Prefer aligned poses if available
        aligned_poses_path = sfm_dir / 'poses_aligned.json'
        poses_path = sfm_dir / 'poses.json'

        if aligned_poses_path.exists():
            poses_file = aligned_poses_path
            logger.info("Using aligned poses: %s", poses_file)
        elif poses_path.exists():
            poses_file = poses_path
            if not initial:
                logger.info("Using SFM poses (not aligned): %s", poses_file)
        else:
            self.poses = {}
            self.pose_index = {}
            message = "No poses found at %s" % poses_path
            if initial:
                logger.info("%s; run the SFM stage first if required.", message)
            else:
                logger.warning(message)
            return

        self.poses = load_poses(str(poses_file))
        self.pose_index = {}
        for key in self.poses.keys():
            stem = Path(key).stem
            if stem in self.pose_index:
                logger.warning("Duplicate pose stem detected for %s; keeping first entry.", stem)
                continue
            self.pose_index[stem] = key

    def _load_class_metadata(self) -> None:
        """Derive ordered class names and colour mappings."""

        yolo_cfg = self.config.get('yolo', {})
        names_from_data: Optional[List[str]] = None
        data_cfg_path = yolo_cfg.get('data_config')
        if data_cfg_path:
            data_cfg_path = Path(data_cfg_path)
            if not data_cfg_path.is_absolute():
                data_cfg_path = (self.config_path.parent / data_cfg_path).resolve()
            if not data_cfg_path.exists():
                logger.warning("YOLO data config not found at %s", data_cfg_path)
                data_cfg_path = None
        if data_cfg_path:
            try:
                data_cfg = load_config(data_cfg_path)
                names_section = data_cfg.get('names')
                if isinstance(names_section, dict):
                    names_from_data = [
                        names_section[key]
                        for key in sorted(names_section, key=lambda item: int(item))
                    ]
                elif isinstance(names_section, list):
                    names_from_data = [str(name) for name in names_section]
                if names_from_data:
                    logger.debug("Loaded %d class names from %s", len(names_from_data), data_cfg_path)
            except Exception as exc:  # pragma: no cover - configuration error path
                logger.warning("Failed to parse YOLO data config %s: %s", data_cfg_path, exc)

        classes_cfg = self.config.get('classes')
        names_from_config: Optional[List[str]] = None
        if isinstance(classes_cfg, dict) and classes_cfg:
            names_from_config = [name for name, _ in sorted(classes_cfg.items(), key=lambda item: item[1])]

        if names_from_data:
            class_names = names_from_data
            if names_from_config and names_from_config != class_names:
                logger.info(
                    "Class ordering from classes config differs from YOLO data config; using YOLO ordering."
                )
        elif names_from_config:
            class_names = names_from_config
        else:
            raise ValueError(
                "No class metadata available. Provide either `classes` mapping in the main config or "
                "set `yolo.data_config` to a YOLO dataset YAML containing class names."
            )

        self.class_names = class_names
        self.class_id_map = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # Persist class mapping for downstream modules that expect it in config
        self.config['classes'] = self.class_id_map

        configured_colors = self.config.get('colors', {}) or {}
        default_palette = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
        ]
        colours: Dict[str, List[int]] = {}
        for idx, name in enumerate(self.class_names):
            colour = configured_colors.get(name)
            if colour is None:
                colour = default_palette[idx % len(default_palette)]
            colours[name] = colour
        self.colors = colours
    
    def setup_paths(self):
        """Setup and validate paths"""
        paths = self.config['paths']

        # Ensure output directories exist
        ensure_dir(paths['out_dir'])
        ensure_dir(f"{paths['out_dir']}/aligned_depth")
        ensure_dir(f"{paths['out_dir']}/fused")
        ensure_dir(f"{paths['out_dir']}/report")
        ensure_dir(paths['masks_dir'])

    def run_sfm(self):
        """
        Stage 0: Structure from Motion with COLMAP
        """
        logger.info("=" * 80)
        logger.info("Stage 0: Structure from Motion (COLMAP)")
        logger.info("=" * 80)
        
        try:
            from .colmap_sfm import run_colmap_sfm_auto
        except ImportError:
            logger.error("colmap_sfm module not found")
            return

        with Timer("SFM"):
            rgb_dir = self.config['paths']['rgb_dir']
            sfm_dir = self.config['paths']['sfm_dir']

            ensure_dir(sfm_dir)
            
            # Get SFM config
            sfm_config = self.config.get('sfm', {})
            camera_model = sfm_config.get('camera_model', 'OPENCV')
            quality = sfm_config.get('quality', 'high')
            dense = sfm_config.get('dense', False)
            
            poses_output = f"{sfm_dir}/poses.json"
            
            logger.info(f"Running COLMAP on images in: {rgb_dir}")
            logger.info(f"Camera model: {camera_model}, Quality: {quality}")
            
            # Run COLMAP
            poses = run_colmap_sfm_auto(
                image_dir=rgb_dir,
                output_dir=sfm_dir,
                poses_json_output=poses_output,
                camera_model=camera_model,
                quality=quality,
                dense=dense
            )
            
            logger.info(f"SFM complete: {len(poses)} images reconstructed")
            logger.info(f"Poses saved to: {poses_output}")

            # Reload poses
            self._load_poses()

        logger.info("SFM stage completed")

    def run_depth_ground_truth(self):
        """
        Phase 1: Depth-only Ground Truth Reconstruction
        Generates absolute-scale 3D model from depth images.
        """
        logger.info("=" * 80)
        logger.info("Phase 1: Depth Ground Truth Reconstruction")
        logger.info("=" * 80)

        self.reload_calibrations()

        with Timer("Depth Reconstruction"):
            rgb_dir = self.config['paths']['rgb_dir']
            depth_dir = self.config['paths']['depth_dir']
            output_dir = self.config['paths'].get('depth_gt_dir', 'output_depth_tsdf')

            # Find RGB-Depth pairs
            pairs = find_rgb_depth_pairs(rgb_dir, depth_dir)

            if not pairs:
                logger.error("No RGB-Depth pairs found!")
                return

            logger.info(f"Found {len(pairs)} RGB-Depth pairs")

            # Get reconstruction config
            depth_config = self.config.get('depth_reconstruction', {})
            tsdf_voxel_size = depth_config.get('tsdf_voxel_size', 0.01)
            tsdf_trunc_factor = depth_config.get('tsdf_trunc_factor', 4.0)
            depth_unit = depth_config.get('depth_unit', 'auto')
            use_icp = depth_config.get('use_icp', True)
            icp_voxel_size = depth_config.get('icp_voxel_size', 0.02)
            icp_max_corr_dist = depth_config.get('icp_max_corr_dist', 0.05)
            use_undistortion = depth_config.get('use_undistortion', False)

            # Run reconstruction
            results = run_depth_reconstruction(
                pairs,
                self.depth_calib.K,
                output_dir=output_dir,
                tsdf_voxel_size=tsdf_voxel_size,
                tsdf_trunc_factor=tsdf_trunc_factor,
                depth_unit=depth_unit,
                use_icp=use_icp,
                icp_voxel_size=icp_voxel_size,
                icp_max_corr_dist=icp_max_corr_dist,
                use_undistortion=use_undistortion,
                depth_D=self.depth_calib.D if use_undistortion else None,
                depth_width=self.depth_calib.width,
                depth_height=self.depth_calib.height
            )

            logger.info(f"Depth reconstruction complete:")
            logger.info(f"  Point cloud: {results['pcd_path']}")
            logger.info(f"  Points: {results['num_points']}")

        logger.info("Phase 1 completed")

    def run_sfm_scale_align(self):
        """
        Phase 2: SFM Scale Alignment
        Aligns SFM poses to absolute scale using depth ground truth.
        """
        logger.info("=" * 80)
        logger.info("Phase 2: SFM Scale Alignment")
        logger.info("=" * 80)

        with Timer("Scale Alignment"):
            sfm_dir = self.config['paths']['sfm_dir']
            sfm_poses_path = f"{sfm_dir}/poses.json"
            sfm_sparse_dir = f"{sfm_dir}/sparse/0"

            depth_gt_dir = self.config['paths'].get('depth_gt_dir', 'output_depth_tsdf')
            depth_gt_pcd = f"{depth_gt_dir}/fused_pointcloud.ply"

            output_poses_path = f"{sfm_dir}/poses_aligned.json"

            # Check if files exist
            if not Path(sfm_poses_path).exists():
                logger.error(f"SFM poses not found: {sfm_poses_path}")
                logger.error("Run SFM stage first!")
                return

            if not Path(depth_gt_pcd).exists():
                logger.error(f"Depth ground truth not found: {depth_gt_pcd}")
                logger.error("Run depth reconstruction stage first!")
                return

            if not Path(sfm_sparse_dir).exists():
                logger.error(f"COLMAP sparse model not found: {sfm_sparse_dir}")
                return

            # Get alignment config
            align_config = self.config.get('scale_alignment', {})
            use_camera_trajectory = align_config.get('use_camera_trajectory', True)
            use_feature_matching = align_config.get('use_feature_matching', True)

            # Depth trajectory path
            depth_trajectory_path = f"{depth_gt_dir}/trajectory.json"

            # Run alignment
            result = run_sfm_scale_alignment(
                sfm_poses_path,
                sfm_sparse_dir,
                depth_gt_pcd,
                output_poses_path,
                depth_trajectory_path=depth_trajectory_path,
                use_camera_trajectory=use_camera_trajectory,
                use_feature_matching=use_feature_matching
            )

            logger.info(f"Scale alignment complete:")
            logger.info(f"  Method: {result.get('alignment_method', 'unknown')}")
            logger.info(f"  Scale factor: {result['scale']:.4f}")
            logger.info(f"  RMSE: {result['rmse']:.4f} m")
            logger.info(f"  Correspondences: {result['num_correspondences']}")
            logger.info(f"  Aligned poses: {output_poses_path}")

            # Update pipeline to use aligned poses
            self._load_poses()

        logger.info("Phase 2 completed")


    def run_alignment(self):
        """
        Stage 1: Align depth to RGB for all images.
        """
        logger.info("=" * 80)
        logger.info("Stage 1: Depth-to-RGB Alignment")
        logger.info("=" * 80)

        self.reload_calibrations()


        # Load extrinsic transformation (Depth → Color)
        import json
        extrinsic_path = Path(self.config['paths'].get('calib_dir', 'calib')) / 'extrinsic_depth_to_color.json'
        T_d2r = None
        if extrinsic_path.exists():
            with open(extrinsic_path, 'r') as f:
                extrinsic_data = json.load(f)
                R = np.array(extrinsic_data['R'])
                t = np.array(extrinsic_data['t']).reshape(3, 1)
                T_d2r = (R, t)
                logger.info(f"✅ Loaded extrinsic transformation from {extrinsic_path}")
                logger.info(f"   Translation: {t.flatten()} meters")
        else:
            logger.warning(f"⚠️  Extrinsic file not found: {extrinsic_path}")
            logger.warning("   Using identity transformation (may cause alignment errors)")

        with Timer("Alignment"):
            # Get list of depth images
            depth_dir = self.config['paths']['depth_dir']
            depth_files = list_files(depth_dir, '.png')
            
            if not depth_files:
                logger.error(f"No depth images found in {depth_dir}")
                return
            
            logger.info(f"Found {len(depth_files)} depth images")
            
            align_config = self.config['align']
            bilateral_params = {
                'd': align_config.get('bilateral_d', 9),
                'sigma_color': align_config.get('bilateral_sigma_color', 75),
                'sigma_space': align_config.get('bilateral_sigma_space', 75)
            }
            
            output_dir = f"{self.config['paths']['out_dir']}/aligned_depth"
            
            for i, depth_file in enumerate(depth_files):
                image_id = Path(depth_file).stem
                
                logger.info(f"Processing [{i+1}/{len(depth_files)}]: {image_id}")
                
                # Load depth
                import cv2
                depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
                
                # Align
                aligned = align_depth_to_rgb(
                    depth_img,
                    self.rgb_calib.K,
                    self.rgb_calib.D,
                    self.depth_calib.K,
                    self.depth_calib.D,
                    rgb_size=(self.rgb_calib.width, self.rgb_calib.height),
                    T_d2r=T_d2r,  # ← 이 줄 추가!
                    depth_unit=align_config['in_depth_unit'],
                    hole_fill=align_config['hole_fill'],
                    joint_bilateral=align_config['joint_bilateral'],
                    bilateral_params=bilateral_params,
                    use_simple_resize=align_config.get('use_simple_resize', False)
                )
                
                # Save
                output_path = f"{output_dir}/{image_id}.png"
                # Save as float32 EXR or scaled uint16
                aligned_scaled = (aligned * 1000).astype(np.uint16)  # mm
                cv2.imwrite(output_path, aligned_scaled)
                
                logger.info(f"  Saved aligned depth: {output_path}")
        
        logger.info("Alignment stage completed")

    def _resolve_rgb_path(self, image_id: str) -> Optional[Path]:
        """Resolve the RGB image path for a given image identifier."""
        rgb_dir = Path(self.config['paths']['rgb_dir'])

        candidates = [
            rgb_dir / image_id,
            *(rgb_dir / f"{image_id}{ext}" for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'])
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def run_detection(self, reinfer_mode: str = 'auto'):
        """Stage 1.5: Run YOLO segmentation to produce mask JSON files."""
        logger.info("=" * 80)
        logger.info("Stage 1.5: YOLO Segmentation Inference")
        logger.info("=" * 80)

        self._load_poses()
        if not self.pose_index:
            logger.error("No SFM poses available. Run the SFM stage before YOLO inference.")
            return

        image_ids = sorted(self.pose_index.keys())
        effective_mode = reinfer_mode
        if effective_mode == 'off':
            effective_mode = self.config.get('reinfer', {}).get('mode', 'auto')

        self.run_yolo_inference(image_ids, effective_mode)

    def run_yolo_inference(self, image_ids: List[str], reinfer_mode: str):
        """Execute YOLO segmentation inference for the provided image ids."""

        if reinfer_mode == 'off':
            logger.info("Reinference mode is 'off'; skipping YOLO inference.")
            return

        yolo_config = self.config.get('yolo')
        if not yolo_config:
            logger.warning("YOLO configuration missing in config file; skipping inference.")
            return

        try:
            from .yolo_inference import YOLOSegmenter
        except ImportError as exc:
            logger.error("Failed to import YOLO inference utilities: %s", exc)
            return

        weights_path = yolo_config.get('weights')
        if not weights_path:
            logger.error("YOLO weights path not specified. Set 'yolo.weights' in the config file.")
            return

        weights_path = Path(weights_path)
        if not weights_path.exists():
            logger.error("YOLO weights not found at %s", weights_path)
            return

        masks_dir = Path(self.config['paths']['masks_dir'])
        ensure_dir(str(masks_dir))

        force = reinfer_mode == 'on'

        images_to_process = []
        for image_id in image_ids:
            image_path = self._resolve_rgb_path(image_id)
            if image_path is None:
                logger.warning("RGB image not found for %s; skipping YOLO inference.", image_id)
                continue

            mask_path = masks_dir / f"{image_id}.json"
            if mask_path.exists() and not force:
                logger.debug("Mask already exists for %s; skipping in auto mode.", image_id)
                continue

            images_to_process.append((image_id, image_path, mask_path))

        if not images_to_process:
            logger.info("No images require YOLO inference.")
            return

        logger.info("Running YOLO segmentation on %d images", len(images_to_process))

        with Timer("YOLO Inference"):
            inferencer = YOLOSegmenter(
                weights_path=str(weights_path),
                class_names=self.class_names,
                conf=yolo_config.get('conf', 0.25),
                iou=yolo_config.get('iou', 0.45),
                img_size=yolo_config.get('img_size', 1024),
                device=yolo_config.get('device'),
                max_det=yolo_config.get('max_det', 300)
            )

            for idx, (image_id, image_path, mask_path) in enumerate(images_to_process, start=1):
                logger.info("[%d/%d] Running YOLO on %s", idx, len(images_to_process), image_path.name)
                inferencer.process_image(str(image_path), str(mask_path))

        logger.info("YOLO inference completed. Masks saved to %s", masks_dir)

    def run_fusion(self, reinfer_mode: str = 'off'):
        """
        Stage 2: 3D Label Fusion

        Args:
            reinfer_mode: 'off', 'on', or 'auto'
        """
        logger.info("=" * 80)
        logger.info("Stage 2: 3D Label Fusion")
        logger.info("=" * 80)

        self.reload_calibrations()
        self._load_poses()
        if not self.pose_index:
            logger.error("No SFM poses available. Run the SFM stage before 3D fusion.")
            return

        with Timer("3D Fusion"):
            fusion_config = self.config['fusion']

            # Initialize fusion
            voxel_size = fusion_config['voxel_size_cm'] / 100.0  # to meters
            fusion = LabelFusion(
                voxel_size=voxel_size,
                num_classes=self.num_classes,
                class_names=self.class_names,
                config=fusion_config
            )
            
            # Process each image
            aligned_depth_dir = Path(self.config['paths']['out_dir']) / 'aligned_depth'
            masks_dir = Path(self.config['paths']['masks_dir'])

            image_ids = sorted(self.pose_index.keys())

            effective_mode = reinfer_mode
            if effective_mode == 'off':
                effective_mode = self.config.get('reinfer', {}).get('mode', 'off')

            if effective_mode in ('on', 'auto'):
                self.run_yolo_inference(image_ids, effective_mode)

            for i, image_id in enumerate(image_ids):
                logger.info(f"Processing image [{i+1}/{len(image_ids)}]: {image_id}")
                
                # Load aligned depth
                depth_path = aligned_depth_dir / f"{image_id}.png"
                if not depth_path.exists():
                    logger.warning(f"Aligned depth not found: {depth_path}")
                    continue

                import cv2
                aligned_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # to meters

                # Load masks
                masks_path = masks_dir / f"{image_id}.json"
                if not masks_path.exists():
                    logger.warning(f"Masks not found: {masks_path}")
                    continue

                # Get pose
                pose_key = self.pose_index.get(image_id)
                if pose_key is None:
                    logger.warning(f"Pose not found for {image_id}")
                    continue

                pose = self.poses[pose_key]
                R = pose['R']
                t = pose['t']
                K = pose.get('K', self.rgb_calib.K)
                
                # Project masks to 3D
                projections = project_all_masks(
                    str(masks_path),
                    aligned_depth,
                    K,
                    self.rgb_calib.D,
                    R,
                    t,
                    self.class_names,
                    image_shape=(self.rgb_calib.height, self.rgb_calib.width)
                )
                
                # Fuse into voxel grid
                fusion.fuse_image(projections)
                
                logger.info(f"  Fused {len(projections)} masks")
            
            # Finalize fusion
            prob_thresh = fusion_config['prob_thresh']
            fusion_result = fusion.finalize(prob_thresh=prob_thresh)
            
            logger.info(f"Fusion complete: {len(fusion_result['labels'])} labeled voxels")
        
        # Stage 3: Instance Merging
        logger.info("=" * 80)
        logger.info("Stage 3: Instance Merging")
        logger.info("=" * 80)
        
        with Timer("Instance Merging"):
            merge_config = self.config['merge']
            
            instances = merge_pipeline(
                fusion_result,
                voxel_size,
                self.class_names,
                merge_config
            )
            
            logger.info(f"Merged to {len(instances)} instances")
        
        # Stage 4: Measurement
        logger.info("=" * 80)
        logger.info("Stage 4: Measurement")
        logger.info("=" * 80)
        
        with Timer("Measurement"):
            measure_config = self.config['measure']
            
            measurements = measure_all_instances(
                instances,
                voxel_size,
                measure_config
            )
            
            logger.info(f"Measured {len(measurements)} instances")
        
        # Stage 5: Export
        logger.info("=" * 80)
        logger.info("Stage 5: Export Results")
        logger.info("=" * 80)
        
        with Timer("Export"):
            output_dir = f"{self.config['paths']['out_dir']}/fused"
            
            export_all_results(
                fusion_result,
                instances,
                measurements,
                output_dir,
                self.class_names,
                self.config,
                self.colors
            )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
    
    def run_report(self):
        """
        Generate final report.
        """
        logger.info("=" * 80)
        logger.info("Generating Final Report")
        logger.info("=" * 80)
        
        # Report already generated in export stage
        report_path = f"{self.config['paths']['out_dir']}/fused/report.md"
        
        if Path(report_path).exists():
            logger.info(f"Report available at: {report_path}")
        else:
            logger.warning("Report not found. Run fusion stage first.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='YOLO + SFM 3D Fusion Pipeline')

    parser.add_argument('command',
                       choices=['sfm', 'depth_gt', 'scale_align', 'align', 'infer', 'fuse3d', 'report', 'full'],
                       help='Pipeline command to run')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--reinfer', type=str, choices=['off', 'on', 'auto'], default='off',
                       help='Reinference mode for YOLO segmentation masks')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Optional log file path')
    parser.add_argument('--skip-scale-align', action='store_true',
                       help='Skip scale alignment step in full pipeline')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    logger.info("=" * 80)
    logger.info("YOLO + SFM 3D Fusion Pipeline")
    logger.info("=" * 80)
    logger.info(f"Command: {args.command}")
    logger.info(f"Config: {args.config}")

    # Check config exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    # Initialize pipeline
    try:
        pipeline = Pipeline(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        sys.exit(1)

    # Run command
    try:
        if args.command == 'sfm':
            pipeline.run_sfm()

        elif args.command == 'depth_gt':
            pipeline.run_depth_ground_truth()

        elif args.command == 'scale_align':
            pipeline.run_sfm_scale_align()

        elif args.command == 'align':
            pipeline.run_alignment()

        elif args.command == 'infer':
            reinfer_mode = args.reinfer if args.reinfer != 'off' else 'auto'
            pipeline.run_detection(reinfer_mode=reinfer_mode)

        elif args.command == 'fuse3d':
            pipeline.run_fusion(reinfer_mode=args.reinfer)

        elif args.command == 'report':
            pipeline.run_report()

        elif args.command == 'full':
            # Full pipeline with all phases
            logger.info("Running FULL pipeline with scale alignment")

            # Phase 0: Check/Run SFM
            poses_path = Path(pipeline.config['paths']['sfm_dir']) / 'poses.json'
            if not poses_path.exists():
                logger.info("Poses not found, running SFM first...")
                pipeline.run_sfm()

            # Phase 1: Depth ground truth reconstruction
            depth_gt_dir = pipeline.config['paths'].get('depth_gt_dir', 'output_depth_tsdf')
            depth_gt_pcd = Path(depth_gt_dir) / 'fused_pointcloud.ply'

            if not depth_gt_pcd.exists():
                logger.info("Depth ground truth not found, running reconstruction...")
                pipeline.run_depth_ground_truth()
            else:
                logger.info(f"Using existing depth ground truth: {depth_gt_pcd}")

            # Phase 2: Scale alignment
            if not args.skip_scale_align:
                aligned_poses_path = Path(pipeline.config['paths']['sfm_dir']) / 'poses_aligned.json'
                if not aligned_poses_path.exists():
                    logger.info("Running SFM scale alignment...")
                    pipeline.run_sfm_scale_align()
                else:
                    logger.info(f"Using existing aligned poses: {aligned_poses_path}")
                    pipeline._load_poses()  # Reload to use aligned poses
            else:
                logger.info("Skipping scale alignment (--skip-scale-align)")

            # Phase 3-7: Rest of pipeline
            pipeline.run_alignment()
            pipeline.run_detection(reinfer_mode=args.reinfer)
            pipeline.run_fusion(reinfer_mode=args.reinfer)
            pipeline.run_report()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Pipeline execution completed")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
