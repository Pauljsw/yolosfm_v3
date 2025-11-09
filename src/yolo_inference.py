"""YOLO segmentation inference utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _polygon_to_list(polygon: Iterable[Iterable[float]]) -> List[List[float]]:
    """Convert polygon coordinates to a JSON-serialisable list."""
    return [[float(x), float(y)] for x, y in polygon]


def visualize_detections(
    image_path: str,
    masks_data: List[dict],
    class_names: List[str],
    output_path: str,
    colors: Optional[dict] = None
) -> None:
    """
    Visualize YOLO detections with masks, bboxes, and labels.

    Args:
        image_path: Path to original RGB image
        masks_data: List of mask dictionaries with polygon, class, score
        class_names: List of class names
        output_path: Path to save visualization
        colors: Optional dict of class_name -> [B, G, R] color
    """
    # Load original image
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Failed to load image for visualization: {image_path}")
        return

    # Create overlay for masks
    overlay = img.copy()

    # Default colors (BGR format)
    if colors is None:
        default_colors = {
            'crack': [0, 0, 255],           # Red
            'spalling': [0, 255, 0],        # Green
            'efflorescence': [255, 0, 0],   # Blue
            'exposed_rebar': [0, 255, 255], # Yellow
            'corrosion': [255, 0, 255],     # Magenta
            'water_leakage': [255, 255, 0], # Cyan
            'honeycomb': [128, 0, 255],     # Purple
        }
        colors = default_colors

    for mask_info in masks_data:
        class_name = mask_info['class']
        score = mask_info['score']
        polygon = mask_info['polygon']

        # Get color for this class (BGR)
        color = colors.get(class_name, [255, 255, 255])  # Default white

        # Convert polygon to numpy array
        polygon_np = np.array(polygon, dtype=np.int32)

        # Draw filled polygon mask with transparency
        cv2.fillPoly(overlay, [polygon_np], color)

        # Draw polygon outline
        cv2.polylines(img, [polygon_np], isClosed=True, color=color, thickness=2)

        # Calculate bounding box for label placement
        x_coords = polygon_np[:, 0]
        y_coords = polygon_np[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Prepare label text
        label = f"{class_name}: {score:.2f}"

        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Draw label background
        label_y = y_min - 10 if y_min > 30 else y_max + 20
        cv2.rectangle(
            img,
            (x_min, label_y - text_height - 5),
            (x_min + text_width + 5, label_y + baseline),
            color,
            -1  # Filled
        )

        # Draw label text
        cv2.putText(
            img,
            label,
            (x_min + 2, label_y - 2),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness,
            cv2.LINE_AA
        )

    # Blend overlay with original image (30% transparency)
    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Save visualization
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

    logger.debug(f"Saved visualization to {output_path}")


class YOLOSegmenter:
    """Run YOLO segmentation and export masks as JSON polygons."""

    def __init__(
        self,
        weights_path: str,
        class_names: Sequence[str],
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 1024,
        device: Optional[str] = None,
        max_det: int = 300,
        visualization_dir: Optional[str] = None,
        colors: Optional[dict] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency import check
            raise ImportError(
                "ultralytics package is required for YOLO inference. "
                "Install it via `pip install ultralytics`."
            ) from exc

        self.model = YOLO(weights_path)
        self.class_names = list(class_names)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.device = device
        self.max_det = max_det
        self.visualization_dir = Path(visualization_dir) if visualization_dir else None
        self.colors = colors

        logger.debug(
            "Initialized YOLOSegmenter with weights=%s, conf=%.2f, iou=%.2f, img_size=%d",
            weights_path,
            conf,
            iou,
            img_size,
        )

        if self.visualization_dir:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"YOLO visualizations will be saved to: {self.visualization_dir}")

    def process_image(self, image_path: str, output_path: str) -> None:
        """Run inference on an image and save mask polygons to JSON."""
        image_path = str(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )

        if not results:
            logger.warning("No YOLO results returned for %s", image_path)
            self._write_output(output_path, [], 0, 0)
            return

        result = results[0]
        height, width = result.orig_shape[:2]

        masks_data = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)

        if masks_data is None or boxes is None or len(masks_data) == 0:
            logger.info("No segmentation masks detected for %s", image_path)
            self._write_output(output_path, [], width, height)
            return

        mask_polygons = masks_data.xy
        class_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        masks: List[dict] = []
        for idx, polygon in enumerate(mask_polygons):
            if idx >= len(class_ids):
                break

            class_id = class_ids[idx]
            if class_id >= len(self.class_names):
                logger.debug(
                    "Skipping detection %d in %s due to class id %d outside configured range",
                    idx,
                    image_path,
                    class_id,
                )
                continue

            polygon_points = _polygon_to_list(polygon)
            if len(polygon_points) < 3:
                continue

            masks.append(
                {
                    "class": self.class_names[class_id],
                    "class_id": int(class_id),
                    "score": float(scores[idx]),
                    "polygon": polygon_points,
                    "instance_id": f"{Path(image_path).stem}_{idx:04d}",
                }
            )

        self._write_output(output_path, masks, width, height)

        # Save visualization if directory is specified
        if self.visualization_dir and len(masks) > 0:
            vis_filename = Path(image_path).stem + ".png"
            vis_path = self.visualization_dir / vis_filename
            visualize_detections(
                image_path,
                masks,
                self.class_names,
                str(vis_path),
                self.colors
            )
            logger.info(f"Saved visualization: {vis_path}")

    def _write_output(self, output_path: Path, masks: List[dict], width: int, height: int) -> None:
        """Persist the YOLO mask predictions to disk."""
        data = {
            "image_id": output_path.stem,
            "width": int(width),
            "height": int(height),
            "masks": masks,
        }

        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.debug("Saved %d masks to %s", len(masks), output_path)


__all__ = ["YOLOSegmenter"]
