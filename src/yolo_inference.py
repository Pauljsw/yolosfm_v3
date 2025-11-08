"""YOLO segmentation inference utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _polygon_to_list(polygon: Iterable[Iterable[float]]) -> List[List[float]]:
    """Convert polygon coordinates to a JSON-serialisable list."""
    return [[float(x), float(y)] for x, y in polygon]


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

        logger.debug(
            "Initialized YOLOSegmenter with weights=%s, conf=%.2f, iou=%.2f, img_size=%d",
            weights_path,
            conf,
            iou,
            img_size,
        )

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
