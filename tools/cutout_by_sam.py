from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

from tools import car_front_seg


CLASS_NAME_REMAP = {
    "front_bumper": "hood",
}


@dataclass(frozen=True)
class SamModelSpec:
    checkpoint: Path
    model_type: str = "vit_h"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def _clip_box_to_image(box: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = box.astype(np.float32).copy()
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, width)
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, height)
    return clipped


def _expand_box(box: np.ndarray, ratio: float, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    margin_x = (x2 - x1) * ratio
    margin_y = (y2 - y1) * ratio
    return _clip_box_to_image(
        np.array([x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y]),
        width,
        height,
    )


def _mask_inside_box(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in _clip_box_to_image(box, width, height)]
    limited = np.zeros_like(mask, dtype=bool)
    limited[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return limited


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if labels_count <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def _fill_largest_external_contour(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return mask.astype(bool)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(bool)
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(m, dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 1, thickness=-1)
    return filled.astype(bool)


def _clean_mask(
    *,
    sam_mask: np.ndarray,
    yolo_box: np.ndarray,
    image_shape: tuple[int, int],
    box_margin_ratio: float,
    keep_largest_component: bool,
    yolo_mask01: np.ndarray | None,
) -> np.ndarray:
    height, width = image_shape
    cleaned = _mask_inside_box(sam_mask, _expand_box(yolo_box, box_margin_ratio, width, height))

    if yolo_mask01 is not None:
        if yolo_mask01.ndim == 2 and (yolo_mask01.shape[0] != height or yolo_mask01.shape[1] != width):
            yolo_mask01 = cv2.resize(
                yolo_mask01.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        cleaned = np.logical_and(cleaned, yolo_mask01 >= 0.5)

    if keep_largest_component:
        cleaned = _largest_component(cleaned)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned.astype(bool)


def _load_sam_predictor(spec: SamModelSpec) -> SamPredictor:
    sam = sam_model_registry[spec.model_type](checkpoint=str(spec.checkpoint))
    sam.to(device=spec.device)
    return SamPredictor(sam)


def _run_yolo_instances(
    *,
    model: YOLO,
    rgb: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
) -> list[dict[str, Any]]:
    results = model.predict(rgb, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    if not results:
        return []
    raw = car_front_seg.unwrap_instances(results[0])
    for inst in raw:
        name = str(inst.get("name") or "")
        inst["name"] = CLASS_NAME_REMAP.get(name, name)
    return raw


def run_cutout_by_sam(
    *,
    stage_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    allowed_parts: set[str],
    sam_checkpoint: Path,
    sam_type: str = "vit_h",
    device: str | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    box_margin_ratio: float = 0.03,
    keep_largest_component: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    carpart_weight = Path(str(config.get("models", {}).get("carpart_weight") or ""))
    front_weight = Path(str(config.get("models", {}).get("front_part_weight") or ""))
    if not carpart_weight.is_file():
        raise FileNotFoundError(str(carpart_weight))
    if not front_weight.is_file():
        raise FileNotFoundError(str(front_weight))
    if not sam_checkpoint.is_file():
        raise FileNotFoundError(str(sam_checkpoint))

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_predictor = _load_sam_predictor(
        SamModelSpec(checkpoint=sam_checkpoint, model_type=sam_type, device=resolved_device)
    )

    model_carpart = YOLO(str(carpart_weight))
    model_front = YOLO(str(front_weight))

    for p in car_front_seg.iter_images(stage_dir):
        rgb = car_front_seg.load_rgb_with_white_bg(p)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        inst_car = _run_yolo_instances(model=model_carpart, rgb=rgb, conf=conf, iou=iou, imgsz=imgsz)
        inst_front = _run_yolo_instances(model=model_front, rgb=rgb, conf=conf, iou=iou, imgsz=imgsz)
        raw = [*inst_car, *inst_front]

        processed = car_front_seg.postprocess_instances(
            raw,
            car_front_seg.RULES,
            side_x_max=car_front_seg.SIDE_X_MAX,
            img_w=int(rgb.shape[1]),
        )

        processed = [x for x in processed if str(car_front_seg._inst_get(x, "name") or "") in allowed_parts]
        if not processed:
            continue

        sam_predictor.set_image(rgb)

        xyxy: list[list[float]] = []
        for inst in processed:
            box = car_front_seg._inst_get(inst, "box_xyxy")
            if box is None or len(box) != 4:
                continue
            xyxy.append([float(v) for v in box])
        if not xyxy:
            continue

        boxes_torch = torch.as_tensor(xyxy, dtype=torch.float32, device=sam_predictor.device)
        transformed = sam_predictor.transform.apply_boxes_torch(boxes_torch, rgb.shape[:2])
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        sam_masks = masks[:, 0].detach().cpu().numpy().astype(bool)

        refined: list[dict[str, Any]] = []
        j = 0
        for inst in processed:
            box = car_front_seg._inst_get(inst, "box_xyxy")
            if box is None or len(box) != 4:
                continue
            if j >= len(sam_masks):
                break

            yolo_mask01 = car_front_seg._as_mask01(car_front_seg._inst_get(inst, "mask01"))
            cleaned = _clean_mask(
                sam_mask=sam_masks[j],
                yolo_box=np.asarray(box, dtype=np.float32),
                image_shape=rgb.shape[:2],
                box_margin_ratio=box_margin_ratio,
                keep_largest_component=keep_largest_component,
                yolo_mask01=yolo_mask01,
            )
            name = str(car_front_seg._inst_get(inst, "name") or "")
            if name == "grille":
                cleaned = _fill_largest_external_contour(cleaned)
            j += 1
            inst2 = dict(inst)
            inst2["mask01"] = cleaned.astype(np.float32)
            refined.append(inst2)

        if not refined:
            continue

        car_front_seg.export_rgba_crops(bgr, refined, output_dir / p.stem, p.stem)
