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


def _shrink_box(box: np.ndarray, ratio: float, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    margin_x = max(0.0, (x2 - x1) * ratio)
    margin_y = max(0.0, (y2 - y1) * ratio)
    if x1 + margin_x >= x2 - margin_x or y1 + margin_y >= y2 - margin_y:
        return _clip_box_to_image(box, width, height)
    return _clip_box_to_image(
        np.array([x1 + margin_x, y1 + margin_y, x2 - margin_x, y2 - margin_y]),
        width,
        height,
    )


def _mask_inside_box(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in _clip_box_to_image(box, width, height)]
    limited = np.zeros_like(mask, dtype=bool)
    limited[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return limited


def _mask_bbox_width(mask: np.ndarray) -> int:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0
    return int(xs.max() - xs.min() + 1)


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


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return m.astype(bool)
    flood = m.copy()
    h, w = flood.shape[:2]
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), dtype=np.uint8), (0, 0), 1)
    holes = flood == 0
    out = m.copy()
    out[holes] = 1
    return out.astype(bool)


def _resize_mask01(mask01: np.ndarray | None, width: int, height: int) -> np.ndarray | None:
    if mask01 is None:
        return None
    if mask01.ndim == 2 and (mask01.shape[0] != height or mask01.shape[1] != width):
        return cv2.resize(mask01.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return mask01


def _clean_grille_mask(
    *,
    sam_mask: np.ndarray,
    yolo_box: np.ndarray,
    image_shape: tuple[int, int],
    yolo_mask01: np.ndarray | None,
) -> np.ndarray:
    height, width = image_shape
    yolo_mask01 = _resize_mask01(yolo_mask01, width, height)
    box = _shrink_box(yolo_box, 0.02, width, height)

    sam_limited = _mask_inside_box(sam_mask, box)
    if yolo_mask01 is not None:
        yolo_limited = _mask_inside_box(yolo_mask01 >= 0.5, box)
        intersect = np.logical_and(sam_limited, yolo_limited)
        yolo_area = int(np.sum(yolo_limited))
        intersect_area = int(np.sum(intersect))
        yolo_width = _mask_bbox_width(yolo_limited)
        intersect_width = _mask_bbox_width(intersect)
        area_ok = yolo_area <= 0 or intersect_area >= int(yolo_area * 0.65)
        width_ok = yolo_width <= 0 or intersect_width >= int(yolo_width * 0.75)
        base = intersect if area_ok and width_ok else yolo_limited
    else:
        base = sam_limited

    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = _mask_inside_box(m.astype(bool), box)
    m = _fill_mask_holes(m)
    m = _mask_inside_box(m, box)
    if yolo_mask01 is not None:
        yolo_width = _mask_bbox_width(yolo_limited)
        final_width = _mask_bbox_width(m)
        if yolo_width > 0 and final_width < int(yolo_width * 0.75):
            m = yolo_limited
    return m.astype(bool)


def _clean_front_bumper_mask(
    *,
    sam_mask: np.ndarray,
    yolo_box: np.ndarray,
    image_shape: tuple[int, int],
    yolo_mask01: np.ndarray | None,
) -> np.ndarray:
    height, width = image_shape
    yolo_mask01 = _resize_mask01(yolo_mask01, width, height)
    box = _clip_box_to_image(yolo_box, width, height)

    sam_limited = _mask_inside_box(sam_mask, box)
    if yolo_mask01 is not None:
        yolo_limited = _mask_inside_box(yolo_mask01 >= 0.5, box)
        base = np.logical_or(sam_limited, yolo_limited)
    else:
        base = sam_limited

    kernel = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    m = _mask_inside_box(m.astype(bool), box)
    m = _largest_component(m)
    m = _fill_mask_holes(m)
    m = _mask_inside_box(m, box)
    return m.astype(bool)


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
        yolo_mask01 = _resize_mask01(yolo_mask01, width, height)
        cleaned = np.logical_and(cleaned, yolo_mask01 >= 0.5)

    if keep_largest_component:
        cleaned = _largest_component(cleaned)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned.astype(bool)


def load_sam_predictor(spec: SamModelSpec) -> SamPredictor:
    sam = sam_model_registry[spec.model_type](checkpoint=str(spec.checkpoint))
    sam.to(device=spec.device)
    return SamPredictor(sam)


def run_sam_cutout_from_instances(
    *,
    rgb: np.ndarray,
    bgr: np.ndarray,
    instances: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
    sam_predictor: SamPredictor,
    box_margin_ratio: float = 0.03,
    keep_largest_component: bool = True,
) -> None:
    if not instances:
        return

    sam_predictor.set_image(rgb)

    xyxy: list[list[float]] = []
    boxed_instances: list[dict[str, Any]] = []
    for inst in instances:
        box = car_front_seg._inst_get(inst, "box_xyxy")
        if box is None or len(box) != 4:
            continue
        xyxy.append([float(v) for v in box])
        boxed_instances.append(inst)
    if not xyxy:
        return

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
    for inst, sam_mask in zip(boxed_instances, sam_masks):
        box = car_front_seg._inst_get(inst, "box_xyxy")
        if box is None or len(box) != 4:
            continue

        yolo_mask01 = car_front_seg._as_mask01(car_front_seg._inst_get(inst, "mask01"))
        name = str(car_front_seg._inst_get(inst, "name") or "")
        if name == "grille":
            cleaned = _clean_grille_mask(
                sam_mask=sam_mask,
                yolo_box=np.asarray(box, dtype=np.float32),
                image_shape=rgb.shape[:2],
                yolo_mask01=yolo_mask01,
            )
        elif name == "front_bumper":
            cleaned = _clean_front_bumper_mask(
                sam_mask=sam_mask,
                yolo_box=np.asarray(box, dtype=np.float32),
                image_shape=rgb.shape[:2],
                yolo_mask01=yolo_mask01,
            )
        else:
            cleaned = _clean_mask(
                sam_mask=sam_mask,
                yolo_box=np.asarray(box, dtype=np.float32),
                image_shape=rgb.shape[:2],
                box_margin_ratio=box_margin_ratio,
                keep_largest_component=keep_largest_component,
                yolo_mask01=yolo_mask01,
            )
        inst2 = dict(inst)
        inst2["mask01"] = cleaned.astype(np.float32)
        refined.append(inst2)

    if refined:
        car_front_seg.export_rgba_crops(bgr, refined, output_dir / stem, stem)


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

    front_weight = Path(str(config.get("models", {}).get("front_part_weight") or ""))
    if not front_weight.is_file():
        raise FileNotFoundError(str(front_weight))
    if not sam_checkpoint.is_file():
        raise FileNotFoundError(str(sam_checkpoint))

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_predictor = load_sam_predictor(
        SamModelSpec(checkpoint=sam_checkpoint, model_type=sam_type, device=resolved_device)
    )

    model_front = YOLO(str(front_weight))

    for p in car_front_seg.iter_images(stage_dir):
        rgb = car_front_seg.load_rgb_with_white_bg(p)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        processed = car_front_seg.detect_processed_instances(
            model=model_front,
            rgb=rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            allowed_parts=allowed_parts,
        )
        run_sam_cutout_from_instances(
            rgb=rgb,
            bgr=bgr,
            instances=processed,
            output_dir=output_dir,
            stem=p.stem,
            sam_predictor=sam_predictor,
            box_margin_ratio=box_margin_ratio,
            keep_largest_component=keep_largest_component,
        )
