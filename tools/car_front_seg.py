from PIL import Image
import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Union

WEIGHTS = "models/yolo-seg/yolo11m-seg-front-6label-2000.pt"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640
DEFAULT_IMG_PATH = Path("result/img-front/")
DEFAULT_LABEL_DIR = Path("result/front_label/")
DEFAULT_OUT_ROOT = Path("result/front_parts/")
SIDE_X_MAX = 0.5

MergeMode = Literal["none", "union"]

@dataclass(frozen=True)
class PartRule:
    name: str
    side: Literal["any", "left"] = "any"
    max_instances: int = 999
    merge_mode: MergeMode = "none"

RULES = {
    "right_mirror": PartRule(name="right_mirror", side="left", max_instances=1, merge_mode="union"),
    "front_right_light": PartRule(name="front_right_light", side="left", max_instances=1, merge_mode="none"),
    "grille": PartRule(name="grille", max_instances=1),
    "front_glass": PartRule(name="front_glass", max_instances=1),
    "hood": PartRule(name="hood", max_instances=1),
    "front_bumper": PartRule(name="front_bumper", max_instances=1),
}

def iter_images(path: Path) -> list[Path]:
    if path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
        return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if path.is_file():
        return [path]
    raise FileNotFoundError(path)

def load_rgb_with_white_bg(path: Path) -> np.ndarray:
    with Image.open(path) as im0:
        im = im0.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        return np.array(Image.alpha_composite(bg, im).convert("RGB"))


def _imwrite_cn(path: Path, img: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _stable_color(name: str) -> tuple[int, int, int]:
    v = 0
    for ch in name:
        v = (v * 131 + ord(ch)) & 0xFFFFFFFF
    b = 64 + (v & 0x7F)
    g = 64 + ((v >> 7) & (0x7F))
    r = 64 + ((v >> 14) & (0x7F))
    return int(b), int(g), int(r)


def _draw_label(
    img_bgr: np.ndarray,
    anchor_xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    occupied_boxes: list[tuple[int, int, int, int]],
) -> None:
    h, w = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_w = tw + 8
    box_h = th + baseline + 8

    ax, ay = anchor_xy
    ax = max(0, min(int(ax), w - 1))
    ay = max(0, min(int(ay), h - 1))

    def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    candidates = [
        (ax + 70, ay - 50),
        (ax + 70, ay + 10),
        (ax - box_w - 70, ay - 50),
        (ax - box_w - 70, ay + 10),
        (ax + 20, ay - box_h - 20),
        (ax + 20, ay + 20),
    ]

    chosen_rect: Optional[tuple[int, int, int, int]] = None
    for cx, cy in candidates:
        tx1 = max(0, min(int(cx), w - box_w))
        ty1 = max(0, min(int(cy), h - box_h))
        rect = (tx1, ty1, tx1 + box_w, ty1 + box_h)
        if any(_overlap(rect, ob) for ob in occupied_boxes):
            continue
        chosen_rect = rect
        break

    if chosen_rect is None:
        tx1 = max(0, min(ax + 20, w - box_w))
        ty1 = max(0, min(ay - box_h - 20, h - box_h))
        chosen_rect = (tx1, ty1, tx1 + box_w, ty1 + box_h)

    tx1, ty1, tx2, ty2 = chosen_rect
    text_org = (tx1 + 4, ty2 - baseline - 3)
    line_end = (tx1, ty1 + box_h // 2) if tx1 > ax else (tx2, ty1 + box_h // 2)

    cv2.line(img_bgr, (ax, ay), line_end, color, 2, cv2.LINE_AA)
    cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), color, -1)
    cv2.putText(img_bgr, text, text_org, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    occupied_boxes.append(chosen_rect)


def _mask_anchor(mask01: np.ndarray, img_w: int, img_h: int) -> Optional[tuple[int, int]]:
    if mask01.ndim != 2:
        return None
    if mask01.shape[0] != img_h or mask01.shape[1] != img_w:
        mask01 = cv2.resize(mask01.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    m = mask01 > 0.5
    if not np.any(m):
        return None
    ys, xs = np.where(m)
    return int(np.mean(xs)), int(np.mean(ys))


def _overlay_mask(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> None:
    h, w = img_bgr.shape[:2]
    if mask01.ndim != 2:
        return
    m = mask01.astype(np.float32)
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    m = np.clip(m, 0.0, 1.0)
    if not np.any(m > 0.01):
        return

    near_binary_ratio = float(np.mean((m < 0.01) | (m > 0.99)))
    if near_binary_ratio > 0.995:
        m = cv2.GaussianBlur(m, (5, 5), 0)
        m = np.clip(m, 0.0, 1.0)

    a = float(max(0.0, min(1.0, alpha)))
    a_map = (m * a)[:, :, None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = img_bgr.astype(np.float32) * (1.0 - a_map) + color_arr * a_map
    img_bgr[:, :, :] = out.clip(0, 255).astype(np.uint8)


def _fill_largest_external_contour(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(np.uint8)
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(m, dtype=np.uint8)
    cv2.drawContours(filled, [largest], -1, 1, thickness=-1)
    return filled


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 1:
        return m
    largest_idx = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_idx).astype(np.uint8)


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return m
    flood = m.copy()
    h, w = flood.shape[:2]
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), dtype=np.uint8), (0, 0), 1)
    holes = flood == 0
    out = m.copy()
    out[holes] = 1
    return out.astype(np.uint8)


def _trim_front_bumper_side_artifacts(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8).copy()
    if m.max() == 0:
        return m
    _, xs = np.where(m > 0)
    if xs.size == 0:
        return m
    x1, x2 = int(xs.min()), int(xs.max())
    width = x2 - x1 + 1
    trim = int(round(width * 0.018))
    trim = max(4, min(trim, 18))
    if width <= trim * 4:
        return m
    m[:, x1 : x1 + trim] = 0
    m[:, x2 - trim + 1 : x2 + 1] = 0
    return m


def _prepare_front_bumper_mask(mask: np.ndarray) -> np.ndarray:
    m = _trim_front_bumper_side_artifacts(mask)
    m = _keep_largest_component(m)
    m = _fill_mask_holes(m)
    m = cv2.erode(
        m,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if not np.any(m):
        return mask.astype(np.uint8)
    return m.astype(np.uint8)


def _mask_to_rgba_crop(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    *,
    fill_external_contour: bool = False,
) -> Optional[np.ndarray]:
    h, w = img_bgr.shape[:2]
    if mask01.ndim != 2:
        return None
    if mask01.shape[0] != h or mask01.shape[1] != w:
        mask01 = cv2.resize(mask01.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m = np.clip(mask01.astype(np.float32), 0.0, 1.0)
    if not np.any(m > 0.03):
        return None

    near_binary_ratio = float(np.mean((m < 0.01) | (m > 0.99)))
    if near_binary_ratio > 0.995:
        m = cv2.GaussianBlur(m, (3, 3), 0)
        m = np.clip(m, 0.0, 1.0)

    crop_fg = (m > 0.35).astype(np.uint8)
    if np.any(crop_fg):
        crop_fg = cv2.morphologyEx(
            crop_fg,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crop_fg, connectivity=8)
        if n_labels > 1:
            largest_idx = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            crop_fg = (labels == largest_idx).astype(np.uint8)

    if not np.any(crop_fg):
        crop_fg = (m > 0.03).astype(np.uint8)
        if not np.any(crop_fg):
            return None

    if fill_external_contour:
        crop_fg = _prepare_front_bumper_mask(crop_fg)
        m = crop_fg.astype(np.float32)

    ys, xs = np.where(crop_fg > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    crop_bgr = img_bgr[y1:y2, x1:x2]
    alpha_support = cv2.dilate(
        crop_fg,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    m_limited = m.copy()
    m_limited[alpha_support == 0] = 0.0
    crop_a = (m_limited[y1:y2, x1:x2] * 255.0).clip(0, 255).astype(np.uint8)

    rgba = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = crop_bgr
    rgba[:, :, 3] = crop_a
    rgba[crop_a == 0, :3] = 0
    return rgba


def _inst_get(item, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)

def _as_mask01(mask) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if torch.is_tensor(mask):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.size == 0:
        return None
    mmax = float(mask.max())
    if mmax > 1.5:
        mask = mask / 255.0
    return np.clip(mask, 0.0, 1.0)


def unwrap_instances(result) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or not len(boxes):
        return []
    if masks is None or getattr(masks, "data", None) is None:
        return []

    xyxy = getattr(boxes, "xyxy", None)
    cls = getattr(boxes, "cls", None)
    conf = getattr(boxes, "conf", None)
    if xyxy is None or cls is None or conf is None:
        return []

    names = getattr(result, "names", {}) or {}
    mask_data = masks.data

    n = int(min(int(xyxy.shape[0]), int(mask_data.shape[0])))
    out: list[dict] = []
    for i in range(n):
        cls_id = int(cls[i].item())
        name = str(names.get(cls_id, str(cls_id)))
        m = _as_mask01(mask_data[i])
        if m is None:
            continue
        box = [float(v) for v in xyxy[i].detach().float().cpu().tolist()]
        out.append(
            {
                "name": name,
                "cls_id": cls_id,
                "conf": float(conf[i].item()),
                "box_xyxy": box,
                "mask01": m.astype(np.float32, copy=False),
            }
        )
    return out


def filter_by_side(instances: list[dict], side_x_max: float, img_w: int) -> list[dict]:
    x_th = float(img_w) * float(side_x_max)
    out: list[dict] = []
    for inst in instances:
        box = _inst_get(inst, "box_xyxy")
        if box is None or len(box) != 4:
            continue
        x1, _, x2, _ = [float(v) for v in box]
        cx = (x1 + x2) / 2.0
        if cx < x_th:
            out.append(inst)
    return out


def keep_topk_by_conf(instances: list[dict], k: int) -> list[dict]:
    k = int(k)
    if k <= 0:
        return []
    return sorted(instances, key=lambda x: float(_inst_get(x, "conf") or 0.0), reverse=True)[:k]


def merge_union(instances: list[dict]) -> dict:
    if not instances:
        raise ValueError("instances is empty")

    name = str(_inst_get(instances[0], "name") or "")
    cls_id = _inst_get(instances[0], "cls_id")
    cls_id = None if cls_id is None else int(cls_id)

    confs = [float(_inst_get(x, "conf") or 0.0) for x in instances]
    conf_u = float(max(confs)) if confs else 0.0

    boxes = [_inst_get(x, "box_xyxy") for x in instances]
    boxes = [b for b in boxes if b is not None and len(b) == 4]
    if boxes:
        b = np.asarray(boxes, dtype=np.float32)
        box_u = [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())]
    else:
        box_u = None

    masks = [_as_mask01(_inst_get(x, "mask01")) for x in instances]
    masks = [m for m in masks if m is not None]
    if not masks:
        raise ValueError("instances have no valid mask01")

    target_h, target_w = int(masks[0].shape[0]), int(masks[0].shape[1])
    aligned: list[np.ndarray] = []
    for m in masks:
        if m.shape[0] != target_h or m.shape[1] != target_w:
            m = cv2.resize(m.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        aligned.append(m.astype(np.float32, copy=False))
    mask_u = np.maximum.reduce(aligned).astype(np.float32, copy=False)
    mask_u = np.clip(mask_u, 0.0, 1.0).astype(np.float32, copy=False)

    out = {"name": name, "cls_id": cls_id, "conf": conf_u, "box_xyxy": box_u, "mask01": mask_u}
    return out


def postprocess_instances(
    instances: list[dict],
    rules: dict[str, PartRule],
    side_x_max: float,
    img_w: int,
) -> list[dict]:
    normalized: list[dict] = []
    for inst in instances:
        m = _as_mask01(_inst_get(inst, "mask01"))
        if m is None:
            continue
        box = _inst_get(inst, "box_xyxy")
        if box is not None and len(box) == 4:
            box = [float(v) for v in box]
        else:
            box = None
        conf = _inst_get(inst, "conf")
        conf = None if conf is None else float(conf)
        cls_id = _inst_get(inst, "cls_id")
        cls_id = None if cls_id is None else int(cls_id)
        normalized.append(
            {
                "name": str(_inst_get(inst, "name") or ""),
                "cls_id": cls_id,
                "conf": conf,
                "box_xyxy": box,
                "mask01": m.astype(np.float32, copy=False),
            }
        )

    out: list[dict] = []
    handled_names = set(rules.keys())

    for name, rule in rules.items():
        group = [x for x in normalized if x["name"] == name]
        if not group:
            continue
        if rule.side == "left":
            group = filter_by_side(group, side_x_max=side_x_max, img_w=img_w)
        if not group:
            continue
        if rule.merge_mode == "union" and len(group) > 1:
            group = [merge_union(group)]
        group = keep_topk_by_conf(group, k=rule.max_instances)
        out.extend(group)

    for inst in normalized:
        if inst["name"] not in handled_names:
            out.append(inst)

    return out


def render_annotated_preview(img_bgr: np.ndarray, instances: Iterable, visual_label_edge: bool = False) -> np.ndarray:
    annotated = img_bgr.copy()
    occupied_label_boxes: list[tuple[int, int, int, int]] = []
    for inst in instances:
        name = str(_inst_get(inst, "name") or "")
        mask01 = _as_mask01(_inst_get(inst, "mask01"))
        if mask01 is None:
            continue
        conf = _inst_get(inst, "conf")
        box_xyxy = _inst_get(inst, "box_xyxy")

        color = _stable_color(name)
        if visual_label_edge:
            _overlay_mask(annotated, mask01, color, alpha=0.35)
        label = name if conf is None else f"{name} {float(conf):.2f}"

        anchor = _mask_anchor(mask01, annotated.shape[1], annotated.shape[0])
        if anchor is None and box_xyxy is not None:
            x1, y1, x2, y2 = [float(v) for v in box_xyxy]
            anchor = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        if anchor is None:
            anchor = (0, 0)
        if name == "front_bumper":
            h, w = annotated.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            box_w = tw + 8
            box_h = th + baseline + 8

            ax, ay = anchor
            if box_xyxy is not None:
                x1, y1, x2, y2 = [float(v) for v in box_xyxy]
                ax = int(x2)
                ay = int((y1 + y2) / 2)
            ax = max(0, min(int(ax), w - 1))
            ay = max(0, min(int(ay), h - 1))

            tx1 = int(max(0, min(w - box_w - 20, w * 0.80 - 60)))
            ty1 = max(0, min(int(ay - 50 - box_h / 2), h - box_h))
            rect = (tx1, ty1, tx1 + box_w, ty1 + box_h)
            line_end = (tx1, ty1 + box_h // 2)

            line_start = (max(0, min(w - 1, tx1 - 10)), max(0, min(h - 1, line_end[1] + 10)))
            cv2.line(annotated, line_start, (max(0, ax - 60), ay), color, 2, cv2.LINE_AA)
            cv2.rectangle(annotated, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
            cv2.putText(
                annotated,
                label,
                (tx1 + 4, ty1 + box_h - baseline - 3),
                font,
                scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
            occupied_label_boxes.append(rect)
            continue

        _draw_label(annotated, anchor, label, color, occupied_label_boxes)
    return annotated


def export_rgba_crops(img_bgr: np.ndarray, instances: Iterable, out_dir: Union[str, Path], stem: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_class_idx: dict[str, int] = {}
    for inst in instances:
        name = str(_inst_get(inst, "name") or "")
        mask01 = _as_mask01(_inst_get(inst, "mask01"))
        if mask01 is None:
            continue
        rgba = _mask_to_rgba_crop(img_bgr, mask01, fill_external_contour=(name == "front_bumper"))
        if rgba is None:
            continue

        k = per_class_idx.get(name, 0) + 1
        per_class_idx[name] = k
        out_name = f"{name}_{stem}.png" if k == 1 else f"{name}_{stem}_{k}.png"
        _imwrite_cn(out_dir / out_name, rgba)

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "--img_dir", dest="input_dir", default=str(DEFAULT_IMG_PATH))
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--conf", type=float, default=float(DEFAULT_CONF))
    parser.add_argument("--iou", type=float, default=float(DEFAULT_IOU))
    parser.add_argument("--imgsz", type=int, default=int(DEFAULT_IMGSZ))
    parser.add_argument("--side_x_max", type=float, default=float(SIDE_X_MAX))
    parser.add_argument("--output-dir", dest="output_dir", default="result")
    parser.add_argument("--label-dir", "--label_dir", dest="label_dir", default=None)
    parser.add_argument("--out-root", "--out_root", dest="out_root", default=None)
    parser.add_argument("--no_labels", action="store_true")
    parser.add_argument("--no_crops", action="store_true")
    parser.add_argument("--visual_label_edge", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    from ultralytics import YOLO

    img_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    label_dir = Path(args.label_dir) if args.label_dir else (output_dir / "front_label")
    out_root = Path(args.out_root) if args.out_root else (output_dir / "front_parts")
    save_labels = not bool(args.no_labels)
    save_crops = not bool(args.no_crops)
    if save_labels:
        label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    for p in iter_images(img_dir):
        rgb = load_rgb_with_white_bg(p)
        results = model.predict(rgb, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
        if not results:
            continue

        raw = unwrap_instances(results[0])
        processed = postprocess_instances(raw, RULES, side_x_max=args.side_x_max, img_w=int(rgb.shape[1]))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if save_labels:
            preview = render_annotated_preview(bgr.copy(), processed, visual_label_edge=bool(args.visual_label_edge))
            _imwrite_cn(label_dir / p.name, preview)

        if save_crops:
            export_rgba_crops(bgr, processed, out_root / p.stem, p.stem)


if __name__ == "__main__":
    main()
