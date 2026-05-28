from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from ultralytics import YOLO

from tools import car_front_seg
from tools.cdse_similarity import (
    CdseSimilarityEngine,
    FeatureName,
    load_dataset,
    merged_overall_weights,
    part_feature_weights_for,
)
from tools.cutout_by_sam import run_cutout_by_sam


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "default_weights.json"
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
UPLOAD_DIR_NAMES = {"_uploads", "uploads", "_query", "query"}


@dataclass(frozen=True)
class ImageItem:
    item_id: str
    role: str
    original_path: Path
    staged_path: Path


def _resolve_path(path: str | Path, *, base: Path = ROOT) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _safe_name(value: str, fallback: str = "item") -> str:
    invalid = '<>:"/\\|?*'
    s = "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in value.strip())
    s = "_".join(s.split())
    s = s.strip(" ._")
    return s or fallback


def _paths_equal(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return str(a) == str(b)


def _load_config(weight: str | Path | None) -> dict[str, Any]:
    cfg_path = _resolve_path(weight or DEFAULT_CONFIG)
    if cfg_path.suffix.lower() == ".json":
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg.setdefault("models", {})["front_part_weight"] = str(cfg_path)
    return cfg


def _config_path(config: dict[str, Any], key: str) -> Path:
    value = (config.get("models") or {}).get(key)
    if not value:
        raise ValueError(f"配置缺少 models.{key}")
    return _resolve_path(value)


def _parse_csv(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def _collect_gallery_images(input_dir: Path, query_path: Path | None = None) -> list[Path]:
    images: list[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if query_path is not None and _paths_equal(p, query_path):
            continue
        rel_parts = {part.lower() for part in p.relative_to(input_dir).parts[:-1]}
        if rel_parts & UPLOAD_DIR_NAMES:
            continue
        images.append(p)
    return images


def _prepare_run_images(
    *,
    input_dir: Path,
    query_image: Path | None,
    output_dir: Path,
    run_id: str,
) -> tuple[list[ImageItem], ImageItem]:
    if not input_dir.is_dir():
        raise ValueError(f"front 图库目录不存在: {input_dir}")

    if query_image is None:
        raise ValueError("请通过 --query-image 指定待比对图片 A。")
    else:
        query_path = query_image.resolve()
        if not query_path.is_file():
            raise ValueError(f"待比对图片 A 不存在: {query_path}")
        if query_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"待比对图片 A 格式不支持: {query_path.suffix}")

    gallery = _collect_gallery_images(input_dir, query_path=query_path)
    if not gallery:
        raise ValueError(f"front 图库目录中没有可比对图片: {input_dir}")

    stage_dir = output_dir / "work" / run_id / "input_flat"
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged: list[ImageItem] = []
    all_sources = [("query", query_path), *[("gallery", p.resolve()) for p in gallery]]
    used_ids: set[str] = set()
    for idx, (role, src) in enumerate(all_sources):
        base = _safe_name(src.stem, fallback=f"image_{idx:04d}")
        item_id = base
        n = 1
        while item_id in used_ids:
            n += 1
            item_id = f"{base}_{n}"
        used_ids.add(item_id)
        staged_path = stage_dir / f"{item_id}{src.suffix.lower()}"
        shutil.copy2(src, staged_path)
        staged.append(ImageItem(item_id=item_id, role=role, original_path=src, staged_path=staged_path))

    query_item = next(x for x in staged if x.role == "query")
    return staged, query_item


def _run_yolo_part_export(
    *,
    input_dir: Path,
    weight_path: Path,
    label_dir: Path,
    parts_dir: Path,
    save_labels: bool,
    conf: float,
    iou: float,
    imgsz: int,
    visual_label_edge: bool,
    allowed_parts: set[str] | None = None,
) -> None:
    model = YOLO(str(weight_path))
    if save_labels:
        label_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    for p in car_front_seg.iter_images(input_dir):
        rgb = car_front_seg.load_rgb_with_white_bg(p)
        results = model.predict(rgb, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        if not results:
            continue
        raw = car_front_seg.unwrap_instances(results[0])
        for inst in raw:
            name = str(car_front_seg._inst_get(inst, "name") or "")
            inst["name"] = "hood" if name == "front_bumper" else name
        processed = car_front_seg.postprocess_instances(
            raw,
            car_front_seg.RULES,
            side_x_max=car_front_seg.SIDE_X_MAX,
            img_w=int(rgb.shape[1]),
        )
        if allowed_parts is not None:
            processed = [x for x in processed if str(car_front_seg._inst_get(x, "name") or "") in allowed_parts]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if save_labels:
            preview = car_front_seg.render_annotated_preview(
                bgr.copy(),
                processed,
                visual_label_edge=visual_label_edge,
            )
            car_front_seg._imwrite_cn(label_dir / p.name, preview)
        car_front_seg.export_rgba_crops(bgr, processed, parts_dir / p.stem, p.stem)


def _run_part_segmentation(
    *,
    stage_dir: Path,
    label_dir: Path,
    parts_dir: Path,
    config: dict[str, Any],
    conf: float,
    iou: float,
    imgsz: int,
    visual_label_edge: bool,
    allowed_parts: set[str],
) -> None:
    carpart_weight = _config_path(config, "carpart_weight")
    front_weight = _config_path(config, "front_part_weight")

    _run_yolo_part_export(
        input_dir=stage_dir,
        weight_path=carpart_weight,
        label_dir=label_dir,
        parts_dir=parts_dir,
        save_labels=False,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        visual_label_edge=False,
        allowed_parts=allowed_parts,
    )
    _run_yolo_part_export(
        input_dir=stage_dir,
        weight_path=front_weight,
        label_dir=label_dir,
        parts_dir=parts_dir,
        save_labels=True,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        visual_label_edge=visual_label_edge,
        allowed_parts=allowed_parts,
    )


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _imread_cn(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _save_image(path: Path, img: np.ndarray) -> str:
    car_front_seg._imwrite_cn(path, img)
    return str(path)


def _largest_car_mask(model: YOLO, image_bgr: np.ndarray) -> np.ndarray | None:
    def pick(results: list[Any] | None) -> np.ndarray | None:
        if not results or results[0].masks is None or len(results[0].boxes) == 0:
            return None
        masks = results[0].masks.data.detach().cpu().numpy()
        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        mask = masks[idx]
        h, w = image_bgr.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    results = model.predict(image_bgr, classes=[2, 3, 5, 7], conf=0.2, verbose=False)
    mask = pick(results)
    if mask is None:
        mask = pick(model.predict(image_bgr, conf=0.2, verbose=False))
    if mask is None:
        return None
    return mask > 0.5


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]


def _contour_score_and_vis(query_mask: np.ndarray, candidate_mask: np.ndarray) -> tuple[float, np.ndarray] | None:
    q = _crop_mask(query_mask)
    c = _crop_mask(candidate_mask)
    if q is None or c is None:
        return None
    c_resized = cv2.resize(c.astype(np.uint8), (q.shape[1], q.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    q_bool = q > 0
    union = np.logical_or(q_bool, c_resized)
    if not np.any(union):
        return None
    overlap = np.logical_and(q_bool, c_resized)
    score = float(np.sum(overlap) / np.sum(union) * 100.0)

    vis = np.zeros((q.shape[0], q.shape[1], 3), dtype=np.uint8)
    vis[q_bool] = [0, 0, 255]
    vis[c_resized] = [0, 255, 0]
    vis[overlap] = [0, 255, 255]
    return round(max(0.0, min(99.0, score)), 1), vis


def _run_contour_compare(
    *,
    items: list[ImageItem],
    query_item: ImageItem,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    model = YOLO(str(_config_path(config, "contour_weight")))
    masks: dict[str, np.ndarray | None] = {}
    for item in items:
        img = _imread_cn(item.staged_path)
        masks[item.item_id] = None if img is None else _largest_car_mask(model, img)

    out: dict[str, dict[str, Any]] = {}
    contour_dir = output_dir / "contour"
    contour_dir.mkdir(parents=True, exist_ok=True)
    query_mask = masks.get(query_item.item_id)
    for item in items:
        if item.role != "gallery":
            continue
        cand_mask = masks.get(item.item_id)
        if query_mask is None or cand_mask is None:
            out[item.item_id] = {"score": None, "diff_image": None, "status": "car mask missing"}
            continue
        scored = _contour_score_and_vis(query_mask, cand_mask)
        if scored is None:
            out[item.item_id] = {"score": None, "diff_image": None, "status": "empty mask"}
            continue
        score, vis = scored
        img_path = contour_dir / f"{query_item.item_id}_vs_{item.item_id}.png"
        out[item.item_id] = {"score": score, "diff_image": _save_image(img_path, vis), "status": "ok"}
    return out


def _feature_list(config: dict[str, Any], features: str | Iterable[str] | None) -> list[FeatureName]:
    raw = _parse_csv(features if features is not None else config.get("features", "dino,ssim,edge"))
    allowed = {"clip", "dino", "ssim", "edge"}
    bad = [x for x in raw if x not in allowed]
    if bad:
        raise ValueError(f"未知特征: {','.join(bad)}")
    return raw or ["dino", "ssim", "edge"]  # type: ignore[return-value]


def _part_matrices(config: dict[str, Any]) -> tuple[dict[str, dict[FeatureName, float]], dict[str, float]]:
    feature = config.get("part_feature_weights") or {}
    overall = config.get("part_overall_weights") or {}
    return feature, {str(k): float(v) for k, v in overall.items()}  # type: ignore[return-value]


def _score_color_text(score: float | None) -> str:
    if score is None or math.isnan(float(score)):
        return "无有效评分"
    if score >= 90:
        return "高度相似"
    if score >= 75:
        return "较高相似"
    if score >= 60:
        return "局部相似"
    return "差异明显"


def _analysis_for_item(item: dict[str, Any]) -> list[str]:
    points = [
        f"最终评分 {item['final_score']:.1f} 分，判定为{_score_color_text(item['final_score'])}。",
        f"整车轮廓相似度: {item.get('contour_score') if item.get('contour_score') is not None else '未计算'}。",
    ]
    part_scores = item.get("part_scores") or {}
    if part_scores:
        ordered = sorted(part_scores.items(), key=lambda x: float(x[1].get("fused") or 0.0))
        low_name, low_obj = ordered[0]
        high_name, high_obj = ordered[-1]
        points.append(f"共对比 {len(part_scores)} 个有效部件。差异较大的部件是 {low_name} ({float(low_obj.get('fused') or 0.0):.1f} 分)。")
        if high_name != low_name:
            points.append(f"相似度最高的部件是 {high_name} ({float(high_obj.get('fused') or 0.0):.1f} 分)。")
    else:
        points.append("未识别到可横向比对的共同部件，本次结果主要依据整车轮廓。")
    return points


def _compare_parts(
    *,
    cutout_dir: Path,
    items: list[ImageItem],
    query_item: ImageItem,
    contour: dict[str, dict[str, Any]],
    config: dict[str, Any],
    parts: list[str],
    ignored_parts: list[str],
    features: list[FeatureName],
    device: str | None,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    active_parts = [p for p in parts if p not in set(ignored_parts)]
    dataset = load_dataset(str(cutout_dir), active_parts)
    part_feature_matrix, part_overall_base = _part_matrices(config)
    model_dir = _config_path(config, "cdse_model_dir")
    clip_dir = _config_path(config, "clip_dir")
    dino_repo = _config_path(config, "dino_repo")
    dino_weights = _config_path(config, "dino_weights")

    engine = CdseSimilarityEngine(
        model_dir=model_dir,
        clip_dir=clip_dir,
        dino_repo_dir=dino_repo,
        dino_weights=dino_weights,
        device=device,
    )
    enabled_features = engine._resolve_features(features, require_clip=False)
    score_weights = config.get("score_weights") or {"contour": 0.4, "parts": 0.6}

    items_by_id = {x.item_id: x for x in items}
    results: list[dict[str, Any]] = []
    query_parts = dataset.get(query_item.item_id, {})

    gallery_items = [item for item in items if item.role == "gallery"]
    if max_workers is not None:
        worker_count = int(max_workers)
    else:
        worker_count = min(8, max(1, (os.cpu_count() or 2) // 2), max(1, len(gallery_items)))

    def compare_item(item: ImageItem) -> dict[str, Any]:
        candidate_parts = dataset.get(item.item_id, {})
        per_part: dict[str, Any] = {}
        part_fused: dict[str, float] = {}
        for part in active_parts:
            q_path = query_parts.get(part)
            c_path = candidate_parts.get(part)
            if not q_path or not c_path:
                continue
            pw = part_feature_weights_for(part, enabled=enabled_features, matrix=part_feature_matrix)
            detail = engine.compare_paths(q_path, c_path, features=enabled_features, weights=pw, auto_renorm=False)
            per_part[part] = {
                "query_path": q_path,
                "candidate_path": c_path,
                "weights_used": pw,
                **detail.to_dict(),
            }
            part_fused[part] = float(detail.fused)

        part_score: float | None = None
        overall_weights_used: dict[str, float] = {}
        if part_fused:
            overall_weights_used = merged_overall_weights(list(part_fused.keys()), part_overall_base)
            if overall_weights_used:
                part_score = float(sum(part_fused[p] * float(overall_weights_used.get(p, 0.0)) for p in part_fused))
            else:
                part_score = float(np.mean(list(part_fused.values())))

        contour_obj = contour.get(item.item_id, {})
        contour_score = contour_obj.get("score")
        final_parts: dict[str, float] = {}
        if contour_score is not None:
            final_parts["contour"] = float(score_weights.get("contour", 0.4))
        if part_score is not None:
            final_parts["parts"] = float(score_weights.get("parts", 0.6))
        weight_sum = sum(max(0.0, x) for x in final_parts.values())
        if weight_sum <= 0:
            final_score = 0.0
        else:
            final_score = 0.0
            if contour_score is not None:
                final_score += float(contour_score) * max(0.0, final_parts.get("contour", 0.0)) / weight_sum
            if part_score is not None:
                final_score += float(part_score) * max(0.0, final_parts.get("parts", 0.0)) / weight_sum
        final_score = round(max(0.0, min(99.0, final_score)), 1)

        row = {
            "candidate_id": item.item_id,
            "candidate_path": str(items_by_id[item.item_id].original_path),
            "final_score": final_score,
            "contour_score": contour_score,
            "part_score": None if part_score is None else round(part_score, 1),
            "part_scores": per_part,
            "ignored_parts": ignored_parts,
            "missing_parts": [p for p in active_parts if not query_parts.get(p) or not candidate_parts.get(p)],
            "overall_part_weights_used": overall_weights_used,
            "final_score_weights_used": final_parts,
            "contour_diff_image": contour_obj.get("diff_image"),
        }
        row["analysis"] = _analysis_for_item(row)
        return row

    if worker_count <= 1 or len(gallery_items) <= 1:
        results = [compare_item(item) for item in gallery_items]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(compare_item, item): item for item in gallery_items}
            for future in as_completed(future_map):
                results.append(future.result())

    return sorted(results, key=lambda x: float(x["final_score"]), reverse=True)


def _write_reports(
    *,
    report_dir: Path,
    run_id: str,
    query_item: ImageItem,
    items: list[ImageItem],
    results: list[dict[str, Any]],
    parts: list[str],
    features: list[str],
    output_paths: dict[str, str],
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "query": {
            "id": query_item.item_id,
            "path": str(query_item.original_path),
            "staged_path": str(query_item.staged_path),
        },
        "gallery_count": sum(1 for x in items if x.role == "gallery"),
        "parts": parts,
        "features": features,
        "outputs": output_paths,
        "results": results,
    }
    json_path = report_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# 汽车图片相似度比对报告",
        "",
        f"- 运行编号: `{run_id}`",
        f"- 待比对图片 A: `{query_item.original_path}`",
        f"- 图库数量: {payload['gallery_count']}",
        f"- 比对部件: {', '.join(parts)}",
        f"- 特征算法: {', '.join(features)}",
        "",
        "## 排名结果",
        "",
        "| 排名 | 图库图片 | 最终分 | 轮廓分 | 部件分 | 有效部件 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(results, start=1):
        parts_used = ", ".join((row.get("part_scores") or {}).keys()) or "-"
        contour_score = row.get("contour_score")
        part_score = row.get("part_score")
        md_lines.append(
            f"| {idx} | `{row['candidate_id']}` | {row['final_score']:.1f} | "
            f"{'-' if contour_score is None else f'{float(contour_score):.1f}'} | "
            f"{'-' if part_score is None else f'{float(part_score):.1f}'} | {parts_used} |"
        )
    md_lines.extend(["", "## 评判分析", ""])
    for idx, row in enumerate(results[:10], start=1):
        md_lines.append(f"### Top {idx}: {row['candidate_id']}")
        for point in row.get("analysis") or []:
            md_lines.append(f"- {point}")
        if row.get("missing_parts"):
            md_lines.append(f"- 缺失或未参与计算的部件: {', '.join(row['missing_parts'])}")
        md_lines.append("")

    md_path = report_dir / f"{run_id}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    latest_json = report_dir.parent / "latest_report.json"
    latest_md = report_dir.parent / "latest_report.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "latest_json": str(latest_json), "latest_markdown": str(latest_md)}


def run_pipeline(
    *,
    input_dir: str | Path = ROOT / "img",
    query_image: str | Path | None,
    weight: str | Path | None = DEFAULT_CONFIG,
    output_dir: str | Path = ROOT / "result",
    parts: str | Iterable[str] | None = None,
    ignore_parts: str | Iterable[str] | None = None,
    features: str | Iterable[str] | None = None,
    topk: int | None = None,
    device: str | None = None,
    skip_seg: bool = False,
    skip_cutout: bool = False,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    visual_label_edge: bool = False,
    compare_workers: int | None = 4,
) -> dict[str, Any]:
    config = _load_config(weight)
    input_dir_p = _resolve_path(input_dir)
    output_dir_p = _resolve_path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    query_p = None if query_image is None else _resolve_path(query_image)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    items, query_item = _prepare_run_images(
        input_dir=input_dir_p,
        query_image=query_p,
        output_dir=output_dir_p,
        run_id=run_id,
    )

    run_root = output_dir_p / "runs" / run_id
    label_dir = output_dir_p / "front_label" / run_id
    parts_dir = output_dir_p / "front_parts" / run_id
    cutout_dir = output_dir_p / "img-cutout" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    parts_used = _parse_csv(parts) or list(config.get("parts") or [])
    ignored = _parse_csv(ignore_parts)
    allowed_parts = set(parts_used)

    if not skip_seg:
        _run_part_segmentation(
            stage_dir=items[0].staged_path.parent,
            label_dir=label_dir,
            parts_dir=parts_dir,
            config=config,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            visual_label_edge=visual_label_edge,
            allowed_parts=allowed_parts,
        )

    if not skip_cutout:
        if _count_images(parts_dir) <= 0:
            raise RuntimeError(f"未产生部件截图，无法进入 SAM 主体识别: {parts_dir}")
        run_cutout_by_sam(
            stage_dir=items[0].staged_path.parent,
            output_dir=cutout_dir,
            config=config,
            allowed_parts=allowed_parts,
            sam_checkpoint=_resolve_path("models/sam/sam_vit_h.pth"),
            sam_type="vit_h",
            device=("cuda:0" if device == "cuda" else "cpu") if device else None,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            box_margin_ratio=0.03,
            keep_largest_component=True,
        )
    else:
        cutout_dir = parts_dir

    feature_names = _feature_list(config, features)
    contour = _run_contour_compare(items=items, query_item=query_item, output_dir=run_root, config=config)
    results = _compare_parts(
        cutout_dir=cutout_dir,
        items=items,
        query_item=query_item,
        contour=contour,
        config=config,
        parts=parts_used,
        ignored_parts=ignored,
        features=feature_names,
        device=device,
        max_workers=compare_workers,
    )
    if topk and topk > 0:
        results = results[: int(topk)]

    output_paths = {
        "front_label": str(label_dir),
        "front_parts": str(parts_dir),
        "img_cutout": str(cutout_dir),
        "run_root": str(run_root),
    }
    report_paths = _write_reports(
        report_dir=output_dir_p / "reports",
        run_id=run_id,
        query_item=query_item,
        items=items,
        results=results,
        parts=parts_used,
        features=[str(x) for x in feature_names],
        output_paths=output_paths,
    )

    return {
        "run_id": run_id,
        "query_id": query_item.item_id,
        "query": str(query_item.original_path),
        "query_staged_path": str(query_item.staged_path),
        "gallery_count": sum(1 for x in items if x.role == "gallery"),
        "results": results,
        "outputs": output_paths,
        "reports": report_paths,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汽车图片相似度整合流程")
    parser.add_argument("--input-dir", default="./img/front", help="front 角度图库目录，将逐张与待比对图片 A 比较。")
    parser.add_argument("--query-image", default=None, help="待比对图片 A。页面上传时默认保存到 ./img/_uploads。")
    parser.add_argument("--weight", default=str(DEFAULT_CONFIG), help="模型/参数配置 JSON；也可直接传 front 部件 YOLO 权重。")
    parser.add_argument("--output-dir", default="./result", help="输出根目录。")
    parser.add_argument("--parts", default=None, help="参与横向比对的部件，逗号分隔。")
    parser.add_argument("--ignore-parts", default="", help="忽略计算的部件，逗号分隔。")
    parser.add_argument("--features", default=None, help="CDSE 特征，默认取配置: dino,ssim,edge。")
    parser.add_argument("--enable-clip", action="store_true", help="在 CDSE 中启用 CLIP。")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--skip-seg", action="store_true")
    parser.add_argument("--skip-cutout", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--visual-label-edge", action="store_true")
    parser.add_argument("--compare-workers", type=int, default=4, help="候选图库相似度计算线程数，默认 4。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    features = args.features
    if args.enable_clip:
        current = _parse_csv(features) or _parse_csv(_load_config(args.weight).get("features"))
        if "clip" not in current:
            current = ["clip", *current]
        features = ",".join(current)

    result = run_pipeline(
        input_dir=args.input_dir,
        query_image=args.query_image,
        weight=args.weight,
        output_dir=args.output_dir,
        parts=args.parts,
        ignore_parts=args.ignore_parts,
        features=features,
        topk=args.topk,
        device=args.device,
        skip_seg=bool(args.skip_seg),
        skip_cutout=bool(args.skip_cutout),
        conf=float(args.conf),
        iou=float(args.iou),
        imgsz=int(args.imgsz),
        visual_label_edge=bool(args.visual_label_edge),
        compare_workers=args.compare_workers,
    )
    print(json.dumps({"run_id": result["run_id"], "reports": result["reports"], "top": result["results"][:3]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
