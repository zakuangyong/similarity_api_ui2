from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import gc
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from tools import car_front_seg
from tools.cutout_by_birefnet import BiRefNetSegmenter, load_birefnet_segmenter
from tools.cdse_similarity import (
    CdseSimilarityEngine,
    FeatureName,
    load_dataset,
    merged_overall_weights,
    part_feature_weights_for,
)
from tools.contour_similarity import contour_score_and_vis
from tools.cutout_by_sam import SamModelSpec, load_sam_predictor, run_cutout_by_sam, run_sam_cutout_from_instances
from tools.car_view_cls import normalize_view_family


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


class PipelineTimer:
    def __init__(self) -> None:
        self._started = time.perf_counter()
        self.stages: list[dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str, label: str, **meta: Any) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            row: dict[str, Any] = {
                "name": name,
                "label": label,
                "seconds": round(seconds, 3),
            }
            if meta:
                row.update(meta)
            self.stages.append(row)

    def snapshot(self) -> dict[str, Any]:
        total = max(time.perf_counter() - self._started, 0.0)
        stages: list[dict[str, Any]] = []
        for idx, row in enumerate(self.stages, start=1):
            seconds = float(row.get("seconds") or 0.0)
            item = dict(row)
            item["index"] = idx
            item["percent"] = round(seconds / total * 100.0, 2) if total > 0 else 0.0
            stages.append(item)

        bottleneck = max(stages, key=lambda x: float(x.get("seconds") or 0.0), default=None)
        return {
            "total_seconds": round(total, 3),
            "stages": stages,
            "bottleneck": bottleneck,
        }


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


def _view_part_weight(config: dict[str, Any], view: str) -> Path:
    view_family = normalize_view_family(view)
    models = config.get("models") or {}
    key = f"{view_family}_part_weight"
    if models.get(key):
        return _resolve_path(models[key])
    if view_family != "front" and models.get("front_part_weight"):
        return _resolve_path(models["front_part_weight"])
    return _config_path(config, key)


def _config_parts_for_view(config: dict[str, Any], view: str) -> list[str]:
    view_family = normalize_view_family(view)
    raw = config.get(f"{view_family}_parts") or config.get("parts") or car_front_seg.get_view_parts(view_family)
    if isinstance(raw, str):
        return _parse_csv(raw)
    return [str(x).strip() for x in raw if str(x).strip()]


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
    view: str,
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
        processed = car_front_seg.detect_processed_instances(
            model=model,
            rgb=rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            view=view,
            allowed_parts=allowed_parts,
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if save_labels:
            preview = car_front_seg.render_annotated_preview(
                bgr.copy(),
                processed,
                visual_label_edge=visual_label_edge,
            )
            car_front_seg._imwrite_cn(label_dir / p.name, preview)
        car_front_seg.export_rgba_crops(bgr, processed, parts_dir / p.stem, p.stem, view=view)


def _run_part_segmentation(
    *,
    stage_dir: Path,
    label_dir: Path,
    parts_dir: Path,
    config: dict[str, Any],
    view: str,
    conf: float,
    iou: float,
    imgsz: int,
    visual_label_edge: bool,
    allowed_parts: set[str],
) -> None:
    part_weight = _view_part_weight(config, view)

    _run_yolo_part_export(
        input_dir=stage_dir,
        weight_path=part_weight,
        label_dir=label_dir,
        parts_dir=parts_dir,
        view=view,
        save_labels=True,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        visual_label_edge=visual_label_edge,
        allowed_parts=allowed_parts,
    )


def _run_part_segmentation_and_sam_cutout(
    *,
    stage_dir: Path,
    label_dir: Path,
    parts_dir: Path,
    cutout_dir: Path,
    config: dict[str, Any],
    view: str,
    conf: float,
    iou: float,
    imgsz: int,
    visual_label_edge: bool,
    allowed_parts: set[str],
    sam_checkpoint: Path,
    sam_type: str,
    device: str | None,
) -> None:
    part_weight = _view_part_weight(config, view)
    if not sam_checkpoint.is_file():
        raise FileNotFoundError(str(sam_checkpoint))

    label_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    cutout_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(part_weight))
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_predictor = load_sam_predictor(
        SamModelSpec(checkpoint=sam_checkpoint, model_type=sam_type, device=resolved_device)
    )

    for p in car_front_seg.iter_images(stage_dir):
        rgb = car_front_seg.load_rgb_with_white_bg(p)
        processed = car_front_seg.detect_processed_instances(
            model=model,
            rgb=rgb,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            view=view,
            allowed_parts=allowed_parts,
        )
        if not processed:
            continue

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        preview = car_front_seg.render_annotated_preview(
            bgr.copy(),
            processed,
            visual_label_edge=visual_label_edge,
        )
        car_front_seg._imwrite_cn(label_dir / p.name, preview)
        car_front_seg.export_rgba_crops(bgr, processed, parts_dir / p.stem, p.stem, view=view)
        run_sam_cutout_from_instances(
            rgb=rgb,
            bgr=bgr,
            instances=processed,
            output_dir=cutout_dir,
            stem=p.stem,
            sam_predictor=sam_predictor,
            view=view,
            box_margin_ratio=0.03,
            keep_largest_component=True,
        )


def _imread_cn(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _alpha_mask_from_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] < 4:
        return None
    alpha = img[:, :, 3]
    # An opaque RGBA image is still a normal-background image. Only use alpha
    # directly when the file actually contains transparent background pixels.
    if not np.any(alpha < 250) or not np.any(alpha > 0):
        return None
    return alpha > 0


def _save_image(path: Path, img: np.ndarray) -> str:
    car_front_seg._imwrite_cn(path, img)
    return str(path)


def _contour_preview(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    preview = np.zeros((*m.shape[:2], 3), dtype=np.uint8)
    preview[m > 0] = (48, 48, 48)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return preview


def _birefnet_cutout(
    *,
    segmenter: BiRefNetSegmenter,
    image_path: Path,
    output_path: Path,
    alpha_threshold: float,
) -> np.ndarray | None:
    with Image.open(image_path) as image:
        cutout = segmenter.cutout(image, alpha_threshold=alpha_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cutout.save(output_path)
    alpha = np.asarray(cutout.getchannel("A"), dtype=np.uint8)
    if not np.any(alpha > 0):
        return None
    return alpha > 0


def _run_contour_compare(
    *,
    items: list[ImageItem],
    query_item: ImageItem,
    output_dir: Path,
    config: dict[str, Any],
    device: str | None,
) -> dict[str, dict[str, Any]]:
    segmenter: BiRefNetSegmenter | None = None
    contour_cfg = config.get("contour") or {}
    birefnet_resolution = int(contour_cfg.get("birefnet_resolution", 1024))
    birefnet_alpha_threshold = float(contour_cfg.get("birefnet_alpha_threshold", 0.13))
    cutout_dir = output_dir / "vehicle_cutout"
    masks: dict[str, np.ndarray | None] = {}
    for item in items:
        mask = _alpha_mask_from_image(item.staged_path)
        if mask is None:
            if segmenter is None:
                resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                segmenter = load_birefnet_segmenter(
                    model_name=_config_path(config, "birefnet_model"),
                    device=resolved_device,
                    resolution=birefnet_resolution,
                )
            mask = _birefnet_cutout(
                segmenter=segmenter,
                image_path=item.staged_path,
                output_path=cutout_dir / f"{item.item_id}.png",
                alpha_threshold=birefnet_alpha_threshold,
            )
        masks[item.item_id] = mask
    if segmenter is not None:
        del segmenter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        scored = contour_score_and_vis(query_mask, cand_mask, contour_cfg=contour_cfg)
        if scored is None:
            out[item.item_id] = {"score": None, "diff_image": None, "status": "empty mask"}
            continue
        score, vis, query_aligned, candidate_aligned = scored
        pair_prefix = f"{query_item.item_id}_vs_{item.item_id}"
        img_path = contour_dir / f"{pair_prefix}.png"
        query_contour_path = contour_dir / f"{pair_prefix}_query_aligned_contour.png"
        candidate_contour_path = contour_dir / f"{pair_prefix}_candidate_aligned_contour.png"
        out[item.item_id] = {
            "score": score,
            "diff_image": _save_image(img_path, vis),
            "query_aligned_contour_image": _save_image(query_contour_path, _contour_preview(query_aligned)),
            "candidate_aligned_contour_image": _save_image(candidate_contour_path, _contour_preview(candidate_aligned)),
            "status": "ok",
        }
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
            "query_aligned_contour_image": contour_obj.get("query_aligned_contour_image"),
            "candidate_aligned_contour_image": contour_obj.get("candidate_aligned_contour_image"),
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
    view: str,
    view_label: str,
    output_paths: dict[str, str],
    timings: dict[str, Any],
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "view": view,
        "view_label": view_label,
        "query": {
            "id": query_item.item_id,
            "path": str(query_item.original_path),
            "staged_path": str(query_item.staged_path),
        },
        "gallery_count": sum(1 for x in items if x.role == "gallery"),
        "parts": parts,
        "features": features,
        "outputs": output_paths,
        "timings": timings,
        "results": results,
    }
    json_path = report_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# 汽车图片相似度比对报告",
        "",
        f"- 运行编号: `{run_id}`",
        f"- 车辆角度: `{view_label}` -> `{view}`",
        f"- 待比对图片 A: `{query_item.original_path}`",
        f"- 图库数量: {payload['gallery_count']}",
        f"- 比对部件: {', '.join(parts)}",
        f"- 特征算法: {', '.join(features)}",
        "",
        "## 耗时统计",
        "",
        f"- 总耗时: {float(timings.get('total_seconds') or 0.0):.3f} 秒",
    ]
    bottleneck = timings.get("bottleneck") or {}
    if bottleneck:
        md_lines.append(
            f"- 当前瓶颈: {bottleneck.get('label') or bottleneck.get('name')} "
            f"({float(bottleneck.get('seconds') or 0.0):.3f} 秒，占比 {float(bottleneck.get('percent') or 0.0):.2f}%)"
        )
    md_lines.extend(
        [
            "",
            "| 顺序 | 阶段 | 耗时(秒) | 占比 |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for row in timings.get("stages") or []:
        md_lines.append(
            f"| {int(row.get('index') or 0)} | {row.get('label') or row.get('name')} | "
            f"{float(row.get('seconds') or 0.0):.3f} | {float(row.get('percent') or 0.0):.2f}% |"
        )
    md_lines.extend(
        [
            "",
            "## 排名结果",
            "",
            "| 排名 | 图库图片 | 最终分 | 轮廓分 | 部件分 | 有效部件 |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
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
    view: str = "front",
    view_label: str | None = None,
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
    timer = PipelineTimer()
    with timer.stage("load_config", "加载配置"):
        config = _load_config(weight)
        view_family = normalize_view_family(view)
        raw_view_label = str(view_label or view or view_family).strip() or view_family
        input_dir_p = _resolve_path(input_dir)
        output_dir_p = _resolve_path(output_dir)
        output_dir_p.mkdir(parents=True, exist_ok=True)
        query_p = None if query_image is None else _resolve_path(query_image)
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    with timer.stage("prepare_images", "准备输入图片"):
        items, query_item = _prepare_run_images(
            input_dir=input_dir_p,
            query_image=query_p,
            output_dir=output_dir_p,
            run_id=run_id,
        )

    with timer.stage("prepare_outputs", "准备输出目录"):
        run_root = output_dir_p / "runs" / run_id
        label_dir = output_dir_p / f"{view_family}_label" / run_id
        parts_dir = output_dir_p / f"{view_family}_parts" / run_id
        cutout_dir = output_dir_p / "img-cutout" / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        parts_used = _parse_csv(parts) or _config_parts_for_view(config, view_family)
        ignored = _parse_csv(ignore_parts)
        allowed_parts = set(parts_used)

    if not skip_seg and not skip_cutout:
        with timer.stage(
            "part_segmentation_and_sam_cutout",
            "部件分割与 SAM 抠图",
            image_count=len(items),
            part_count=len(allowed_parts),
        ):
            _run_part_segmentation_and_sam_cutout(
                stage_dir=items[0].staged_path.parent,
                label_dir=label_dir,
                parts_dir=parts_dir,
                cutout_dir=cutout_dir,
                config=config,
                view=view_family,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                visual_label_edge=visual_label_edge,
                allowed_parts=allowed_parts,
                sam_checkpoint=_resolve_path("models/sam/sam_vit_h.pth"),
                sam_type="vit_h",
                device=("cuda:0" if device == "cuda" else "cpu") if device else None,
            )
    elif not skip_seg:
        with timer.stage(
            "part_segmentation",
            "部件分割",
            image_count=len(items),
            part_count=len(allowed_parts),
        ):
            _run_part_segmentation(
                stage_dir=items[0].staged_path.parent,
                label_dir=label_dir,
                parts_dir=parts_dir,
                config=config,
                view=view_family,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                visual_label_edge=visual_label_edge,
                allowed_parts=allowed_parts,
            )

    if skip_seg and not skip_cutout:
        with timer.stage(
            "sam_cutout",
            "SAM 抠图",
            image_count=len(items),
            part_count=len(allowed_parts),
        ):
            run_cutout_by_sam(
                stage_dir=items[0].staged_path.parent,
                output_dir=cutout_dir,
                config=config,
                allowed_parts=allowed_parts,
                sam_checkpoint=_resolve_path("models/sam/sam_vit_h.pth"),
                sam_type="vit_h",
                device=("cuda:0" if device == "cuda" else "cpu") if device else None,
                view=view_family,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                box_margin_ratio=0.03,
                keep_largest_component=True,
            )
    elif skip_cutout:
        with timer.stage("reuse_part_crops", "复用部件截图"):
            cutout_dir = parts_dir

    with timer.stage("resolve_features", "解析特征配置"):
        feature_names = _feature_list(config, features)

    with timer.stage("contour_compare", "整车轮廓对比", image_count=len(items)):
        contour = _run_contour_compare(
            items=items,
            query_item=query_item,
            output_dir=run_root,
            config=config,
            device=device,
        )

    with timer.stage(
        "part_feature_compare",
        "部件特征比对",
        candidate_count=sum(1 for x in items if x.role == "gallery"),
        compare_workers=compare_workers,
    ):
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
        with timer.stage("apply_topk", "截取 Top-K", topk=int(topk)):
            results = results[: int(topk)]

    output_paths = {
        "view": view_family,
        "view_label": raw_view_label,
        "label_dir": str(label_dir),
        "parts_dir": str(parts_dir),
        "img_cutout": str(cutout_dir),
        "vehicle_cutout": str(run_root / "vehicle_cutout"),
        "run_root": str(run_root),
    }
    output_paths[f"{view_family}_label"] = str(label_dir)
    output_paths[f"{view_family}_parts"] = str(parts_dir)
    timings = timer.snapshot()
    with timer.stage("write_reports", "写出报告"):
        report_paths = _write_reports(
            report_dir=output_dir_p / "reports",
            run_id=run_id,
            query_item=query_item,
            items=items,
            results=results,
            parts=parts_used,
            features=[str(x) for x in feature_names],
            view=view_family,
            view_label=raw_view_label,
            output_paths=output_paths,
            timings=timings,
        )
    timings = timer.snapshot()

    return {
        "run_id": run_id,
        "query_id": query_item.item_id,
        "query": str(query_item.original_path),
        "query_staged_path": str(query_item.staged_path),
        "view": view_family,
        "view_label": raw_view_label,
        "gallery_count": sum(1 for x in items if x.role == "gallery"),
        "results": results,
        "outputs": output_paths,
        "reports": report_paths,
        "timings": timings,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汽车图片相似度整合流程")
    parser.add_argument("--input-dir", default="./img/front", help="front 角度图库目录，将逐张与待比对图片 A 比较。")
    parser.add_argument("--query-image", default=None, help="待比对图片 A。页面上传时默认保存到 ./img/_uploads。")
    parser.add_argument("--weight", default=str(DEFAULT_CONFIG), help="模型/参数配置 JSON；也可直接传 front 部件 YOLO 权重。")
    parser.add_argument("--output-dir", default="./result", help="输出根目录。")
    parser.add_argument("--view", default="front", help="图库/部件视角，当前支持 front/back。")
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
        view=args.view,
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
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "reports": result["reports"],
                "timings": result["timings"],
                "top": result["results"][:3],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
