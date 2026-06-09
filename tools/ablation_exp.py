from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from similarity_pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    IMAGE_EXTS,
    ROOT,
    _alpha_mask_from_image,
    _config_path,
    _load_config,
    _safe_name,
)
from tools.cutout_by_birefnet import BiRefNetSegmenter, load_birefnet_segmenter  # noqa: E402
from tools.contour_similarity import (  # noqa: E402
    _bottom_region_mask,
    _crop_mask,
    _edge_band,
    _iou_percent,
    _mask_centroid_xy,
    _resize_keep_aspect_pad,
    _shift_mask,
)


DEFAULT_INPUT_DIR = Path(r"C:\Users\Lenovo\Desktop\img-test")
DEFAULT_OUTPUT_DIR = ROOT / "result" / "ablation"
EXPERIMENTS = (
    ("E0_baseline", "Baseline"),
    ("E1_no_fill_iou", "No Fill IoU"),
    ("E2_no_aspect", "No Aspect Penalty"),
    ("E3_no_gamma", "No Gamma Calibration"),
)


def _resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _collect_images(input_dir: Path, limit: int | None) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"input dir not found: {input_dir}")
    images = sorted(
        p.resolve()
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if limit is not None:
        images = images[: max(0, int(limit))]
    return images


def _unique_ids(images: list[Path]) -> dict[Path, str]:
    used: set[str] = set()
    out: dict[Path, str] = {}
    for index, image in enumerate(images, start=1):
        base = _safe_name(image.stem, fallback=f"image_{index:04d}")
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        out[image] = candidate
    return out


def _extract_masks(
    images: list[Path],
    ids: dict[Path, str],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    masks: dict[str, np.ndarray] = {}
    failures: list[dict[str, str]] = []
    segmenter: BiRefNetSegmenter | None = None
    contour_cfg = config.get("contour") or {}
    resolution = int(contour_cfg.get("birefnet_resolution", 1024))
    alpha_threshold = float(contour_cfg.get("birefnet_alpha_threshold", 0.13))
    for index, image in enumerate(images, start=1):
        image_id = ids[image]
        mask = _alpha_mask_from_image(image)
        source = "alpha"
        if mask is None:
            source = "birefnet"
            if segmenter is None:
                segmenter = load_birefnet_segmenter(
                    model_name=_config_path(config, "birefnet_model"),
                    device="auto",
                    resolution=resolution,
                )
            with Image.open(image) as pil_image:
                alpha = segmenter.predict_alpha(pil_image.convert("RGB"), alpha_threshold=alpha_threshold)
            alpha_array = np.asarray(alpha, dtype=np.uint8)
            mask = alpha_array > 0 if np.any(alpha_array > 0) else None
        if mask is None:
            failures.append({"image_id": image_id, "path": str(image), "error": "mask missing"})
            print(f"[mask {index}/{len(images)}] failed {image_id}", file=sys.stderr)
            continue
        masks[image_id] = mask.astype(bool)
        print(f"[mask {index}/{len(images)}] ok {image_id} source={source}")
    if segmenter is not None:
        del segmenter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return masks, failures


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _gamma_score(value: float, gamma: float) -> float:
    capped = _clip(value)
    if gamma <= 0 or abs(gamma - 1.0) <= 1e-6:
        return capped
    return 100.0 * ((capped / 100.0) ** gamma)


def _align_masks(
    query_mask: np.ndarray,
    candidate_mask: np.ndarray,
    contour_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    query = _crop_mask(query_mask)
    candidate = _crop_mask(candidate_mask)
    if query is None or candidate is None:
        return None

    query_bool = query > 0
    if bool(contour_cfg.get("keep_aspect", True)):
        aligned = _resize_keep_aspect_pad(candidate, query_bool.shape)
    else:
        aligned = cv2.resize(
            candidate.astype(np.uint8),
            (query_bool.shape[1], query_bool.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    if str(contour_cfg.get("align", "centroid")).strip().lower() == "centroid":
        query_center = _mask_centroid_xy(query_bool)
        candidate_center = _mask_centroid_xy(aligned)
        if query_center is not None and candidate_center is not None:
            aligned = _shift_mask(
                aligned,
                dx=query_center[0] - candidate_center[0],
                dy=query_center[1] - candidate_center[1],
            )
    return query, candidate, query_bool, aligned


def _pair_components(
    query_mask: np.ndarray,
    candidate_mask: np.ndarray,
    contour_cfg: dict[str, Any],
) -> dict[str, float] | None:
    aligned_pack = _align_masks(query_mask, candidate_mask, contour_cfg)
    if aligned_pack is None:
        return None
    query, candidate, query_bool, aligned = aligned_pack

    fill_iou = _iou_percent(query_bool, aligned)
    if fill_iou is None:
        return None

    edge_thickness = int(contour_cfg.get("edge_thickness", 4))
    bottom_thickness = int(contour_cfg.get("bottom_edge_thickness", edge_thickness))
    bottom_start = float(contour_cfg.get("bottom_region_y", 1.0))
    bottom_weight = _clip(float(contour_cfg.get("bottom_edge_weight", 0.0)), 0.0, 1.0)

    bottom_region = _bottom_region_mask(query_bool.shape, start_ratio=bottom_start)
    top_region = ~bottom_region
    query_top = np.logical_and(_edge_band(query_bool, thickness=edge_thickness), top_region)
    candidate_top = np.logical_and(_edge_band(aligned, thickness=edge_thickness), top_region)
    query_bottom = np.logical_and(_edge_band(query_bool, thickness=bottom_thickness), bottom_region)
    candidate_bottom = np.logical_and(_edge_band(aligned, thickness=bottom_thickness), bottom_region)

    top_edge_iou = _iou_percent(query_top, candidate_top)
    bottom_edge_iou = _iou_percent(query_bottom, candidate_bottom)
    if top_edge_iou is None and bottom_edge_iou is None:
        edge_score = fill_iou
    elif top_edge_iou is None:
        edge_score = float(bottom_edge_iou)
    elif bottom_edge_iou is None:
        edge_score = float(top_edge_iou)
    else:
        edge_score = (1.0 - bottom_weight) * top_edge_iou + bottom_weight * bottom_edge_iou

    aspect_k = float(contour_cfg.get("aspect_penalty_k", 1.2))
    query_ar = float(query.shape[0]) / max(1.0, float(query.shape[1]))
    candidate_ar = float(candidate.shape[0]) / max(1.0, float(candidate.shape[1]))
    aspect_delta = abs(math.log(max(1e-6, candidate_ar) / max(1e-6, query_ar)))
    aspect_penalty = math.exp(-aspect_k * aspect_delta) if aspect_k > 0 else 1.0

    edge_weight = _clip(float(contour_cfg.get("edge_weight", 0.6)), 0.0, 1.0)
    base_score = (1.0 - edge_weight) * fill_iou + edge_weight * edge_score
    score_scale = float(contour_cfg.get("score_scale", 1.0))
    scale = score_scale if score_scale > 0 else 1.0
    gamma = float(contour_cfg.get("calibration_gamma", 1.0))

    baseline_before_gamma = scale * aspect_penalty * base_score
    no_fill_before_gamma = scale * aspect_penalty * edge_score
    no_aspect_before_gamma = scale * base_score

    return {
        "fill_iou": float(fill_iou),
        "top_edge_iou": float("nan") if top_edge_iou is None else float(top_edge_iou),
        "bottom_edge_iou": float("nan") if bottom_edge_iou is None else float(bottom_edge_iou),
        "edge_score": float(edge_score),
        "query_aspect_ratio": query_ar,
        "candidate_aspect_ratio": candidate_ar,
        "aspect_delta": aspect_delta,
        "aspect_penalty": aspect_penalty,
        "baseline_before_gamma": baseline_before_gamma,
        "E0_baseline": round(_clip(_gamma_score(baseline_before_gamma, gamma)), 1),
        "E1_no_fill_iou": round(_clip(_gamma_score(no_fill_before_gamma, gamma)), 1),
        "E2_no_aspect": round(_clip(_gamma_score(no_aspect_before_gamma, gamma)), 1),
        "E3_no_gamma": round(_clip(baseline_before_gamma), 1),
    }


def _read_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        return np.zeros((320, 480, 3), dtype=np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        background = np.full_like(bgr, 28.0)
        return (bgr * alpha + background * (1.0 - alpha)).astype(np.uint8)
    return image[:, :, :3]


def _fit_panel(image: np.ndarray, width: int, height: int, background: tuple[int, int, int] = (24, 28, 34)) -> np.ndarray:
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return canvas
    scale = min(width / float(w), height / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _contour_diff_image(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    vis = np.zeros((*query.shape[:2], 3), dtype=np.uint8)
    vis[query] = (0, 0, 255)
    vis[candidate] = (0, 255, 0)
    vis[np.logical_and(query, candidate)] = (0, 255, 255)
    return vis


def _put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.65,
    color: tuple[int, int, int] = (232, 238, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _score_color(score: float) -> tuple[int, int, int]:
    if score >= 85:
        return (72, 210, 126)
    if score >= 70:
        return (70, 190, 240)
    return (90, 120, 245)


def _render_pair_visual(
    *,
    query_path: Path,
    candidate_path: Path,
    query_id: str,
    candidate_id: str,
    query_mask: np.ndarray,
    candidate_mask: np.ndarray,
    row: dict[str, Any],
    contour_cfg: dict[str, Any],
    output_path: Path,
) -> None:
    aligned_pack = _align_masks(query_mask, candidate_mask, contour_cfg)
    if aligned_pack is None:
        return
    _, _, query_aligned, candidate_aligned = aligned_pack

    width = 1540
    height = 760
    canvas = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)
    _put_text(canvas, f"Contour Ablation: {query_id} vs {candidate_id}", (34, 46), scale=0.9, thickness=2)

    panel_w = 430
    panel_h = 300
    query_panel = _fit_panel(_read_image_bgr(query_path), panel_w, panel_h)
    candidate_panel = _fit_panel(_read_image_bgr(candidate_path), panel_w, panel_h)
    diff_panel = _fit_panel(_contour_diff_image(query_aligned, candidate_aligned), panel_w, panel_h)
    canvas[82 : 82 + panel_h, 34 : 34 + panel_w] = query_panel
    canvas[82 : 82 + panel_h, 484 : 484 + panel_w] = candidate_panel
    canvas[82 : 82 + panel_h, 934 : 934 + panel_w] = diff_panel
    _put_text(canvas, "Query", (34, 406))
    _put_text(canvas, "Candidate", (484, 406))
    _put_text(canvas, "Aligned contour: red=query, green=candidate, yellow=overlap", (934, 406), scale=0.48)

    metrics = [
        ("Fill IoU", float(row["fill_iou"])),
        ("Top edge IoU", float(row["top_edge_iou"])),
        ("Bottom edge IoU", float(row["bottom_edge_iou"])),
        ("Aspect penalty", float(row["aspect_penalty"]) * 100.0),
    ]
    x = 34
    for label, value in metrics:
        cv2.rectangle(canvas, (x, 445), (x + 250, 535), (35, 43, 53), -1)
        _put_text(canvas, label, (x + 16, 474), scale=0.55, color=(172, 185, 200))
        _put_text(canvas, f"{value:.1f}", (x + 16, 516), scale=1.0, color=(235, 241, 248), thickness=2)
        x += 270

    experiment_labels = [
        ("E0 Baseline", "E0_baseline"),
        ("E1 No Fill IoU", "E1_no_fill_iou"),
        ("E2 No Aspect", "E2_no_aspect"),
        ("E3 No Gamma", "E3_no_gamma"),
    ]
    x = 34
    for label, key in experiment_labels:
        score = float(row[key])
        cv2.rectangle(canvas, (x, 565), (x + 340, 715), (30, 37, 46), -1)
        _put_text(canvas, label, (x + 18, 600), scale=0.62)
        _put_text(canvas, f"{score:.1f}", (x + 18, 658), scale=1.45, color=_score_color(score), thickness=3)
        rank = row.get(f"{key}_rank")
        _put_text(canvas, f"rank: {'-' if rank is None else rank}", (x + 190, 657), scale=0.58, color=(175, 188, 204))
        x += 365

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if ok:
        encoded.tofile(str(output_path))


def _render_summary_chart(summaries: dict[str, Any], output_path: Path) -> None:
    width = 1200
    height = 700
    canvas = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)
    _put_text(canvas, "Whole-car Contour Ablation Summary", (42, 58), scale=1.0, thickness=2)
    chart_left, chart_top, chart_right, chart_bottom = 90, 120, 1140, 590
    cv2.line(canvas, (chart_left, chart_bottom), (chart_right, chart_bottom), (110, 122, 136), 2)
    cv2.line(canvas, (chart_left, chart_top), (chart_left, chart_bottom), (110, 122, 136), 2)
    for tick in range(0, 101, 20):
        y = int(chart_bottom - (chart_bottom - chart_top) * tick / 100.0)
        cv2.line(canvas, (chart_left, y), (chart_right, y), (42, 49, 59), 1)
        _put_text(canvas, str(tick), (42, y + 6), scale=0.5, color=(155, 168, 184))

    bar_width = 150
    gap = 100
    x = chart_left + 80
    colors = [(72, 210, 126), (90, 120, 245), (70, 190, 240), (190, 140, 230)]
    for index, (experiment, title) in enumerate(EXPERIMENTS):
        mean = float(summaries[experiment].get("mean_score") or 0.0)
        std = float(summaries[experiment].get("std_score") or 0.0)
        y = int(chart_bottom - (chart_bottom - chart_top) * mean / 100.0)
        cv2.rectangle(canvas, (x, y), (x + bar_width, chart_bottom), colors[index], -1)
        _put_text(canvas, f"{mean:.2f}", (x + 28, y - 14), scale=0.6)
        _put_text(canvas, f"std {std:.2f}", (x + 18, chart_bottom + 34), scale=0.48, color=(170, 184, 200))
        label = title.replace(" Penalty", "").replace(" Calibration", "")
        _put_text(canvas, label, (x - 4, chart_bottom + 68), scale=0.48)
        x += bar_width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if ok:
        encoded.tofile(str(output_path))


def _score_query(
    query_id: str,
    image_ids: list[str],
    masks: dict[str, np.ndarray],
    contour_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_mask = masks[query_id]
    for candidate_id in image_ids:
        components = _pair_components(query_mask, masks[candidate_id], contour_cfg)
        if components is None:
            continue
        rows.append(
            {
                "query_id": query_id,
                "candidate_id": candidate_id,
                "is_self": query_id == candidate_id,
                **components,
            }
        )
    return rows


def _assign_ranks(rows: list[dict[str, Any]], include_self: bool) -> None:
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_id"]), []).append(row)
    for query_rows in by_query.values():
        for experiment, _ in EXPERIMENTS:
            ranked = sorted(
                (r for r in query_rows if include_self or not bool(r["is_self"])),
                key=lambda r: (-float(r[experiment]), str(r["candidate_id"])),
            )
            for rank, row in enumerate(ranked, start=1):
                row[f"{experiment}_rank"] = rank
        for row in query_rows:
            if bool(row["is_self"]) and not include_self:
                for experiment, _ in EXPERIMENTS:
                    row[f"{experiment}_rank"] = None


def _load_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("labels CSV must contain image_id,label columns")
        for row in reader:
            image_id = str(row.get("image_id") or "").strip()
            label = str(row.get("label") or "").strip()
            if image_id and label:
                labels[image_id] = label
    return labels


def _pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for p in positive:
        for n in negative:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / float(len(positive) * len(negative))


def _experiment_summary(
    rows: list[dict[str, Any]],
    experiment: str,
    labels: dict[str, str],
) -> dict[str, Any]:
    nonself = [row for row in rows if not bool(row["is_self"])]
    values = [float(row[experiment]) for row in nonself]
    baseline_ranks: list[float] = []
    experiment_ranks: list[float] = []
    for row in nonself:
        baseline_rank = row.get("E0_baseline_rank")
        current_rank = row.get(f"{experiment}_rank")
        if baseline_rank is not None and current_rank is not None:
            baseline_ranks.append(float(baseline_rank))
            experiment_ranks.append(float(current_rank))

    summary: dict[str, Any] = {
        "pair_count": len(nonself),
        "mean_score": None if not values else float(np.mean(values)),
        "std_score": None if not values else float(np.std(values)),
        "min_score": None if not values else float(np.min(values)),
        "max_score": None if not values else float(np.max(values)),
        "spearman_vs_baseline": _pearson(baseline_ranks, experiment_ranks),
    }
    if labels:
        positive = [
            float(row[experiment])
            for row in nonself
            if labels.get(str(row["query_id"])) is not None
            and labels.get(str(row["query_id"])) == labels.get(str(row["candidate_id"]))
        ]
        negative = [
            float(row[experiment])
            for row in nonself
            if labels.get(str(row["query_id"])) is not None
            and labels.get(str(row["candidate_id"])) is not None
            and labels.get(str(row["query_id"])) != labels.get(str(row["candidate_id"]))
        ]
        positive_mean = None if not positive else float(np.mean(positive))
        negative_mean = None if not negative else float(np.mean(negative))
        hit_at: dict[str, float] = {}
        queries = sorted({str(row["query_id"]) for row in nonself if str(row["query_id"]) in labels})
        for k in (1, 3, 5, 10):
            hits = 0
            valid = 0
            for query_id in queries:
                candidates = sorted(
                    (
                        row
                        for row in nonself
                        if str(row["query_id"]) == query_id and str(row["candidate_id"]) in labels
                    ),
                    key=lambda row: (-float(row[experiment]), str(row["candidate_id"])),
                )
                if not candidates:
                    continue
                valid += 1
                query_label = labels[query_id]
                if any(labels[str(row["candidate_id"])] == query_label for row in candidates[:k]):
                    hits += 1
            hit_at[f"hit_at_{k}"] = 0.0 if valid == 0 else hits / float(valid)
        summary.update(
            {
                "positive_count": len(positive),
                "negative_count": len(negative),
                "positive_mean": positive_mean,
                "negative_mean": negative_mean,
                "mean_separation": None if positive_mean is None or negative_mean is None else positive_mean - negative_mean,
                "auc": _auc(positive, negative),
                **hit_at,
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run whole-car contour ablation experiments.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--weight", default=str(DEFAULT_CONFIG))
    parser.add_argument("--labels", default=None, help="Optional CSV with image_id,label columns.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-self-ranking", action="store_true")
    parser.add_argument(
        "--render-topk",
        type=int,
        default=5,
        help="Render the top K non-self candidates per query by baseline rank (default: 5).",
    )
    parser.add_argument(
        "--render-all",
        action="store_true",
        help="Render every non-self pair instead of only --render-topk pairs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    started = time.time()
    input_dir = _resolve_path(args.input_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.weight)
    contour_cfg = config.get("contour") or {}
    images = _collect_images(input_dir, args.limit)
    if not images:
        raise SystemExit(f"no images found in {input_dir}")

    ids = _unique_ids(images)
    _write_csv(
        output_dir / "image_manifest.csv",
        [{"image_id": ids[image], "path": str(image)} for image in images],
        ["image_id", "path"],
    )
    masks, failures = _extract_masks(images, ids, config)
    valid_ids = [ids[image] for image in images if ids[image] in masks]
    if not valid_ids:
        raise SystemExit("no valid masks")

    rows: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_score_query, query_id, valid_ids, masks, contour_cfg): query_id
            for query_id in valid_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            query_id = futures[future]
            query_rows = future.result()
            rows.extend(query_rows)
            print(f"[query {index}/{len(valid_ids)}] ok {query_id} pairs={len(query_rows)}")

    _assign_ranks(rows, include_self=bool(args.include_self_ranking))
    labels = _load_labels(None if args.labels is None else _resolve_path(args.labels))

    common_fields = [
        "query_id",
        "candidate_id",
        "is_self",
        "fill_iou",
        "top_edge_iou",
        "bottom_edge_iou",
        "edge_score",
        "query_aspect_ratio",
        "candidate_aspect_ratio",
        "aspect_delta",
        "aspect_penalty",
        "baseline_before_gamma",
    ]
    all_fields = [
        *common_fields,
        *[name for name, _ in EXPERIMENTS],
        *[f"{name}_rank" for name, _ in EXPERIMENTS],
    ]
    rows.sort(key=lambda row: (str(row["query_id"]), str(row["candidate_id"])))
    _write_csv(output_dir / "all_pairs.csv", rows, all_fields)

    summaries: dict[str, Any] = {}
    for experiment, title in EXPERIMENTS:
        experiment_dir = output_dir / experiment
        experiment_rows = [
            {
                **{key: row.get(key) for key in common_fields},
                "score": row.get(experiment),
                "rank": row.get(f"{experiment}_rank"),
            }
            for row in rows
        ]
        _write_csv(experiment_dir / "pairs.csv", experiment_rows, [*common_fields, "score", "rank"])
        summary = _experiment_summary(rows, experiment, labels)
        summary["title"] = title
        summaries[experiment] = summary
        (experiment_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    visualization_dir = output_dir / "visualizations"
    _render_summary_chart(summaries, visualization_dir / "ablation_summary.png")
    id_to_path = {ids[image]: image for image in images if ids[image] in masks}
    render_topk = max(0, int(args.render_topk))
    rendered_count = 0
    for query_id in valid_ids:
        query_rows = sorted(
            (
                row
                for row in rows
                if str(row["query_id"]) == query_id and not bool(row["is_self"])
            ),
            key=lambda row: (
                float("inf") if row.get("E0_baseline_rank") is None else float(row["E0_baseline_rank"]),
                str(row["candidate_id"]),
            ),
        )
        selected_rows = query_rows if bool(args.render_all) else query_rows[:render_topk]
        for row in selected_rows:
            candidate_id = str(row["candidate_id"])
            rank = row.get("E0_baseline_rank")
            rank_text = "unranked" if rank is None else f"{int(rank):03d}"
            output_path = (
                visualization_dir
                / query_id
                / f"rank_{rank_text}_{query_id}_vs_{candidate_id}.png"
            )
            _render_pair_visual(
                query_path=id_to_path[query_id],
                candidate_path=id_to_path[candidate_id],
                query_id=query_id,
                candidate_id=candidate_id,
                query_mask=masks[query_id],
                candidate_mask=masks[candidate_id],
                row=row,
                contour_cfg=contour_cfg,
                output_path=output_path,
            )
            rendered_count += 1
    print(f"visualizations={visualization_dir} pair_images={rendered_count}")

    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_count": len(images),
        "valid_mask_count": len(valid_ids),
        "failed_mask_count": len(failures),
        "pair_count": len(rows),
        "workers": workers,
        "include_self_ranking": bool(args.include_self_ranking),
        "visualizations": {
            "directory": str(visualization_dir),
            "pair_image_count": rendered_count,
            "render_all": bool(args.render_all),
            "render_topk": render_topk,
            "summary_chart": str(visualization_dir / "ablation_summary.png"),
        },
        "contour_config": contour_cfg,
        "experiments": summaries,
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = [
        "# Whole-car Contour Ablation Summary",
        "",
        f"- Images: {len(images)}",
        f"- Valid masks: {len(valid_ids)}",
        f"- Ordered pairs: {len(rows)}",
        f"- Pair visualizations: {rendered_count}",
        f"- Visualization directory: `{visualization_dir}`",
        "",
        "| Experiment | Mean | Std | Spearman vs Baseline | Positive Mean | Negative Mean | Separation | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment, title in EXPERIMENTS:
        summary = summaries[experiment]
        fmt = lambda value: "-" if value is None else f"{float(value):.4f}"
        md.append(
            f"| {title} | {fmt(summary.get('mean_score'))} | {fmt(summary.get('std_score'))} | "
            f"{fmt(summary.get('spearman_vs_baseline'))} | {fmt(summary.get('positive_mean'))} | "
            f"{fmt(summary.get('negative_mean'))} | {fmt(summary.get('mean_separation'))} | "
            f"{fmt(summary.get('auc'))} |"
        )
    (output_dir / "ablation_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"summary={output_dir / 'ablation_summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
