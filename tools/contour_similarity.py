from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]


def _iou_percent(a: np.ndarray, b: np.ndarray) -> float | None:
    union = np.logical_or(a, b)
    if not np.any(union):
        return None
    overlap = np.logical_and(a, b)
    return float(np.sum(overlap) / np.sum(union) * 100.0)


def _resize_keep_aspect_pad(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = mask.shape[:2]
    if th <= 0 or tw <= 0 or h <= 0 or w <= 0:
        return np.zeros((max(1, th), max(1, tw)), dtype=bool)
    scale = min(float(tw) / float(w), float(th) / float(h))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    out = np.zeros((th, tw), dtype=bool)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _mask_centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _shift_mask(mask: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    dx_i = int(round(dx))
    dy_i = int(round(dy))
    if dx_i == 0 and dy_i == 0:
        return mask
    h, w = mask.shape[:2]
    out = np.zeros_like(mask, dtype=bool)
    src_x0 = max(0, -dx_i)
    src_y0 = max(0, -dy_i)
    dst_x0 = max(0, dx_i)
    dst_y0 = max(0, dy_i)
    copy_w = w - abs(dx_i)
    copy_h = h - abs(dy_i)
    if copy_w <= 0 or copy_h <= 0:
        return out
    out[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = mask[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    return out


def _edge_band(mask: np.ndarray, *, thickness: int) -> np.ndarray:
    t = max(1, int(thickness))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
    m = (mask > 0).astype(np.uint8) * 255
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k)
    return grad > 0


def contour_score_and_vis(
    query_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    contour_cfg: dict[str, Any] | None = None,
) -> tuple[float, np.ndarray] | None:
    cfg = contour_cfg or {}
    q = _crop_mask(query_mask)
    c = _crop_mask(candidate_mask)
    if q is None or c is None:
        return None
    q_bool = q > 0
    keep_aspect = bool(cfg.get("keep_aspect", True))
    if keep_aspect:
        c_aligned = _resize_keep_aspect_pad(c, (q.shape[0], q.shape[1]))
    else:
        c_aligned = cv2.resize(c.astype(np.uint8), (q.shape[1], q.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    align = str(cfg.get("align", "centroid")).strip().lower()
    if align == "centroid":
        qc = _mask_centroid_xy(q_bool)
        cc = _mask_centroid_xy(c_aligned)
        if qc is not None and cc is not None:
            c_aligned = _shift_mask(c_aligned, dx=qc[0] - cc[0], dy=qc[1] - cc[1])

    score_fill = _iou_percent(q_bool, c_aligned)
    if score_fill is None:
        return None

    edge_weight = float(cfg.get("edge_weight", 0.6))
    edge_weight = max(0.0, min(1.0, edge_weight))
    thickness = int(cfg.get("edge_thickness", 4))
    score = score_fill
    if edge_weight > 0:
        qe = _edge_band(q_bool, thickness=thickness)
        ce = _edge_band(c_aligned, thickness=thickness)
        score_edge = _iou_percent(qe, ce)
        if score_edge is not None:
            score = (1.0 - edge_weight) * score_fill + edge_weight * score_edge

    aspect_k = float(cfg.get("aspect_penalty_k", 1.2))
    if aspect_k > 0:
        ar_q = float(q.shape[0]) / max(1.0, float(q.shape[1]))
        ar_c = float(c.shape[0]) / max(1.0, float(c.shape[1]))
        ratio = math.log(max(1e-6, ar_c) / max(1e-6, ar_q))
        penalty = math.exp(-aspect_k * abs(ratio))
        score *= float(penalty)

    score_scale = float(cfg.get("score_scale", 1.0))
    if score_scale > 0:
        score *= score_scale

    vis = np.zeros((q.shape[0], q.shape[1], 3), dtype=np.uint8)
    vis[q_bool] = [0, 0, 255]
    vis[c_aligned] = [0, 255, 0]
    overlap = np.logical_and(q_bool, c_aligned)
    vis[overlap] = [0, 255, 255]
    return round(max(0.0, min(99.0, score)), 1), vis
