import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


FeatureName = Literal["clip", "dino", "ssim", "edge"]

DEFAULT_PART_FEATURE_WEIGHTS: dict[str, dict[FeatureName, float]] = {
    "front_right_light": {"clip": 0.1, "dino": 0.4, "ssim": 0.4, "edge": 0.1},
    "front_bumper": {"clip": 0.25, "dino": 0.35, "ssim": 0.15, "edge": 0.25},
    "grille": {"clip": 0.25, "dino": 0.35, "ssim": 0.20, "edge": 0.20},
    "hood": {"clip": 0.25, "dino": 0.5, "ssim": 0.15, "edge": 0.1},
    "front_glass": {"clip": 0.20, "dino": 0.40, "ssim": 0.25, "edge": 0.15},
    "right_mirror": {"clip": 0.3, "dino": 0.45, "ssim": 0.15, "edge": 0.1},
    "left_mirror": {"clip": 0.3, "dino": 0.45, "ssim": 0.15, "edge": 0.1},
    "left_taillight": {"clip": 0.12, "dino": 0.38, "ssim": 0.30, "edge": 0.20},
    "line_taillight": {"clip": 0.10, "dino": 0.35, "ssim": 0.35, "edge": 0.20},
    "back_glass": {"clip": 0.18, "dino": 0.42, "ssim": 0.25, "edge": 0.15},
    "back_bumper": {"clip": 0.20, "dino": 0.32, "ssim": 0.18, "edge": 0.30},
    "trunk_lid": {"clip": 0.18, "dino": 0.45, "ssim": 0.22, "edge": 0.15},
}

DEFAULT_PART_OVERALL_WEIGHTS: dict[str, float] = {
    "right_mirror": 0.12,
    "front_right_light": 0.22,
    "front_bumper": 0.18,
    "grille": 0.18,
    "hood": 0.15,
    "front_glass": 0.15,
    "left_mirror": 0.12,
    "left_taillight": 0.20,
    "line_taillight": 0.14,
    "back_glass": 0.15,
    "back_bumper": 0.195,
    "trunk_lid": 0.195,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(1, -1)
    b = np.asarray(b, dtype=np.float32).reshape(1, -1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    s = float((a @ b.T)[0][0] / (na * nb))
    return s * 100.0


def _crop_to_content(img: Image.Image, threshold: int = 240, margin: int = 8) -> Image.Image:
    gray = np.array(img.convert("L"))
    mask = gray < threshold
    if not bool(mask.any()):
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = gray.shape
    rmin = max(0, int(rmin) - margin)
    rmax = min(h, int(rmax) + margin)
    cmin = max(0, int(cmin) - margin)
    cmax = min(w, int(cmax) + margin)
    if (rmax - rmin) * (cmax - cmin) < 0.1 * h * w:
        return img
    return img.crop((cmin, rmin, cmax, rmax))


def _pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    max_side = max(w, h)
    padded = Image.new(img.mode, (max_side, max_side), (255, 255, 255))
    padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    return padded


def preprocess_color(img_path: str, img_size: int = 224) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    img = _crop_to_content(img)
    img = _pad_to_square(img)
    return img.resize((img_size, img_size), Image.LANCZOS)


def preprocess_gray(img_path: str, img_size: int = 224) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    img = _crop_to_content(img)
    img = _pad_to_square(img)
    gray = np.array(img.convert("L"))
    equalized = cv2.equalizeHist(gray)
    return Image.fromarray(equalized).resize((img_size, img_size), Image.LANCZOS)


def extract_edge(image: Image.Image, img_size: int = 224) -> np.ndarray:
    gray = np.array(image.convert("L").resize((img_size, img_size)))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100).astype(np.float32)
    norm = float(np.linalg.norm(edges))
    if norm > 0:
        edges = edges / norm
    return edges.flatten().reshape(1, -1)


def make_diff_highlight(img_a: Image.Image, img_b: Image.Image, img_size: int = 224) -> Image.Image:
    a = np.array(img_a.resize((img_size, img_size)).convert("L"), dtype=np.float32)
    b = np.array(img_b.resize((img_size, img_size)).convert("L"), dtype=np.float32)
    diff = np.abs(a - b)
    p95 = float(np.percentile(diff, 95))
    if p95 > 0:
        diff_norm = np.clip(diff / p95 * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        diff_norm = diff.astype(np.uint8)
    out = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    out[:, :, 0] = diff_norm
    out[:, :, 1] = 255 - diff_norm
    return Image.fromarray(out, mode="RGB")


def _try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return ImageFont.truetype(str(p), size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _safe_filename(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_")
    s = s.replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
    s = "_".join([x for x in s.split() if x])
    if len(s) > max_len:
        s = s[:max_len]
    return s or "item"


def _score_color(score: float) -> tuple[int, int, int]:
    if score >= 75:
        return (46, 125, 50)
    if score >= 45:
        return (230, 81, 0)
    return (21, 101, 192)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _compose_row_card(
    *,
    width: int,
    height: int,
    title: str,
    detail: dict[str, Any],
    weights_used: dict[str, Any] | None,
    font: ImageFont.ImageFont,
) -> Image.Image:
    fused = float(detail.get("fused") or 0.0)
    col = _score_color(fused)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pad = 10
    draw.rounded_rectangle((pad, pad, width - pad, height - pad), radius=14, outline=col, width=3, fill=(245, 245, 245))
    draw.text((pad + 10, pad + 8), title, fill=(0, 0, 0), font=font)
    y = pad + 34
    lines = [
        f"Fused: {detail.get('fused')}",
        f"CLIP:  {detail.get('clip')}",
        f"DINO:  {detail.get('dino')}",
        f"SSIM:  {detail.get('ssim')}",
        f"EDGE:  {detail.get('edge')}",
    ]
    if weights_used:
        wc = float(weights_used.get("clip", 0.0))
        wd = float(weights_used.get("dino", 0.0))
        ws = float(weights_used.get("ssim", 0.0))
        we = float(weights_used.get("edge", 0.0))
        lines.append(f"W: C{round(wc,2)} D{round(wd,2)} S{round(ws,2)} E{round(we,2)}")
    for line in lines:
        draw.text((pad + 10, y), line, fill=col, font=font)
        y += 18
    return img


def _render_part_pages(
    *,
    part_key: str,
    part_pairs: list[dict[str, Any]],
    out_dir: Path,
    topk: int,
    pairs_per_page: int,
    cell: int,
    img_size: int,
):
    font = _try_load_font(16)
    font_small = _try_load_font(13)
    _ensure_dir(out_dir)

    take = part_pairs[: max(0, int(topk))]
    total = len(take)
    if total <= 0:
        return

    pages = (total + pairs_per_page - 1) // pairs_per_page
    gap = 10
    header_h = 78
    card_w = int(cell * 1.25)
    cols = 6
    w = gap + cols * cell + card_w + gap * (cols + 1)

    for page in range(pages):
        page_items = take[page * pairs_per_page : (page + 1) * pairs_per_page]
        rows = len(page_items)
        h = header_h + rows * (cell + gap) + gap
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        title = f"part={part_key} page={page+1}/{pages} total={total} topk={topk}"
        draw.text((gap, 14), title, fill=(0, 0, 0), font=font)
        draw.text((gap, 40), "A color | B color | A gray | B gray | diff | scores", fill=(90, 90, 90), font=font_small)

        y0 = header_h
        for idx, item in enumerate(page_items, start=1):
            rank = page * pairs_per_page + idx
            carA = str(item.get("carA", "A"))
            carB = str(item.get("carB", "B"))
            a_path = str(item.get("a_path") or "")
            b_path = str(item.get("b_path") or "")
            scores = item.get("scores") or {}
            weights_used = item.get("weights_used") or None

            color_a = preprocess_color(a_path, img_size=img_size).resize((cell, cell), Image.LANCZOS)
            color_b = preprocess_color(b_path, img_size=img_size).resize((cell, cell), Image.LANCZOS)
            gray_a = preprocess_gray(a_path, img_size=img_size).resize((cell, cell), Image.LANCZOS)
            gray_b = preprocess_gray(b_path, img_size=img_size).resize((cell, cell), Image.LANCZOS)
            diff = make_diff_highlight(gray_a, gray_b, img_size=cell)

            row_y = y0 + (idx - 1) * (cell + gap)
            x = gap
            canvas.paste(color_a, (x, row_y))
            draw.text((x, row_y - 14), f"#{rank} {carA}", fill=(0, 0, 0), font=font_small)
            x += cell + gap
            canvas.paste(color_b, (x, row_y))
            draw.text((x, row_y - 14), f"{carB}", fill=(0, 0, 0), font=font_small)
            x += cell + gap
            canvas.paste(gray_a.convert("RGB"), (x, row_y))
            x += cell + gap
            canvas.paste(gray_b.convert("RGB"), (x, row_y))
            x += cell + gap
            canvas.paste(diff, (x, row_y))
            x += cell + gap

            card = _compose_row_card(width=card_w, height=cell, title="scores", detail=scores, weights_used=weights_used, font=font_small)
            canvas.paste(card, (x, row_y))

        save_path = out_dir / f"{part_key}_page{page+1}.png"
        canvas.save(str(save_path))


def _render_summary_images(
    *,
    ranked_avg: list[dict[str, Any]],
    parts: list[str],
    out_dir: Path,
    topk: int,
    cell: int,
    img_size: int,
):
    font = _try_load_font(18)
    font_small = _try_load_font(13)
    _ensure_dir(out_dir)

    for idx, item in enumerate(ranked_avg[: max(0, int(topk))], start=1):
        carA = str(item.get("carA", "A"))
        carB = str(item.get("carB", "B"))
        avg = float(item.get("avg") or 0.0)
        parts_map: dict[str, Any] = item.get("parts") or {}

        header_h = 86
        row_h = cell
        rows = len(parts)
        gap = 10
        w = gap * 7 + cell * 5 + int(cell * 1.25)
        h = header_h + rows * (row_h + gap) + gap
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        col = _score_color(avg)
        draw.text((gap, 12), f"rank={idx} avg={round(avg,2)}", fill=col, font=font)
        draw.text((gap, 42), f"{carA} vs {carB}", fill=(0, 0, 0), font=font_small)

        y0 = header_h
        for r, part_key in enumerate(parts):
            part_obj = parts_map.get(part_key) or {}
            a_path = str(part_obj.get("a_path") or "")
            b_path = str(part_obj.get("b_path") or "")
            scores = part_obj.get("scores") or {}
            weights_used = part_obj.get("weights_used") or None
            if not a_path or not b_path:
                continue

            color_a = preprocess_color(a_path, img_size=img_size).resize((cell, cell), Image.LANCZOS)
            color_b = preprocess_color(b_path, img_size=img_size).resize((cell, cell), Image.LANCZOS)
            gray_a = preprocess_gray(a_path, img_size=img_size).resize((cell, cell), Image.LANCZOS)
            gray_b = preprocess_gray(b_path, img_size=img_size).resize((cell, cell), Image.LANCZOS)
            diff = make_diff_highlight(gray_a, gray_b, img_size=cell)

            row_y = y0 + r * (row_h + gap)
            x = gap
            draw.text((x, row_y - 14), part_key, fill=(0, 0, 0), font=font_small)
            canvas.paste(color_a, (x, row_y))
            x += cell + gap
            canvas.paste(color_b, (x, row_y))
            x += cell + gap
            canvas.paste(gray_a.convert("RGB"), (x, row_y))
            x += cell + gap
            canvas.paste(gray_b.convert("RGB"), (x, row_y))
            x += cell + gap
            canvas.paste(diff, (x, row_y))
            x += cell + gap
            card = _compose_row_card(
                width=int(cell * 1.25),
                height=cell,
                title=f"part={part_key}",
                detail=scores,
                weights_used=weights_used,
                font=font_small,
            )
            canvas.paste(card, (x, row_y))

        fname = f"summary_rank{idx:02d}_{_safe_filename(carA)}_vs_{_safe_filename(carB)}.png"
        canvas.save(str(out_dir / fname))


def _render_self_check(
    *,
    ranked_avg: list[dict[str, Any]],
    out_dir: Path,
    topk: int,
    pairs_per_page: int,
    cell: int,
    img_size: int,
):
    self_items = [x for x in ranked_avg if str(x.get("carA")) == str(x.get("carB"))]
    if not self_items:
        return
    items = self_items[: max(0, int(topk))]
    _ensure_dir(out_dir)
    Path(out_dir / "self_check.json").write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_ssim_score(img_a: Image.Image, img_b: Image.Image, img_size: int = 224) -> float:
    a = np.array(img_a.convert("L").resize((img_size, img_size)), dtype=np.float32)
    b = np.array(img_b.convert("L").resize((img_size, img_size)), dtype=np.float32)

    ksize = 11
    sigma = 1.5
    mu1 = cv2.GaussianBlur(a, (ksize, ksize), sigma)
    mu2 = cv2.GaussianBlur(b, (ksize, ksize), sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(a * a, (ksize, ksize), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (ksize, ksize), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(a * b, (ksize, ksize), sigma) - mu1_mu2

    l = 255.0
    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    s = float(np.mean(ssim_map))
    s = max(-1.0, min(1.0, s))
    return round((s + 1.0) / 2.0 * 100.0, 1)


def _parse_features(s: str) -> list[FeatureName]:
    items = [x.strip().lower() for x in (s or "").split(",") if x.strip()]
    allowed: set[str] = {"clip", "dino", "ssim", "edge"}
    out: list[FeatureName] = []
    for x in items:
        if x not in allowed:
            raise SystemExit(f"未知特征: {x}，可选: clip,dino,ssim,edge")
        out.append(x)  # type: ignore[arg-type]
    if not out:
        raise SystemExit("features 不能为空")
    return out


def _weights_from_args(args) -> dict[FeatureName, float]:
    w = {"clip": float(args.w_clip), "dino": float(args.w_dino), "ssim": float(args.w_ssim), "edge": float(args.w_edge)}
    return w  # type: ignore[return-value]


def _normalize_weights(weights: dict[FeatureName, float], enabled: Iterable[FeatureName]) -> dict[FeatureName, float]:
    enabled = list(enabled)
    s = 0.0
    for k in enabled:
        s += max(0.0, float(weights.get(k, 0.0)))
    if s <= 0:
        per = 1.0 / float(len(enabled)) if enabled else 0.0
        return {k: per for k in enabled}
    return {k: max(0.0, float(weights.get(k, 0.0))) / s for k in enabled}


def load_weight_config(
    path: Path | None,
) -> tuple[dict[str, dict[FeatureName, float]], dict[str, float]]:
    part_feature = dict(DEFAULT_PART_FEATURE_WEIGHTS)
    part_overall = dict(DEFAULT_PART_OVERALL_WEIGHTS)
    if path is None:
        return part_feature, part_overall

    p = path
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    if not p.exists():
        raise SystemExit(f"weights-config 不存在: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    pf = raw.get("part_feature_weights") or {}
    po = raw.get("part_overall_weights") or {}

    if not isinstance(pf, dict) or not isinstance(po, dict):
        raise SystemExit("weights-config 格式错误：需要包含 part_feature_weights/part_overall_weights 对象")

    for part, row in pf.items():
        if not isinstance(row, dict):
            raise SystemExit(f"weights-config 格式错误：part_feature_weights[{part}] 不是对象")
        w = {k: float(row.get(k, 0.0)) for k in ["clip", "dino", "ssim", "edge"]}
        s = float(sum(w.values()))
        if abs(s - 1.0) > 1e-6:
            raise SystemExit(f"weights-config 错误：{part} 特征权重之和不等于 1.0")
        part_feature[str(part)] = w  # type: ignore[assignment]

    for part, v in po.items():
        part_overall[str(part)] = float(v)

    return part_feature, part_overall


def part_feature_weights_for(
    part_key: str,
    *,
    enabled: list[FeatureName],
    matrix: dict[str, dict[FeatureName, float]],
) -> dict[FeatureName, float]:
    if not enabled:
        return {"clip": 0.0, "dino": 0.0, "ssim": 0.0, "edge": 0.0}  # type: ignore[return-value]
    if part_key not in matrix:
        per = 1.0 / float(len(enabled))
        return {k: (per if k in enabled else 0.0) for k in ["clip", "dino", "ssim", "edge"]}  # type: ignore[return-value]
    row = matrix[part_key]
    base = {k: float(row.get(k, 0.0)) for k in ["clip", "dino", "ssim", "edge"]}  # type: ignore[assignment]
    w = _normalize_weights(base, enabled)
    return {k: (float(w.get(k, 0.0)) if k in enabled else 0.0) for k in ["clip", "dino", "ssim", "edge"]}  # type: ignore[return-value]


def merged_overall_weights(parts_present: list[str], base: dict[str, float]) -> dict[str, float]:
    parts_present = [p for p in parts_present if p]
    if not parts_present:
        return {}
    known = [p for p in parts_present if p in base]
    missing = [p for p in parts_present if p not in base]
    known_sum = float(sum(float(base.get(p, 0.0)) for p in known))
    if known_sum > 1.0:
        return {p: float(base.get(p, 0.0)) / known_sum for p in known}
    remain = max(0.0, 1.0 - known_sum)
    out: dict[str, float] = {p: float(base.get(p, 0.0)) for p in known}
    if missing:
        per = remain / float(len(missing))
        for p in missing:
            out[p] = per
    s = float(sum(out.values()))
    if s > 0:
        out = {k: float(v) / s for k, v in out.items()}
    return out


def _find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _default_dino_repo(repo_root: Path) -> Path:
    return repo_root / "models" / "dino" / "facebookresearch_dinov2_main"


def _select_weight_file(candidates: list[Path], *, min_bytes: int = 10 * 1024 * 1024) -> Path | None:
    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.stat().st_size < min_bytes:
                continue
        except Exception:
            continue
        return p
    return None


def _default_dino_weights(repo_root: Path, model_dir: Path, variant: str) -> Path | None:
    v = (variant or "").lower()
    prefer_reg = "reg" in v
    reg_candidates = [
        model_dir / "dinov2_vitb14_reg4_pretrain.pth",
        repo_root / "models" / "dino" / "dinov2_vitb14_reg4_pretrain.pth",
    ]
    base_candidates = [
        model_dir / "dinov2_vitb14_pretrain.pth",
        repo_root / "models" / "dino" / "checkpoints" / "dinov2_vitb14_pretrain.pth",
    ]
    candidates = reg_candidates + base_candidates if prefer_reg else base_candidates + reg_candidates
    return _select_weight_file(candidates)


def _extract_state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError("无法解析权重文件内容")


def _load_dino_model_local(
    *,
    variant: str,
    dino_repo_dir: Path,
    dino_weights: Path,
    device: torch.device,
) -> torch.nn.Module:
    dino_repo_dir = dino_repo_dir.resolve()
    if not dino_repo_dir.is_dir():
        raise RuntimeError(f"dinov2 repo 不存在: {dino_repo_dir}")
    v = (variant or "").strip()
    if v.startswith("dinov2_"):
        v = v[len("dinov2_") :]
    names = [f"dinov2_{v}"]
    errs: list[Exception] = []
    for name in names:
        try:
            model = torch.hub.load(str(dino_repo_dir), name, source="local", pretrained=False)
            sd = _extract_state_dict(torch.load(str(dino_weights), map_location="cpu"))
            model.load_state_dict(sd, strict=True)
            model.eval()
            return model.to(device)
        except Exception as e:
            errs.append(e)
    msg = "; ".join([str(e) for e in errs[-3:]]) if errs else "unknown"
    raise RuntimeError(f"无法加载 DINOv2: variant={variant} weights={dino_weights}. 原始错误: {msg}")


def _try_load_clip_local(clip_dir: Path, device: torch.device):
    try:
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except Exception as e:
        raise RuntimeError(f"未安装 transformers，无法启用 CLIP。原始错误: {e}")

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if not clip_dir.exists():
        raise RuntimeError(f"CLIP 目录不存在: {clip_dir}")

    processor = CLIPProcessor.from_pretrained(str(clip_dir), local_files_only=True)
    model = CLIPModel.from_pretrained(str(clip_dir), local_files_only=True).to(device)
    model.eval()
    return processor, model


def _extract_clip(processor, model, image: Image.Image, device: torch.device) -> np.ndarray:
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        feat = outputs.pooler_output
        feat = model.visual_projection(feat)
        denom = feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feat = feat / denom
    return feat.detach().cpu().numpy()


def _extract_dino(model: torch.nn.Module, transform, image: Image.Image, device: torch.device) -> np.ndarray:
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            out = model.forward_features(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                out = out["x_norm_clstoken"]
        else:
            out = model(x)
        if isinstance(out, dict) and "x_norm_clstoken" in out:
            out = out["x_norm_clstoken"]
        if isinstance(out, (tuple, list)):
            out = out[0]
    out = out.detach().cpu().numpy()
    feat = out.reshape(out.shape[0], -1)[0]
    n = float(np.linalg.norm(feat))
    if n > 0:
        feat = feat / n
    return feat.reshape(1, -1)


@dataclass(frozen=True)
class SimilarityDetail:
    clip: float | None
    dino: float | None
    ssim: float | None
    edge: float | None
    fused: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "dino": self.dino,
            "ssim": self.ssim,
            "edge": self.edge,
            "fused": self.fused,
        }


class CdseSimilarityEngine:
    def __init__(
        self,
        *,
        model_dir: str | Path = "./model",
        clip_dir: str | Path | None = None,
        dino_variant: str = "vitb14_reg",
        dino_repo_dir: str | Path | None = None,
        dino_weights: str | Path | None = None,
        img_size: int = 224,
        device: str | None = None,
    ):
        self._repo_root = _repo_root()
        self._model_dir = (_as_path(model_dir) if model_dir else Path("./model")).expanduser()
        if not self._model_dir.is_absolute():
            self._model_dir = (self._repo_root / self._model_dir).resolve()

        self._clip_dir = _as_path(clip_dir).expanduser() if clip_dir else (self._model_dir / "clip-vit-large-patch14")
        if not self._clip_dir.is_absolute():
            self._clip_dir = (self._repo_root / self._clip_dir).resolve()

        self._dino_variant = dino_variant
        self._dino_repo_dir = _as_path(dino_repo_dir).expanduser() if dino_repo_dir else _default_dino_repo(self._repo_root)
        if not self._dino_repo_dir.is_absolute():
            self._dino_repo_dir = (self._repo_root / self._dino_repo_dir).resolve()

        self._dino_weights = (
            _as_path(dino_weights).expanduser()
            if dino_weights
            else _default_dino_weights(self._repo_root, self._model_dir, self._dino_variant)
        )
        self._img_size = int(img_size)

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._clip_processor = None
        self._clip_model = None
        self._dino_model = None
        self._dino_transform = transforms.Compose(
            [
                transforms.Resize((self._img_size, self._img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def ensure_models(self, features: Iterable[FeatureName]):
        feats = set(features)
        if "clip" in feats and self._clip_model is None:
            self._clip_processor, self._clip_model = _try_load_clip_local(self._clip_dir, self._device)
        if "dino" in feats and self._dino_model is None:
            if self._dino_weights is None:
                raise RuntimeError("未找到 DINOv2 权重文件，请用 --dino-weights 指定")
            self._dino_model = _load_dino_model_local(
                variant=self._dino_variant,
                dino_repo_dir=self._dino_repo_dir,
                dino_weights=self._dino_weights,
                device=self._device,
            )

    def _resolve_features(self, requested: Iterable[FeatureName], *, require_clip: bool) -> list[FeatureName]:
        req = list(requested)
        out: list[FeatureName] = []
        for f in req:
            if f == "clip":
                try:
                    self.ensure_models(["clip"])
                    out.append("clip")
                except Exception as e:
                    if require_clip:
                        raise
                    print(f"CLIP disabled: {e}", file=sys.stderr)
            elif f == "dino":
                self.ensure_models(["dino"])
                out.append("dino")
            else:
                out.append(f)
        return out

    def extract_features(self, color_img: Image.Image, features: Iterable[FeatureName]) -> dict[FeatureName, np.ndarray]:
        out: dict[FeatureName, np.ndarray] = {}
        feats = set(features)
        if "clip" in feats:
            self.ensure_models(["clip"])
            assert self._clip_processor is not None and self._clip_model is not None
            out["clip"] = _extract_clip(self._clip_processor, self._clip_model, color_img, self._device)
        if "dino" in feats:
            self.ensure_models(["dino"])
            assert self._dino_model is not None
            out["dino"] = _extract_dino(self._dino_model, self._dino_transform, color_img, self._device)
        if "edge" in feats:
            out["edge"] = extract_edge(color_img, img_size=self._img_size)
        return out

    def compare_paths(
        self,
        img_a_path: str,
        img_b_path: str,
        *,
        features: list[FeatureName],
        weights: dict[FeatureName, float],
        auto_renorm: bool = True,
        require_clip: bool = False,
    ) -> SimilarityDetail:
        color_a = preprocess_color(img_a_path, img_size=self._img_size)
        color_b = preprocess_color(img_b_path, img_size=self._img_size)
        gray_a = preprocess_gray(img_a_path, img_size=self._img_size)
        gray_b = preprocess_gray(img_b_path, img_size=self._img_size)

        enabled: list[FeatureName] = self._resolve_features(features, require_clip=require_clip)
        feat_a = self.extract_features(color_a, enabled)
        feat_b = self.extract_features(color_b, enabled)

        w = _normalize_weights(weights, enabled) if auto_renorm else weights

        clip_score = None
        dino_score = None
        ssim_score = None
        edge_score = None

        fused = 0.0
        if "clip" in enabled and "clip" in feat_a and "clip" in feat_b:
            clip_score = round(_cosine_score(feat_a["clip"], feat_b["clip"]), 1)
            fused += clip_score * float(w.get("clip", 0.0))
        if "dino" in enabled and "dino" in feat_a and "dino" in feat_b:
            dino_score = round(_cosine_score(feat_a["dino"], feat_b["dino"]), 1)
            fused += dino_score * float(w.get("dino", 0.0))
        if "ssim" in enabled:
            ssim_score = float(compute_ssim_score(gray_a, gray_b, img_size=self._img_size))
            fused += ssim_score * float(w.get("ssim", 0.0))
        if "edge" in enabled and "edge" in feat_a and "edge" in feat_b:
            edge_score = round(_cosine_score(feat_a["edge"], feat_b["edge"]), 1)
            fused += edge_score * float(w.get("edge", 0.0))

        return SimilarityDetail(
            clip=clip_score,
            dino=dino_score,
            ssim=ssim_score,
            edge=edge_score,
            fused=round(float(fused), 1),
        )


def load_dataset(front_dir: str, target_parts: list[str]) -> dict[str, dict[str, str]]:
    front_dir = os.path.abspath(front_dir)
    if not os.path.isdir(front_dir):
        raise SystemExit(f"目录不存在: {front_dir}")

    dataset: dict[str, dict[str, str]] = {}
    for car in sorted(os.listdir(front_dir)):
        car_dir = os.path.join(front_dir, car)
        if not os.path.isdir(car_dir):
            continue
        mapping: dict[str, str] = {}
        for fname in sorted(os.listdir(car_dir)):
            low = fname.lower()
            if not low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")):
                continue
            for part in target_parts:
                if part in low and part not in mapping:
                    mapping[part] = os.path.join(car_dir, fname)
        if mapping:
            dataset[car] = mapping
    return dataset


def rank_pairs(
    dataset: dict[str, dict[str, str]],
    *,
    engine: CdseSimilarityEngine,
    parts: list[str],
    features: list[FeatureName],
    weights: dict[FeatureName, float],
    part_feature_weights: dict[str, dict[FeatureName, float]],
    part_overall_weights: dict[str, float],
    auto_renorm: bool = True,
    include_self: bool = False,
    require_clip: bool = False,
) -> list[dict[str, Any]]:
    cars = sorted(dataset.keys())
    pair_items: dict[tuple[str, str], dict[str, Any]] = {}

    enabled_features = engine._resolve_features(features, require_clip=require_clip)
    global_weights = _normalize_weights(weights, enabled_features) if auto_renorm else weights

    if include_self:
        pairs_iter = ((cars[i], cars[j]) for i in range(len(cars)) for j in range(i, len(cars)))
    else:
        pairs_iter = itertools.combinations(cars, 2)

    for car_a, car_b in pairs_iter:
        per_part: dict[str, dict[str, Any]] = {}
        part_fused: dict[str, float] = {}
        for part in parts:
            p1 = dataset.get(car_a, {}).get(part)
            p2 = dataset.get(car_b, {}).get(part)
            if not p1 or not p2:
                continue
            pw = part_feature_weights_for(part, enabled=enabled_features, matrix=part_feature_weights)
            detail = engine.compare_paths(p1, p2, features=enabled_features, weights=pw, auto_renorm=False)
            per_part[part] = {"a_path": p1, "b_path": p2, "scores": detail.to_dict(), "weights_used": pw}
            part_fused[part] = float(detail.fused)
        if not part_fused:
            continue

        present_parts = list(per_part.keys())
        ow = merged_overall_weights(present_parts, part_overall_weights)
        if not ow:
            avg = float(np.mean(list(part_fused.values())))
            ow = {p: 1.0 / float(len(part_fused)) for p in part_fused}
        else:
            avg = float(sum(part_fused[p] * float(ow.get(p, 0.0)) for p in part_fused))

        pair_items[(car_a, car_b)] = {
            "carA": car_a,
            "carB": car_b,
            "avg": round(avg, 2),
            "parts": per_part,
            "overall_weights_used": ow,
            "enabled_features": enabled_features,
            "global_weights_used": global_weights,
        }

    ranked = sorted(pair_items.values(), key=lambda x: float(x["avg"]), reverse=True)
    return ranked


def _cmd_compare(args) -> int:
    features = _parse_features(args.features)
    if args.clip:
        if "clip" not in features:
            features = ["clip", *features]
    if args.no_clip:
        features = [f for f in features if f != "clip"]
    weights = _weights_from_args(args)
    engine = CdseSimilarityEngine(
        model_dir=args.model_dir,
        clip_dir=args.clip_dir,
        dino_variant=args.dino_variant,
        dino_repo_dir=args.dino_repo,
        dino_weights=args.dino_weights,
        img_size=args.img_size,
        device=args.device,
    )
    detail = engine.compare_paths(
        args.a,
        args.b,
        features=features,
        weights=weights,
        auto_renorm=not args.no_auto_renorm,
        require_clip=bool(args.clip_required),
    )
    payload = {"a": args.a, "b": args.b, "device": str(engine.device), "detail": detail.to_dict()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_rank(args) -> int:
    parts = [x.strip() for x in (args.parts or "").split(",") if x.strip()]
    if not parts:
        raise SystemExit("parts 不能为空，例如: front_bumper,hood,right_mirror")

    part_feature_matrix, part_overall_base = load_weight_config(Path(args.weights_config) if args.weights_config else None)

    features = _parse_features(args.features)
    if args.clip:
        if "clip" not in features:
            features = ["clip", *features]
    if args.no_clip:
        features = [f for f in features if f != "clip"]
    weights = _weights_from_args(args)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    rank_json_path = Path(args.out) if args.out else (out_dir / "rank.json")
    engine = CdseSimilarityEngine(
        model_dir=args.model_dir,
        clip_dir=args.clip_dir,
        dino_variant=args.dino_variant,
        dino_repo_dir=args.dino_repo,
        dino_weights=args.dino_weights,
        img_size=args.img_size,
        device=args.device,
    )
    dataset = load_dataset(args.front_dir, parts)
    ranked = rank_pairs(
        dataset,
        engine=engine,
        parts=parts,
        features=features,
        weights=weights,
        part_feature_weights=part_feature_matrix,
        part_overall_weights=part_overall_base,
        auto_renorm=not args.no_auto_renorm,
        include_self=not args.no_include_self,
        require_clip=bool(args.clip_required),
    )
    rank_json_path.write_text(
        json.dumps(
            {"config_used": {"part_feature_weights": part_feature_matrix, "part_overall_weights": part_overall_base}, "items": ranked},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written={rank_json_path}")
    print(f"pairs={len(ranked)}")

    if not args.no_render:
        topk = int(args.topk)
        pairs_per_page = int(args.pairs_per_page)
        cell = int(args.cell)

        summary_dir = out_dir / "summary"
        parts_dir = out_dir / "parts"
        self_dir = out_dir / "self_check"

        if args.render in {"both", "summary"}:
            _render_summary_images(
                ranked_avg=ranked,
                parts=parts,
                out_dir=summary_dir,
                topk=topk,
                cell=cell,
                img_size=int(args.img_size),
            )

        if args.render in {"both", "parts"}:
            for part_key in parts:
                part_pairs = []
                for item in ranked:
                    pobj = (item.get("parts") or {}).get(part_key)
                    if not pobj:
                        continue
                    scores = pobj.get("scores") or {}
                    fused = float(scores.get("fused") or 0.0)
                    part_pairs.append(
                        {
                            "carA": item.get("carA"),
                            "carB": item.get("carB"),
                            "a_path": pobj.get("a_path"),
                            "b_path": pobj.get("b_path"),
                            "scores": scores,
                            "weights_used": pobj.get("weights_used"),
                            "fused": fused,
                        }
                    )
                part_pairs.sort(key=lambda x: float(x.get("fused") or 0.0), reverse=True)
                _render_part_pages(
                    part_key=part_key,
                    part_pairs=part_pairs,
                    out_dir=parts_dir / part_key,
                    topk=topk,
                    pairs_per_page=pairs_per_page,
                    cell=cell,
                    img_size=int(args.img_size),
                )

        _render_self_check(
            ranked_avg=ranked,
            out_dir=self_dir,
            topk=topk,
            pairs_per_page=pairs_per_page,
            cell=cell,
            img_size=int(args.img_size),
        )

    return 0


def main():
    parser = argparse.ArgumentParser(description="CDSE 相似度（复用 direction_B_new.ipynb，支持离线本地模型）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--model-dir", default="./models")
        p.add_argument("--clip-dir", default="./models/clip-vit-large-patch14")
        p.add_argument("--weights-config", "--weight", dest="weights_config", default=None)
        p.add_argument("--clip", action="store_true", help="启用 CLIP（需要本地模型目录+transformers；不可用时默认自动跳过）")
        p.add_argument("--no-clip", action="store_true", help="禁用 CLIP（即使 --features 包含 clip）")
        p.add_argument("--clip-required", action="store_true", help="若启用 CLIP 但不可用，则直接报错退出")
        p.add_argument("--dino-variant", default="vitb14_reg")
        p.add_argument("--dino-repo", default=None)
        p.add_argument("--dino-weights", default=None)
        p.add_argument("--img-size", type=int, default=224)
        p.add_argument("--device", default=None)
        p.add_argument("--features", default="clip,dino,ssim,edge")
        p.add_argument("--w-clip", type=float, default=0.30)
        p.add_argument("--w-dino", type=float, default=0.40)
        p.add_argument("--w-ssim", type=float, default=0.15)
        p.add_argument("--w-edge", type=float, default=0.15)
        p.add_argument("--no-auto-renorm", action="store_true")

    p_cmp = sub.add_parser("compare", help="两张图片相似度")
    add_common(p_cmp)
    p_cmp.add_argument("--a", required=True)
    p_cmp.add_argument("--b", required=True)
    p_cmp.set_defaults(func=_cmd_compare)

    p_rank = sub.add_parser("rank", help="按部件目录批量两两对比并排序（可输出可视化拼图）")
    add_common(p_rank)
    p_rank.add_argument("--front-dir", "--input-dir", dest="front_dir", required=True)
    p_rank.add_argument("--parts", required=True)
    p_rank.add_argument("--out", default=None)
    p_rank.add_argument("--out-dir", "--output-dir", dest="out_dir", default="./result_cdse")
    p_rank.add_argument("--topk", type=int, default=50)
    p_rank.add_argument("--pairs-per-page", type=int, default=15)
    p_rank.add_argument("--cell", type=int, default=180)
    p_rank.add_argument("--render", choices=["both", "summary", "parts"], default="both")
    p_rank.add_argument("--no-render", action="store_true")
    p_rank.add_argument("--no-include-self", action="store_true")
    p_rank.set_defaults(func=_cmd_rank)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
