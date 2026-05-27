from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class GalleryRoots:
    img_root: Path

    def resolve_view_dir(self, view: str) -> Path:
        v = (view or "front").strip().lower()
        if v in {"front", "正脸", "正脸视图"}:
            return self.img_root / "front"
        if v in {"side", "侧面", "侧面车身视图"}:
            return self.img_root / "side"
        if v in {"rear", "尾部", "尾部视图"}:
            return self.img_root / "rear"
        return self.img_root / v


def _safe_rel_url(path: Path) -> str:
    return "/".join(path.parts)


def path_to_url(*, path: str | Path, img_root: Path, result_root: Path) -> str | None:
    p = Path(path).expanduser()
    try:
        rp = p.resolve()
    except Exception:
        rp = p

    if not rp.exists():
        hint = str(path).strip().replace("\\", "/")
        lower = hint.lower()
        for anchor, root, prefix in (
            ("/result/", result_root, "/assets/result"),
            ("/img/", img_root, "/assets/gallery"),
        ):
            idx = lower.find(anchor)
            if idx >= 0:
                tail = hint[idx + len(anchor) :].lstrip("/")
                cand = (root / Path(tail)).resolve()
                if cand.exists():
                    rel = cand.relative_to(root.resolve())
                    return f"{prefix}/{_safe_rel_url(rel)}"

        name = Path(hint).name
        if name:
            for root, prefix in ((result_root, "/assets/result"), (img_root, "/assets/gallery")):
                hits = [x for x in root.rglob(name) if x.is_file()]
                if hits:
                    rel = hits[0].resolve().relative_to(root.resolve())
                    return f"{prefix}/{_safe_rel_url(rel)}"

    for root, prefix in ((result_root, "/assets/result"), (img_root, "/assets/gallery")):
        try:
            rel = rp.relative_to(root.resolve())
        except Exception:
            continue
        return f"{prefix}/{_safe_rel_url(rel)}"
    return None


def _find_existing_with_any_ext(dir_path: Path, stem: str) -> Path | None:
    for ext in sorted({x.lower() for x in IMAGE_EXTS}):
        p = dir_path / f"{stem}{ext}"
        if p.is_file():
            return p
    matches = sorted(dir_path.glob(f"{stem}.*"))
    for p in matches:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def list_gallery_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []

    upload_dir_names = {"_uploads", "uploads", "_query", "query"}
    out: list[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_parts = {part.lower() for part in p.relative_to(input_dir).parts[:-1]}
        if rel_parts & upload_dir_names:
            continue
        out.append(p)
    return out


def run_compare(
    *,
    query_image_path: Path,
    view: str,
    vehicle_type: str,
    topk: int,
    img_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        from similarity_pipeline import run_pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "无法导入 similarity_pipeline（请安装后端依赖：python -m pip install -r backend/requirements.txt）: "
            + str(e)
        )

    roots = GalleryRoots(img_root=img_root)
    input_dir = roots.resolve_view_dir(view)

    payload = run_pipeline(
        input_dir=input_dir,
        query_image=query_image_path,
        output_dir=result_root,
        topk=int(topk),
    )

    run_id = str(payload.get("run_id") or "")
    query_id = str(payload.get("query_id") or "query")
    label_dir = Path(str(payload.get("outputs", {}).get("front_label") or result_root / "front_label" / run_id))
    label_query = _find_existing_with_any_ext(label_dir, query_id)

    query_staged = payload.get("query_staged_path")
    query_url = path_to_url(path=str(query_staged), img_root=img_root, result_root=result_root) if query_staged else None
    query_anno_url = (
        path_to_url(path=label_query, img_root=img_root, result_root=result_root) if label_query else None
    )

    results: list[dict[str, Any]] = []
    for row in payload.get("results") or []:
        candidate_path = str(row.get("candidate_path") or "")
        url = path_to_url(path=candidate_path, img_root=img_root, result_root=result_root)
        candidate_name = Path(candidate_path).stem or str(row.get("candidate_id") or "")
        diff = row.get("contour_diff_image")
        diff_url = path_to_url(path=str(diff), img_root=img_root, result_root=result_root) if diff else None
        results.append(
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "candidate_name": candidate_name,
                "candidate_path": url or "",
                "final_score": float(row.get("final_score") or 0.0),
                "contour_score": None if row.get("contour_score") is None else float(row.get("contour_score")),
                "part_score": None if row.get("part_score") is None else float(row.get("part_score")),
                "contour_diff_image": diff_url,
                "analysis": list(row.get("analysis") or []),
            }
        )

    return {
        "run_id": run_id,
        "query_name": "上传比对图片",
        "query_staged_path": query_url or "",
        "query_annotation_url": query_anno_url,
        "vehicle_type": vehicle_type,
        "view": view,
        "results": results,
    }


def compare_from_report(*, report: dict[str, Any], img_root: Path, result_root: Path) -> dict[str, Any]:
    run_id = str(report.get("run_id") or "latest")
    query = report.get("query") or {}
    query_staged = query.get("staged_path") or query.get("path")
    query_url = path_to_url(path=str(query_staged), img_root=img_root, result_root=result_root) if query_staged else None

    results: list[dict[str, Any]] = []
    for row in report.get("results") or []:
        candidate_path = str(row.get("candidate_path") or "")
        url = path_to_url(path=candidate_path, img_root=img_root, result_root=result_root)
        candidate_name = Path(candidate_path).stem or str(row.get("candidate_id") or "")
        diff = row.get("contour_diff_image")
        diff_url = path_to_url(path=str(diff), img_root=img_root, result_root=result_root) if diff else None
        results.append(
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "candidate_name": candidate_name,
                "candidate_path": url or "",
                "final_score": float(row.get("final_score") or 0.0),
                "contour_score": None if row.get("contour_score") is None else float(row.get("contour_score")),
                "part_score": None if row.get("part_score") is None else float(row.get("part_score")),
                "contour_diff_image": diff_url,
                "analysis": list(row.get("analysis") or []),
            }
        )

    return {
        "run_id": run_id,
        "query_name": "上传比对图片",
        "query_staged_path": query_url or "",
        "results": results,
    }


def load_report(*, run_id: str, result_root: Path) -> dict[str, Any]:
    rid = (run_id or "").strip()
    if rid in {"", "latest"}:
        p = result_root / "latest_report.json"
    else:
        p = result_root / "reports" / f"{rid}.json"
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8"))


def score_tag(score: float | None) -> str:
    if score is None or not isinstance(score, (int, float)):
        return "无有效评分"
    if score >= 85:
        return "高相似"
    if score >= 72:
        return "中高相似"
    if score >= 65:
        return "局部相似"
    return "差异明显"


PART_LABELS = {
    "right_mirror": "后视镜",
    "front_right_light": "车灯",
    "front_bumper": "前保险杠",
    "front_glass": "前挡风玻璃",
}


def build_candidate_detail(
    *,
    report: dict[str, Any],
    candidate_id: str,
    img_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    run_id = str(report.get("run_id") or "latest")
    query = report.get("query") or {}
    query_id = str(query.get("id") or "query")

    results = report.get("results") or []
    row = next((x for x in results if str(x.get("candidate_id")) == candidate_id), None)
    if row is None:
        raise KeyError(candidate_id)

    outputs = report.get("outputs") or {}
    label_dir = Path(str(outputs.get("front_label") or result_root / "front_label" / run_id))
    label_query = _find_existing_with_any_ext(label_dir, query_id)
    label_candidate = _find_existing_with_any_ext(label_dir, candidate_id)

    query_staged = query.get("staged_path") or query.get("path")
    query_img_url = path_to_url(path=str(query_staged), img_root=img_root, result_root=result_root) or ""
    cand_img_url = path_to_url(path=str(row.get("candidate_path") or ""), img_root=img_root, result_root=result_root) or ""

    contour_score = None if row.get("contour_score") is None else float(row.get("contour_score"))
    part_score = None if row.get("part_score") is None else float(row.get("part_score"))
    final_score = float(row.get("final_score") or 0.0)

    final_weights = row.get("final_score_weights_used") or {"contour": 0.4, "parts": 0.6}
    cw = float(final_weights.get("contour") or 0.4)
    pw = float(final_weights.get("parts") or 0.6)

    diff_path = row.get("contour_diff_image")
    diff_url = path_to_url(path=str(diff_path), img_root=img_root, result_root=result_root) if diff_path else None

    evidence: list[dict[str, Any]] = []
    per_part = row.get("part_scores") or {}
    for part_key, obj in per_part.items():
        q_path = obj.get("query_path")
        c_path = obj.get("candidate_path")
        a_color = path_to_url(path=str(q_path), img_root=img_root, result_root=result_root) if q_path else None
        b_color = path_to_url(path=str(c_path), img_root=img_root, result_root=result_root) if c_path else None
        fused = float(obj.get("fused") or 0.0)
        evidence.append(
            {
                "part_name": PART_LABELS.get(str(part_key), str(part_key)),
                "fused": fused,
                "tag": score_tag(fused),
                "tiles": {
                    "a_color": a_color or "",
                    "b_color": b_color or "",
                    "a_gray": a_color or "",
                    "b_gray": b_color or "",
                },
                "metrics": {
                    "clip": obj.get("clip"),
                    "dino": obj.get("dino"),
                    "ssim": obj.get("ssim"),
                    "edge": obj.get("edge"),
                },
            }
        )

    evidence.sort(key=lambda x: float(x.get("fused") or 0.0), reverse=True)
    points = list(row.get("analysis") or [])

    return {
        "run_id": run_id,
        "query": {
            "name": "A车",
            "image_url": query_img_url,
            "annotation_url": path_to_url(path=label_query, img_root=img_root, result_root=result_root) if label_query else "",
        },
        "candidate": {
            "id": str(candidate_id),
            "name": "B车",
            "image_url": cand_img_url,
            "annotation_url": path_to_url(path=label_candidate, img_root=img_root, result_root=result_root) if label_candidate else "",
        },
        "summary": {
            "final_score": final_score,
            "tag": score_tag(final_score),
            "points": points,
            "weights": {
                "contour": {"weight": cw, "score": 0.0 if contour_score is None else contour_score},
                "parts": {"weight": pw, "score": 0.0 if part_score is None else part_score},
            },
        },
        "contour": {
            "score": 0.0 if contour_score is None else contour_score,
            "tag": score_tag(0.0 if contour_score is None else contour_score),
            "diff_image_url": diff_url or "",
            "conclusion": "结论：两车主体轮廓接近，差异主要集中在局部外扩与边缘细节区域。",
        },
        "parts": {
            "score": 0.0 if part_score is None else part_score,
            "tag": score_tag(0.0 if part_score is None else part_score),
            "a_annotation_url": path_to_url(path=label_query, img_root=img_root, result_root=result_root) if label_query else "",
            "b_annotation_url": path_to_url(path=label_candidate, img_root=img_root, result_root=result_root) if label_candidate else "",
            "evidence": evidence,
        },
    }
