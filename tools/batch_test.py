from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from similarity_pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    IMAGE_EXTS,
    ImageItem,
    ROOT,
    _config_path,
    _compare_parts,
    _feature_list,
    _load_config,
    _parse_csv,
    _prepare_run_images,
    _run_contour_compare,
    _safe_name,
    _write_reports,
    run_pipeline,
)
from tools import car_front_seg  # noqa: E402
from tools.cutout_by_sam import SamModelSpec, load_sam_predictor, run_sam_cutout_from_instances  # noqa: E402


DEFAULT_INPUT_DIR = Path(r"C:\Users\Lenovo\Desktop\img-test")
DEFAULT_OUTPUT_DIR = ROOT / "result" / "batch_test"


def _resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"input dir not found: {input_dir}")
    return sorted(
        p.resolve()
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _unique_job_names(images: list[Path]) -> dict[Path, str]:
    used: set[str] = set()
    out: dict[Path, str] = {}
    for idx, image in enumerate(images, start=1):
        base = _safe_name(image.stem, fallback=f"image_{idx:04d}")
        name = f"{idx:04d}_{base}"
        candidate = name
        n = 1
        while candidate in used:
            n += 1
            candidate = f"{name}_{n}"
        used.add(candidate)
        out[image] = candidate
    return out


def _prepare_query_copy(*, query_path: Path, query_dir: Path) -> Path:
    query_copy_dir = query_dir / "_query"
    query_copy_dir.mkdir(parents=True, exist_ok=True)
    query_copy = query_copy_dir / f"query{query_path.suffix.lower()}"
    shutil.copy2(query_path, query_copy)
    return query_copy


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _atomic_mode_enabled(args: argparse.Namespace) -> bool:
    return bool(args.cutout_only or args.label_only or args.contour_only)


def _run_part_atomic_steps(
    *,
    stage_dir: Path,
    label_dir: Path,
    parts_dir: Path,
    cutout_dir: Path,
    config: dict[str, Any],
    allowed_parts: set[str],
    args: argparse.Namespace,
    save_label: bool,
    save_cutout: bool,
) -> dict[str, Any]:
    front_weight = _config_path(config, "front_part_weight")
    model = YOLO(str(front_weight))
    sam_predictor = None
    if save_cutout:
        sam_checkpoint = _resolve_path("models/sam/sam_vit_h.pth")
        if not sam_checkpoint.is_file():
            raise FileNotFoundError(str(sam_checkpoint))
        resolved_device = ("cuda:0" if args.device == "cuda" else "cpu") if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        sam_predictor = load_sam_predictor(SamModelSpec(checkpoint=sam_checkpoint, model_type="vit_h", device=resolved_device))
        parts_dir.mkdir(parents=True, exist_ok=True)
        cutout_dir.mkdir(parents=True, exist_ok=True)
    if save_label:
        label_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    parts_count = 0
    cutout_count_before = sum(1 for p in cutout_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if cutout_dir.exists() else 0
    for p in car_front_seg.iter_images(stage_dir):
        image_count += 1
        rgb = car_front_seg.load_rgb_with_white_bg(p)
        processed = car_front_seg.detect_processed_instances(
            model=model,
            rgb=rgb,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            allowed_parts=allowed_parts,
        )
        if not processed:
            continue

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if save_label:
            preview = car_front_seg.render_annotated_preview(
                bgr.copy(),
                processed,
                visual_label_edge=bool(args.visual_label_edge),
            )
            car_front_seg._imwrite_cn(label_dir / p.name, preview)
            label_count += 1
        if save_cutout:
            part_dir = parts_dir / p.stem
            before = sum(1 for x in part_dir.glob("*") if x.is_file()) if part_dir.exists() else 0
            car_front_seg.export_rgba_crops(bgr, processed, part_dir, p.stem)
            after = sum(1 for x in part_dir.glob("*") if x.is_file()) if part_dir.exists() else 0
            parts_count += max(0, after - before)
            assert sam_predictor is not None
            run_sam_cutout_from_instances(
                rgb=rgb,
                bgr=bgr,
                instances=processed,
                output_dir=cutout_dir,
                stem=p.stem,
                sam_predictor=sam_predictor,
                box_margin_ratio=0.03,
                keep_largest_component=True,
            )

    cutout_count_after = sum(1 for p in cutout_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if cutout_dir.exists() else 0
    return {
        "images_seen": image_count,
        "labels_written": label_count,
        "front_parts_written": parts_count,
        "img_cutout_written": max(0, cutout_count_after - cutout_count_before),
    }


def _run_atomic_steps(
    *,
    query_path: Path,
    query_name: str,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    query_output_dir = output_dir / query_name
    query_output_dir.mkdir(parents=True, exist_ok=True)
    query_copy = _prepare_query_copy(query_path=query_path, query_dir=query_output_dir)
    config = _load_config(args.weight)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    items, query_item = _prepare_run_images(
        input_dir=input_dir,
        query_image=query_copy,
        output_dir=query_output_dir,
        run_id=run_id,
    )

    run_root = query_output_dir / "runs" / run_id
    label_dir = query_output_dir / "front_label" / run_id
    parts_dir = query_output_dir / "front_parts" / run_id
    cutout_dir = query_output_dir / "img-cutout" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    parts_used = _parse_csv(args.parts) or list(config.get("parts") or [])
    ignored = _parse_csv(args.ignore_parts)
    allowed_parts = set(p for p in parts_used if p not in set(ignored))

    steps: dict[str, Any] = {}
    if args.label_only or args.cutout_only:
        steps["parts"] = _run_part_atomic_steps(
            stage_dir=items[0].staged_path.parent,
            label_dir=label_dir,
            parts_dir=parts_dir,
            cutout_dir=cutout_dir,
            config=config,
            allowed_parts=allowed_parts,
            args=args,
            save_label=bool(args.label_only),
            save_cutout=bool(args.cutout_only),
        )

    contour_count = 0
    if args.contour_only:
        contour = _run_contour_compare(items=items, query_item=query_item, output_dir=run_root, config=config)
        contour_count = sum(1 for x in contour.values() if x.get("status") == "ok")
        steps["contour"] = {
            "items": contour,
            "ok_count": contour_count,
        }

    output_paths = {
        "front_label": str(label_dir) if args.label_only else None,
        "front_parts": str(parts_dir) if args.cutout_only else None,
        "img_cutout": str(cutout_dir) if args.cutout_only else None,
        "run_root": str(run_root) if args.contour_only else None,
    }
    item = {
        "query_name": query_name,
        "query_path": str(query_path),
        "query_copy": str(query_copy),
        "output_dir": str(query_output_dir),
        "status": "ok",
        "mode": "atomic",
        "selected_steps": {
            "label_only": bool(args.label_only),
            "cutout_only": bool(args.cutout_only),
            "contour_only": bool(args.contour_only),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "run_id": run_id,
        "query_id": query_item.item_id,
        "gallery_count": sum(1 for x in items if x.role == "gallery"),
        "outputs": {k: v for k, v in output_paths.items() if v},
        "steps": steps,
    }
    (query_output_dir / "batch_item.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return item


def _precompute_yolo_sam_cache(
    *,
    input_dir: Path,
    output_dir: Path,
    images: list[Path],
    names: dict[Path, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    cache_root = output_dir / "_precompute"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    stage_dir = cache_root / "input_flat"
    label_dir = cache_root / "front_label"
    parts_dir = cache_root / "front_parts"
    cutout_dir = cache_root / "img-cutout"
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged_paths: dict[str, str] = {}
    for image in images:
        image_id = names[image]
        staged = stage_dir / f"{image_id}{image.suffix.lower()}"
        shutil.copy2(image, staged)
        staged_paths[str(image)] = str(staged)

    config = _load_config(args.weight)
    parts_used = _parse_csv(args.parts) or list(config.get("parts") or [])
    ignored = set(_parse_csv(args.ignore_parts))
    allowed_parts = set(p for p in parts_used if p not in ignored)
    stats = _run_part_atomic_steps(
        stage_dir=stage_dir,
        label_dir=label_dir,
        parts_dir=parts_dir,
        cutout_dir=cutout_dir,
        config=config,
        allowed_parts=allowed_parts,
        args=args,
        save_label=True,
        save_cutout=True,
    )
    payload = {
        "input_dir": str(input_dir),
        "cache_root": str(cache_root),
        "stage_dir": str(stage_dir),
        "front_label": str(label_dir),
        "front_parts": str(parts_dir),
        "img_cutout": str(cutout_dir),
        "image_count": len(images),
        "elapsed_seconds": round(time.time() - started, 3),
        "stats": stats,
        "items": [
            {
                "image_id": names[image],
                "original_path": str(image),
                "staged_path": staged_paths[str(image)],
            }
            for image in images
        ],
    }
    (cache_root / "precompute_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _cached_items_from_precompute(precompute: dict[str, Any], names: dict[Path, str]) -> list[ImageItem]:
    by_original = {str(item["original_path"]): str(item["staged_path"]) for item in precompute.get("items") or []}
    out: list[ImageItem] = []
    for path, image_id in names.items():
        out.append(
            ImageItem(
                item_id=image_id,
                role="gallery",
                original_path=path,
                staged_path=Path(by_original[str(path)]),
            )
        )
    return out


def _run_cached_pairwise_query(
    *,
    query_path: Path,
    query_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    precompute = args._precompute
    names: dict[Path, str] = args._names
    query_output_dir = output_dir / query_name
    query_output_dir.mkdir(parents=True, exist_ok=True)

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = query_output_dir / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    items = _cached_items_from_precompute(precompute, names)
    query_gallery_item = next(x for x in items if x.original_path == query_path)
    query_item = ImageItem(
        item_id=query_gallery_item.item_id,
        role="query",
        original_path=query_gallery_item.original_path,
        staged_path=query_gallery_item.staged_path,
    )
    compare_items = [query_item, *items]

    config = _load_config(args.weight)
    parts_used = _parse_csv(args.parts) or list(config.get("parts") or [])
    ignored = _parse_csv(args.ignore_parts)
    feature_names = _feature_list(config, args.features)
    contour = _run_contour_compare(items=compare_items, query_item=query_item, output_dir=run_root, config=config)
    results = _compare_parts(
        cutout_dir=Path(str(precompute["img_cutout"])),
        items=compare_items,
        query_item=query_item,
        contour=contour,
        config=config,
        parts=parts_used,
        ignored_parts=ignored,
        features=feature_names,
        device=args.device,
        max_workers=int(args.compare_workers),
    )
    if args.topk and args.topk > 0:
        results = results[: int(args.topk)]

    output_paths = {
        "front_label": str(precompute["front_label"]),
        "front_parts": str(precompute["front_parts"]),
        "img_cutout": str(precompute["img_cutout"]),
        "run_root": str(run_root),
    }
    report_paths = _write_reports(
        report_dir=query_output_dir / "reports",
        run_id=run_id,
        query_item=query_item,
        items=compare_items,
        results=results,
        parts=parts_used,
        features=[str(x) for x in feature_names],
        output_paths=output_paths,
    )
    item = {
        "query_name": query_name,
        "query_path": str(query_path),
        "query_copy": str(query_item.staged_path),
        "output_dir": str(query_output_dir),
        "status": "ok",
        "mode": "cached_pairwise",
        "elapsed_seconds": round(time.time() - started, 3),
        "run_id": run_id,
        "gallery_count": len(items),
        "precompute_root": str(precompute["cache_root"]),
        "report_json": report_paths.get("json"),
        "report_markdown": report_paths.get("markdown"),
        "top_results": [
            {
                "candidate_id": row.get("candidate_id"),
                "candidate_path": row.get("candidate_path"),
                "final_score": row.get("final_score"),
                "contour_score": row.get("contour_score"),
                "part_score": row.get("part_score"),
            }
            for row in results[:5]
        ],
    }
    (query_output_dir / "batch_item.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return item


def _run_one_query(
    *,
    query_path: Path,
    query_name: str,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    query_output_dir = output_dir / query_name
    query_output_dir.mkdir(parents=True, exist_ok=True)

    if _atomic_mode_enabled(args):
        return _run_atomic_steps(
            query_path=query_path,
            query_name=query_name,
            input_dir=input_dir,
            output_dir=output_dir,
            args=args,
        )
    if getattr(args, "_precompute", None) is not None:
        return _run_cached_pairwise_query(
            query_path=query_path,
            query_name=query_name,
            output_dir=output_dir,
            args=args,
        )

    query_copy = _prepare_query_copy(query_path=query_path, query_dir=query_output_dir)
    payload = run_pipeline(
        input_dir=input_dir,
        query_image=query_copy,
        weight=args.weight,
        output_dir=query_output_dir,
        parts=args.parts,
        ignore_parts=args.ignore_parts,
        features=args.features,
        topk=args.topk if args.topk and args.topk > 0 else None,
        device=args.device,
        skip_seg=bool(args.skip_seg),
        skip_cutout=bool(args.skip_cutout),
        conf=float(args.conf),
        iou=float(args.iou),
        imgsz=int(args.imgsz),
        visual_label_edge=bool(args.visual_label_edge),
        compare_workers=int(args.compare_workers),
    )

    item = {
        "query_name": query_name,
        "query_path": str(query_path),
        "query_copy": str(query_copy),
        "output_dir": str(query_output_dir),
        "status": "ok",
        "mode": "full_pipeline",
        "elapsed_seconds": round(time.time() - started, 3),
        "run_id": payload.get("run_id"),
        "gallery_count": payload.get("gallery_count"),
        "report_json": (payload.get("reports") or {}).get("json"),
        "report_markdown": (payload.get("reports") or {}).get("markdown"),
        "top_results": [
            {
                "candidate_id": row.get("candidate_id"),
                "candidate_path": row.get("candidate_path"),
                "final_score": row.get("final_score"),
                "contour_score": row.get("contour_score"),
                "part_score": row.get("part_score"),
            }
            for row in (payload.get("results") or [])[:5]
        ],
    }
    (query_output_dir / "batch_item.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return item


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pairwise batch validation by calling similarity_pipeline for each query image."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing images to compare.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Batch output root.")
    parser.add_argument("--weight", default=str(DEFAULT_CONFIG), help="Pipeline config JSON or weight path.")
    parser.add_argument("--workers", type=int, default=2, help="Number of query pipelines to run concurrently.")
    parser.add_argument("--compare-workers", type=int, default=1, help="Per-pipeline candidate comparison threads.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of query images for smoke tests.")
    parser.add_argument("--topk", type=int, default=None, help="Optional top-k truncation. Omit or set 0 to keep all.")
    parser.add_argument("--parts", default=None, help="Comma-separated parts used by the pipeline.")
    parser.add_argument("--ignore-parts", default="", help="Comma-separated parts to ignore.")
    parser.add_argument("--features", default=None, help="Comma-separated features, e.g. dino,ssim,edge.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--skip-seg", action="store_true")
    parser.add_argument("--skip-cutout", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--visual-label-edge", action="store_true")
    parser.add_argument("--cutout-only", nargs="?", const=True, default=False, type=_str_to_bool, help="Run only the part crop/cutout atomic step. Can be combined with other --*-only flags.")
    parser.add_argument("--label-only", nargs="?", const=True, default=False, type=_str_to_bool, help="Run only the vehicle/part annotation label atomic step. Can be combined with other --*-only flags.")
    parser.add_argument("--contour-only", nargs="?", const=True, default=False, type=_str_to_bool, help="Run only the whole-car contour comparison atomic step. Can be combined with other --*-only flags.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_dir = _resolve_path(args.input_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(input_dir)
    if args.limit is not None:
        images = images[: max(0, int(args.limit))]
    if not images:
        raise SystemExit(f"no images found in {input_dir}")

    workers = max(1, int(args.workers))
    names = _unique_job_names(images)
    started = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    precompute: dict[str, Any] | None = None

    print(f"batch input={input_dir}")
    print(f"batch output={output_dir}")
    print(f"queries={len(images)} gallery_per_query={len(_collect_images(input_dir))} workers={workers}")
    if _atomic_mode_enabled(args):
        print(
            "atomic mode="
            f"label:{bool(args.label_only)} "
            f"cutout:{bool(args.cutout_only)} "
            f"contour:{bool(args.contour_only)}"
        )
    elif not args.skip_seg and not args.skip_cutout:
        print("precompute=yolo/sam cache enabled")
        precompute = _precompute_yolo_sam_cache(
            input_dir=input_dir,
            output_dir=output_dir,
            images=images,
            names=names,
            args=args,
        )
        args._precompute = precompute
        args._names = names
        print(f"precompute summary={Path(str(precompute['cache_root'])) / 'precompute_summary.json'}")
    else:
        args._precompute = None
        args._names = names

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _run_one_query,
                query_path=image,
                query_name=names[image],
                input_dir=input_dir,
                output_dir=output_dir,
                args=args,
            ): image
            for image in images
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            image = future_map[future]
            try:
                item = future.result()
                results.append(item)
                print(f"[{index}/{len(images)}] ok {item['query_name']} elapsed={item['elapsed_seconds']}s")
            except Exception as exc:
                failed = {
                    "query_name": names[image],
                    "query_path": str(image),
                    "status": "failed",
                    "error": str(exc),
                }
                failures.append(failed)
                print(f"[{index}/{len(images)}] failed {failed['query_name']}: {exc}", file=sys.stderr)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "query_count": len(images),
        "gallery_per_query": len(_collect_images(input_dir)),
        "workers": workers,
        "compare_workers": int(args.compare_workers),
        "mode": "atomic" if _atomic_mode_enabled(args) else ("cached_pairwise" if precompute is not None else "full_pipeline"),
        "selected_steps": {
            "label_only": bool(args.label_only),
            "cutout_only": bool(args.cutout_only),
            "contour_only": bool(args.contour_only),
        },
        "precompute": precompute,
        "elapsed_seconds": round(time.time() - started, 3),
        "ok_count": len(results),
        "failed_count": len(failures),
        "items": sorted(results, key=lambda x: str(x.get("query_name") or "")),
        "failures": failures,
    }
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"summary={output_dir / 'batch_summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
