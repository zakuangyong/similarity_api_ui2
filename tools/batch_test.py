from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from similarity_pipeline import DEFAULT_CONFIG, IMAGE_EXTS, ROOT, _safe_name, run_pipeline  # noqa: E402


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

    print(f"batch input={input_dir}")
    print(f"batch output={output_dir}")
    print(f"queries={len(images)} gallery_per_query={len(_collect_images(input_dir))} workers={workers}")

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
