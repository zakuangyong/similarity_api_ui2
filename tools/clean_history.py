from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _sort_by_mtime_desc(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except Exception:
            return 0.0

    return sorted(paths, key=key, reverse=True)


def _delete_path(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)


def _plan_run_dirs(root: Path, keep_latest: int) -> list[Path]:
    if not root.is_dir():
        return []
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    ordered = _sort_by_mtime_desc(subdirs)
    keep = max(0, int(keep_latest))
    return ordered[keep:]


def _plan_report_files(reports_dir: Path, *, keep_latest: int) -> list[Path]:
    if not reports_dir.is_dir():
        return []
    files = [p for p in reports_dir.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".md"}]
    ordered = _sort_by_mtime_desc(files)
    keep = max(0, int(keep_latest))
    return ordered[keep:]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="清理 ./result 目录中的历史产物与缓存。默认只预览，不会实际删除。")
    parser.add_argument(
        "--result-dir",
        default=str(REPO_ROOT / "result"),
        help="result 输出根目录（默认：./result）",
    )
    parser.add_argument(
        "--scope",
        choices=(
            "cache",
            "tiles",
            "runs",
            "work",
            "front_label",
            "front_parts",
            "img-cutout",
            "reports",
            "all",
        ),
        default="all",
        help="清理范围：cache=仅缓存(tiles/batch_test)，all=全部；默认 all。",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=0,
        help="对按 run_id 分组的目录（runs/work/front_* 等）保留最近 N 个 run_id，其余删除；默认 0 表示不保留。",
    )
    parser.add_argument(
        "--keep-report-latest",
        type=int,
        default=0,
        help="在清理 reports 时保留最近 N 个报告文件（按修改时间排序）；默认 0。",
    )
    parser.add_argument(
        "--keep-reports",
        action="store_true",
        help="scope=all 时保留 reports/ 与 latest_report.*",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="确认执行删除（不带该参数仅打印将要删除的路径）。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result_dir = _resolve_path(args.result_dir, base=REPO_ROOT)
    if not result_dir.exists():
        print(f"result dir not found: {result_dir}")
        return 0
    if not result_dir.is_dir():
        raise SystemExit(f"result dir is not a directory: {result_dir}")

    keep_latest = max(0, int(args.keep_latest))
    keep_report_latest = max(0, int(args.keep_report_latest))
    scope = str(args.scope)

    plan: list[Path] = []

    def add_dir_run_based(name: str) -> None:
        plan.extend(_plan_run_dirs(result_dir / name, keep_latest))

    if scope in {"tiles", "cache", "all"}:
        p = result_dir / "tiles"
        if p.exists():
            plan.append(p)
        p = result_dir / "batch_test"
        if p.exists():
            plan.append(p)

    if scope in {"runs", "all"}:
        add_dir_run_based("runs")
    if scope in {"work", "all"}:
        add_dir_run_based("work")
    if scope in {"front_label", "all"}:
        add_dir_run_based("front_label")
    if scope in {"front_parts", "all"}:
        add_dir_run_based("front_parts")
    if scope in {"img-cutout", "all"}:
        add_dir_run_based("img-cutout")

    if scope in {"reports", "all"} and not (scope == "all" and bool(args.keep_reports)):
        reports_dir = result_dir / "reports"
        plan.extend(_plan_report_files(reports_dir, keep_latest=keep_report_latest))
        if keep_report_latest <= 0 and reports_dir.exists():
            plan.append(reports_dir)
        for fn in ("latest_report.json", "latest_report.md"):
            p = result_dir / fn
            if p.exists():
                plan.append(p)

    if scope == "all" and keep_latest <= 0:
        for child in result_dir.iterdir():
            if child.name in {"reports", "tiles", "batch_test", "runs", "work", "front_label", "front_parts", "img-cutout"}:
                continue
            if child.name in {"latest_report.json", "latest_report.md"}:
                continue
            if child.name == ".gitkeep":
                continue
            plan.append(child)

    planned = []
    for p in _sort_by_mtime_desc(list({x.resolve() for x in plan})):
        if _is_under(p, result_dir):
            planned.append(p)

    if not planned:
        print(f"nothing to clean under: {result_dir}")
        return 0

    print(f"result_dir={result_dir}")
    print(f"scope={scope} keep_latest={keep_latest} keep_report_latest={keep_report_latest} apply={bool(args.yes)}")
    for p in planned:
        rel = str(p.relative_to(result_dir))
        print(f"- {rel}")

    if not args.yes:
        print("dry-run: add --yes to apply")
        return 0

    failed: list[tuple[Path, str]] = []
    for p in planned:
        try:
            _delete_path(p)
        except Exception as exc:
            failed.append((p, str(exc)))

    if failed:
        print("failed:", file=sys.stderr)
        for p, msg in failed:
            print(f"- {p}: {msg}", file=sys.stderr)
        return 1

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

