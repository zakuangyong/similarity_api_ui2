from __future__ import annotations

import asyncio
import json
import random
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import AsyncIterator
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .pipeline_api import (
    GalleryRoots,
    build_candidate_detail,
    compare_from_report,
    list_gallery_images,
    load_report,
    path_to_url,
    run_compare,
)
from .settings import load_settings
from .v4_retrieval import (
    V4RetrievalRuntime,
    V4RuntimeConfig,
    V4UnsupportedViewError,
)


GALLERY_DISPLAY_DIR_NAME = "img-store"


def create_app(*, settings_override=None, v4_runtime_override=None) -> FastAPI:
    settings = settings_override or load_settings()
    v4_runtime = v4_runtime_override or V4RetrievalRuntime(
        V4RuntimeConfig(
            gallery_root=settings.v4_gallery_root,
            project_root=settings.v4_project_root,
            preprocess_config=settings.v4_preprocess_config,
            query_store=settings.v4_query_store,
            result_root=settings.result_root,
            device=settings.v4_device,
            candidate_k=settings.v4_candidate_k,
        )
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if settings.retrieval_mode in {"v4", "shadow"}:
            try:
                await asyncio.to_thread(v4_runtime.initialize)
            except Exception:
                if not settings.v4_fallback_on_error:
                    raise
        yield

    app = FastAPI(title="similarity-api-ui2 backend", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    settings.img_root.mkdir(parents=True, exist_ok=True)
    settings.result_root.mkdir(parents=True, exist_ok=True)

    app.mount("/assets/gallery", StaticFiles(directory=str(settings.img_root), html=False), name="assets_gallery")
    app.mount("/assets/result", StaticFiles(directory=str(settings.result_root), html=False), name="assets_result")
    for directory_name, route, name in (
        ("display-images", "/assets/v4-display", "assets_v4_display"),
        ("source-images", "/assets/v4-source", "assets_v4_source"),
    ):
        directory = settings.v4_gallery_root / directory_name
        if directory.is_dir():
            app.mount(route, StaticFiles(directory=str(directory), html=False), name=name)

    app.state.v4_runtime = v4_runtime

    @app.get("/health")
    def health(deep: bool = False):
        out = {
            "ok": True,
            "python": sys.executable,
            "img_root": str(settings.img_root),
            "result_root": str(settings.result_root),
            "retrieval_mode": settings.retrieval_mode,
            "v4": v4_runtime.status,
        }
        if deep:
            try:
                import torch  # type: ignore

                out["torch"] = {
                    "version": getattr(torch, "__version__", None),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_version": getattr(torch.version, "cuda", None),
                }
            except Exception as e:
                out["torch"] = {"error": str(e)}
        return out

    @app.post("/api/compare")
    async def api_compare(
        query_image: UploadFile | None = File(None),
        view: str = Form("front"),
        vehicle_type: str = Form("SUV"),
        topk: int = Form(10),
        retrieval_mode: str | None = Form(None),
    ):
        try:
            mode = (retrieval_mode or settings.retrieval_mode).strip().lower()
            if mode not in {"legacy", "v4", "shadow"}:
                raise ValueError("retrieval_mode must be legacy, v4, or shadow")
            if query_image is None:
                try:
                    report = load_report(run_id="latest", result_root=settings.result_root)
                except FileNotFoundError:
                    raise HTTPException(status_code=404, detail="latest_report.json not found")
                out = compare_from_report(report=report, img_root=settings.img_root, result_root=settings.result_root)
                return JSONResponse(out)

            suffix = Path(query_image.filename or "query").suffix
            if not suffix:
                suffix = ".jpg"
            temp_dir = Path(tempfile.mkdtemp(prefix="similarity_compare_"))
            tmp = temp_dir / f"query{suffix}"
            with tmp.open("wb") as f:
                shutil.copyfileobj(query_image.file, f)

            def execute_compare():
                if mode == "legacy":
                    return run_compare(
                        query_image_path=tmp,
                        view=view,
                        vehicle_type=vehicle_type,
                        topk=int(topk),
                        img_root=settings.img_root,
                        result_root=settings.result_root,
                    )
                try:
                    out = v4_runtime.compare(tmp, topk=int(topk))
                    out["vehicle_type"] = vehicle_type
                    out["view"] = view
                    return out
                except V4UnsupportedViewError as exc:
                    out = run_compare(
                        query_image_path=tmp,
                        view=view,
                        vehicle_type=vehicle_type,
                        topk=int(topk),
                        img_root=settings.img_root,
                        result_root=settings.result_root,
                    )
                    out["retrieval_mode"] = "legacy-fallback"
                    out["fallback_reason"] = str(exc)
                    return out
                except Exception as exc:
                    if not settings.v4_fallback_on_error:
                        raise
                    out = run_compare(
                        query_image_path=tmp,
                        view=view,
                        vehicle_type=vehicle_type,
                        topk=int(topk),
                        img_root=settings.img_root,
                        result_root=settings.result_root,
                    )
                    out["retrieval_mode"] = "legacy-fallback"
                    out["fallback_reason"] = f"V4 failed: {type(exc).__name__}: {exc}"
                    return out

            task = asyncio.create_task(asyncio.to_thread(execute_compare))
            task.add_done_callback(lambda _: shutil.rmtree(temp_dir, ignore_errors=True))

            async def stream_result() -> AsyncIterator[bytes]:
                # Send a first byte immediately, then periodic JSON whitespace so
                # upstream proxies do not treat a long GPU job as an idle origin.
                yield b"\n"
                while not task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
                    except asyncio.TimeoutError:
                        yield b" \n"

                try:
                    out = await task
                    query_path = out.pop("_query_image_path", None)
                    if mode == "shadow" and out.get("retrieval_mode") == "v4" and query_path:
                        asyncio.create_task(
                            _run_shadow_compare(
                                query_image_path=Path(query_path),
                                v4_run_id=str(out.get("run_id") or "unknown"),
                                view=view,
                                vehicle_type=vehicle_type,
                                topk=int(topk),
                                img_root=settings.img_root,
                                result_root=settings.result_root,
                            )
                        )
                    yield json.dumps(out, ensure_ascii=False).encode("utf-8")
                except Exception as e:
                    payload = {"detail": f"compare failed: {e}"}
                    yield json.dumps(payload, ensure_ascii=False).encode("utf-8")

            return StreamingResponse(
                stream_result(),
                media_type="application/json",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                },
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("无法导入 similarity_pipeline"):
                raise HTTPException(status_code=503, detail=msg)
            raise HTTPException(status_code=500, detail=msg)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"compare failed: {e}")

    @app.get("/api/gallery")
    def api_gallery(view: str = "front", vehicle_type: str = "SUV"):
        if settings.retrieval_mode in {"v4", "shadow"}:
            display_dir = settings.v4_gallery_root / "display-images"
            display_images = list_gallery_images(display_dir)
            if display_images:
                items = [
                    {
                        "id": path.stem,
                        "name": path.stem,
                        "url": _mounted_asset_url(
                            path, root=display_dir, prefix="/assets/v4-display"
                        ),
                    }
                    for path in display_images
                ]
                random.shuffle(items)
                return {
                    "total": len(items),
                    "items": items,
                    "view": view,
                    "vehicle_type": vehicle_type,
                    "display_dir": str(display_dir),
                }
        roots = GalleryRoots(img_root=settings.img_root)
        input_dir = roots.resolve_view_dir(view)
        if not input_dir.is_dir() or not list_gallery_images(input_dir):
            input_dir = (settings.img_root / GALLERY_DISPLAY_DIR_NAME).resolve()
        items = []
        for p in list_gallery_images(input_dir):
            url = path_to_url(path=p, img_root=settings.img_root, result_root=settings.result_root) or ""
            items.append({"id": p.stem, "name": p.stem, "url": url})
        random.shuffle(items)
        return {
            "total": len(items),
            "items": items,
            "view": view,
            "vehicle_type": vehicle_type,
            "display_dir": GALLERY_DISPLAY_DIR_NAME,
        }

    @app.get("/api/runs/{run_id}/candidates/{candidate_id}")
    def api_candidate_detail(run_id: str, candidate_id: str):
        try:
            report = load_report(run_id=run_id, result_root=settings.result_root)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="report not found")

        try:
            return build_candidate_detail(
                report=report,
                candidate_id=candidate_id,
                img_root=settings.img_root,
                result_root=settings.result_root,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="candidate not found")

    return app


async def _run_shadow_compare(
    *,
    query_image_path: Path,
    v4_run_id: str,
    view: str,
    vehicle_type: str,
    topk: int,
    img_root: Path,
    result_root: Path,
) -> None:
    shadow_root = result_root / "v4-shadow" / v4_run_id
    shadow_root.mkdir(parents=True, exist_ok=True)
    try:
        payload = await asyncio.to_thread(
            run_compare,
            query_image_path=query_image_path,
            view=view,
            vehicle_type=vehicle_type,
            topk=topk,
            img_root=img_root,
            result_root=shadow_root,
        )
    except Exception as exc:
        payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    (shadow_root / "compare.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _mounted_asset_url(path: Path, *, root: Path, prefix: str) -> str:
    relative = path.resolve().relative_to(root.resolve())
    encoded = "/".join(quote(part) for part in relative.parts)
    return f"{prefix}/{encoded}"


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
