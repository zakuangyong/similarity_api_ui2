from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline_api import (
    build_candidate_detail,
    compare_from_report,
    list_gallery_images,
    load_report,
    path_to_url,
    run_compare,
)
from .settings import load_settings


def create_app() -> FastAPI:
    settings = load_settings()

    app = FastAPI(title="similarity-api-ui2 backend")
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

    @app.get("/health")
    def health(deep: bool = False):
        out = {
            "ok": True,
            "python": sys.executable,
            "img_root": str(settings.img_root),
            "result_root": str(settings.result_root),
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
    ):
        try:
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
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / f"query{suffix}"
                with tmp.open("wb") as f:
                    shutil.copyfileobj(query_image.file, f)
                out = run_compare(
                    query_image_path=tmp,
                    view=view,
                    vehicle_type=vehicle_type,
                    topk=int(topk),
                    img_root=settings.img_root,
                    result_root=settings.result_root,
                )
                return JSONResponse(out)
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
        input_dir = (settings.img_root / view).resolve() if view else settings.img_root
        items = []
        for p in list_gallery_images(input_dir):
            url = path_to_url(path=p, img_root=settings.img_root, result_root=settings.result_root) or ""
            items.append({"id": p.stem, "name": p.stem, "url": url})
        return {"total": len(items), "items": items, "view": view, "vehicle_type": vehicle_type}

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


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
