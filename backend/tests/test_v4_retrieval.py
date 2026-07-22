from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.pipeline_api import build_candidate_detail, compare_from_report
from backend.settings import load_settings
from backend.v4_retrieval import COMPONENTS, V4GalleryBundle, _weighted_part_score


class _NumpyIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype(np.float32)
        self.ntotal, self.d = self.vectors.shape

    def search(self, query: np.ndarray, count: int):
        scores = query @ self.vectors.T
        indexes = np.argsort(-scores, axis=1)[:, :count]
        distances = np.take_along_axis(scores, indexes, axis=1)
        return distances.astype(np.float32), indexes.astype(np.int64)


class _FakeFaiss:
    def __init__(self, index: _NumpyIndex) -> None:
        self.index = index

    def read_index(self, _):
        return self.index


class _FakeRuntime:
    def __init__(self, query_path: Path) -> None:
        self.query_path = query_path
        self.initialized = 0
        self.compared = 0

    @property
    def status(self):
        return {"ready": True, "gallery_count": 2, "error": None}

    def initialize(self):
        self.initialized += 1

    def compare(self, query_image_path, *, topk):
        self.compared += 1
        return {
            "run_id": "v4-test",
            "retrieval_mode": "v4",
            "model_version": "test-model",
            "query_name": "上传比对图片",
            "query_staged_path": "/assets/result/query.png",
            "predicted_view": "front",
            "predicted_view_label": "front",
            "timings": None,
            "results": [],
            "_query_image_path": str(self.query_path),
        }


def test_gallery_bundle_loads_and_searches_unnormalized_vectors(tmp_path: Path):
    root = tmp_path / "gallery"
    index_dir = root / "indexes" / "v4"
    index_dir.mkdir(parents=True)
    checkpoint = index_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    vectors = np.zeros((2, 640), dtype=np.float16)
    vectors[0, 0] = 0.5
    vectors[1, 0] = 0.9
    np.save(index_dir / "vector_score.f16.npy", vectors)
    np.save(index_dir / "valid_mask.u8.npy", np.ones((2, 7), dtype=np.uint8))
    slices = {}
    offset = 0
    dimensions = (256, 64, 64, 64, 64, 64, 64)
    weights = (0.4, 0.09, 0.132, 0.108, 0.108, 0.09, 0.072)
    for name, dimension in zip(COMPONENTS, dimensions, strict=True):
        slices[name] = [offset, offset + dimension]
        offset += dimension
    manifest = {
        "row_count": 2,
        "vector_dim": 640,
        "component_order": list(COMPONENTS),
        "component_slices": slices,
        "component_weights": dict(zip(COMPONENTS, weights, strict=True)),
        "score_vector_normalized": False,
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (index_dir / "faiss_score.index").write_bytes(b"fake")
    for directory in (root / "display-images", root / "source-images"):
        directory.mkdir(parents=True)
    rows = pd.DataFrame(
        [
            {
                "faiss_id": index,
                "candidate_id": f"candidate-{index}",
                "image_id": f"image-{index}.png",
                "candidate_name": f"image-{index}",
                "artifact_path": f"artifacts/{index}/artifact.json",
                "source_image_path": f"source-images/image-{index}.png",
                "display_image_path": f"display-images/image-{index}.png",
            }
            for index in range(2)
        ]
    )
    rows.to_parquet(index_dir / "gallery_manifest.parquet", index=False)

    bundle = V4GalleryBundle.load(
        root,
        faiss_module=_FakeFaiss(_NumpyIndex(vectors.astype(np.float32))),
    )
    distances, indexes = bundle.search(vectors[0].astype(np.float32), 2)

    assert bundle.index.ntotal == 2
    assert indexes.tolist() == [1, 0]
    assert distances[0] > distances[1]


def test_v4_candidate_detail_uses_projection_components(tmp_path: Path):
    report = {
        "run_id": "v4-run",
        "retrieval_mode": "v4",
        "model_version": "model",
        "query": {"url": "/query.png"},
        "results": [
            {
                "candidate_id": "candidate",
                "candidate_name": "候选车",
                "candidate_url": "/candidate.png",
                "final_score": 82.5,
                "part_score": 80.0,
                "component_scores": {
                    "vehicle": 85.0,
                    "front_glass": 75.0,
                    "front_right_light": 88.0,
                },
                "analysis": ["V4结果"],
            }
        ],
    }

    detail = build_candidate_detail(
        report=report,
        candidate_id="candidate",
        img_root=tmp_path,
        result_root=tmp_path,
    )

    assert detail["retrieval_mode"] == "v4"
    assert detail["summary"]["final_score"] == 82.5
    assert detail["contour"]["score"] == 85.0
    assert len(detail["parts"]["evidence"]) == 2


def test_weighted_part_score_renormalizes_available_components():
    scores = {
        "vehicle": 95.0,
        "front_glass": 80.0,
        "front_right_light": 50.0,
        "front_bumper": None,
    }
    weights = {
        "vehicle": 0.4,
        "front_glass": 0.1,
        "front_right_light": 0.2,
        "front_bumper": 0.3,
    }

    assert _weighted_part_score(scores, weights) == 60.0


def test_compare_from_v4_report_preserves_deployed_urls(tmp_path: Path):
    report = {
        "run_id": "v4-run",
        "retrieval_mode": "v4",
        "model_version": "model",
        "query": {
            "path": str(tmp_path / "query.png"),
            "url": "/assets/result/v4/query.png",
        },
        "results": [
            {
                "candidate_id": "candidate",
                "candidate_name": "候选车",
                "candidate_path": "/gallery/source-images/candidate.png",
                "candidate_url": "/assets/v4-display/candidate.png",
                "final_score": 88.0,
                "vector_score": 0.88,
                "component_scores": {"vehicle": 90.0},
                "contour_score": 90.0,
                "part_score": 80.0,
            }
        ],
    }

    response = compare_from_report(
        report=report,
        img_root=tmp_path / "img",
        result_root=tmp_path / "result",
    )

    assert response["retrieval_mode"] == "v4"
    assert response["query_staged_path"] == "/assets/result/v4/query.png"
    assert response["results"][0]["candidate_name"] == "候选车"
    assert response["results"][0]["candidate_path"] == "/assets/v4-display/candidate.png"
    assert response["results"][0]["component_scores"] == {"vehicle": 90.0}


def test_compare_endpoint_can_select_v4_without_legacy_pipeline(tmp_path: Path):
    base = load_settings()
    img_root = tmp_path / "img"
    result_root = tmp_path / "result"
    gallery_root = tmp_path / "gallery"
    img_root.mkdir()
    result_root.mkdir()
    (gallery_root / "display-images").mkdir(parents=True)
    (gallery_root / "source-images").mkdir()
    query_path = result_root / "persisted-query.png"
    query_path.write_bytes(b"query")
    settings = replace(
        base,
        img_root=img_root,
        result_root=result_root,
        retrieval_mode="v4",
        v4_gallery_root=gallery_root,
        v4_query_store=result_root / "query-cache",
        v4_fallback_on_error=False,
    )
    runtime = _FakeRuntime(query_path)
    app = create_app(settings_override=settings, v4_runtime_override=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/api/compare",
            data={"topk": "5", "retrieval_mode": "v4"},
            files={"query_image": ("query.png", b"image", "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["retrieval_mode"] == "v4"
    assert runtime.initialized == 1
    assert runtime.compared == 1
