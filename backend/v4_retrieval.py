from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from tools.car_view_cls import DEFAULT_VIEW_CLS_WEIGHT, predict_vehicle_view


COMPONENTS = (
    "vehicle",
    "front_glass",
    "front_right_light",
    "front_bumper",
    "grille",
    "hood",
    "right_mirror",
)

COMPONENT_LABELS = {
    "vehicle": "整车",
    "front_glass": "前挡风玻璃",
    "front_right_light": "右车灯",
    "front_bumper": "前保险杠",
    "grille": "中网",
    "hood": "引擎盖",
    "right_mirror": "右后视镜",
}

TRAINABLE_STATUSES = {"valid", "low_quality"}


class V4UnsupportedViewError(RuntimeError):
    pass


@dataclass(frozen=True)
class V4RuntimeConfig:
    gallery_root: Path
    project_root: Path
    preprocess_config: Path
    query_store: Path
    result_root: Path
    device: str = "auto"
    candidate_k: int = 100


class V4GalleryBundle:
    def __init__(
        self,
        *,
        root: Path,
        index: Any,
        rows: Any,
        vectors: np.ndarray,
        valid_mask: np.ndarray,
        manifest: dict[str, Any],
        calibration: dict[str, Any] | None,
    ) -> None:
        self.root = root
        self.index = index
        self.rows = rows
        self.vectors = vectors
        self.valid_mask = valid_mask
        self.manifest = manifest
        self.calibration = calibration
        self.component_slices = {
            name: tuple(int(value) for value in manifest["component_slices"][name])
            for name in COMPONENTS
        }
        self.component_weights = {
            name: float(manifest["component_weights"][name]) for name in COMPONENTS
        }

    @classmethod
    def load(cls, gallery_root: str | Path, *, faiss_module: Any | None = None):
        root = Path(gallery_root).resolve()
        index_dir = root / "indexes" / "v4"
        required_paths = {
            "checkpoint": index_dir / "best.pt",
            "manifest": index_dir / "manifest.json",
            "vectors": index_dir / "vector_score.f16.npy",
            "valid_mask": index_dir / "valid_mask.u8.npy",
            "rows": index_dir / "gallery_manifest.parquet",
            "faiss": index_dir / "faiss_score.index",
        }
        missing = [str(path) for path in required_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("V4 deployment bundle is incomplete: " + ", ".join(missing))

        if faiss_module is None:
            try:
                import faiss as faiss_module
            except ImportError as exc:
                raise RuntimeError("FAISS is required for V4 retrieval") from exc
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas and pyarrow are required for V4 retrieval") from exc

        manifest = json.loads(required_paths["manifest"].read_text(encoding="utf-8"))
        if tuple(manifest.get("component_order", ())) != COMPONENTS:
            raise ValueError("V4 manifest component order does not match the runtime")
        if bool(manifest.get("score_vector_normalized")):
            raise ValueError("V4 vector_score must not be L2-normalized")
        if int(manifest.get("vector_dim", 0)) != 640:
            raise ValueError("V4 vector dimension must be 640")
        if _sha256(required_paths["checkpoint"]) != manifest.get("checkpoint_sha256"):
            raise ValueError("V4 checkpoint fingerprint does not match vector manifest")

        vectors = np.load(required_paths["vectors"], mmap_mode="r")
        valid_mask = np.load(required_paths["valid_mask"], mmap_mode="r")
        rows = pd.read_parquet(required_paths["rows"]).sort_values(
            "faiss_id", kind="stable"
        ).reset_index(drop=True)
        index = faiss_module.read_index(str(required_paths["faiss"]))
        row_count = int(manifest.get("row_count", -1))
        if vectors.shape != (row_count, 640):
            raise ValueError(f"invalid V4 vector shape: {vectors.shape}")
        if valid_mask.shape != (row_count, len(COMPONENTS)):
            raise ValueError(f"invalid V4 valid-mask shape: {valid_mask.shape}")
        if len(rows) != row_count or int(index.ntotal) != row_count:
            raise ValueError("V4 vectors, gallery manifest, and FAISS index are misaligned")
        if int(index.d) != 640:
            raise ValueError("V4 FAISS dimension must be 640")
        required_columns = {
            "faiss_id",
            "candidate_id",
            "image_id",
            "candidate_name",
            "artifact_path",
            "source_image_path",
            "display_image_path",
        }
        missing_columns = sorted(required_columns - set(rows.columns))
        if missing_columns:
            raise ValueError(f"gallery manifest is missing columns: {missing_columns}")
        expected_ids = np.arange(row_count)
        if not np.array_equal(rows["faiss_id"].to_numpy(), expected_ids):
            raise ValueError("gallery manifest faiss_id must be contiguous and ordered")
        if rows["candidate_id"].astype(str).duplicated().any():
            raise ValueError("gallery manifest contains duplicate candidate_id")

        calibration_path = index_dir / "score_calibration.json"
        calibration = (
            json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration_path.is_file()
            else None
        )
        return cls(
            root=root,
            index=index,
            rows=rows,
            vectors=vectors,
            valid_mask=valid_mask,
            manifest=manifest,
            calibration=calibration,
        )

    def search(self, query: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        if values.shape != (1, 640) or not np.isfinite(values).all():
            raise ValueError("query vector must be one finite 640D vector")
        distances, indexes = self.index.search(values, min(max(1, count), len(self.rows)))
        return distances[0], indexes[0]

    def calibrate(self, value: float) -> float:
        if not self.calibration:
            return float(np.clip(value, 0.0, 1.0))
        if "x" in self.calibration and "y" in self.calibration:
            x_values = np.asarray(self.calibration["x"], dtype=np.float64)
            y_values = np.asarray(self.calibration["y"], dtype=np.float64)
            if len(x_values) >= 2 and len(x_values) == len(y_values):
                return float(np.clip(np.interp(value, x_values, y_values), 0.0, 1.0))
        slope = float(self.calibration.get("slope", 1.0))
        intercept = float(self.calibration.get("intercept", 0.0))
        return float(np.clip(value * slope + intercept, 0.0, 1.0))

    def resolve_path(self, value: object) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def asset_url(self, path: Path) -> str:
        for directory_name, prefix in (
            ("display-images", "/assets/v4-display"),
            ("source-images", "/assets/v4-source"),
        ):
            directory = (self.root / directory_name).resolve()
            try:
                relative = path.resolve().relative_to(directory)
            except ValueError:
                continue
            encoded = "/".join(quote(part) for part in relative.parts)
            return f"{prefix}/{encoded}"
        raise ValueError(f"gallery image is outside deployed static directories: {path}")


class V4RetrievalRuntime:
    def __init__(self, config: V4RuntimeConfig) -> None:
        self.config = config
        self._state_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._ready = False
        self._initialization_error: str | None = None
        self._gallery: V4GalleryBundle | None = None
        self._torch: Any = None
        self._device: Any = None
        self._model: Any = None
        self._model_config: Any = None
        self._part_segmenter: Any = None
        self._vehicle_segmenter: Any = None
        self._feature_encoder: Any = None
        self._run_preprocess: Any = None
        self._image_artifact_type: Any = None

    @property
    def status(self) -> dict[str, Any]:
        gallery = self._gallery
        return {
            "ready": self._ready,
            "error": self._initialization_error,
            "device": str(self._device) if self._device is not None else None,
            "gallery_count": int(gallery.index.ntotal) if gallery is not None else 0,
            "model_fingerprint": (
                gallery.manifest.get("model_fingerprint") if gallery is not None else None
            ),
        }

    def initialize(self) -> None:
        with self._state_lock:
            if self._ready:
                return
            try:
                self._initialize_impl()
            except Exception as exc:
                self._initialization_error = f"{type(exc).__name__}: {exc}"
                raise
            self._initialization_error = None
            self._ready = True

    def _initialize_impl(self) -> None:
        source_root = self.config.project_root / "src"
        if not source_root.is_dir():
            raise FileNotFoundError(f"V4 project source not found: {source_root}")
        source_value = str(source_root)
        if source_value not in sys.path:
            sys.path.insert(0, source_value)

        try:
            import torch
            from similarity_teach_data_pre.config import load_settings as load_preprocess_settings
            from similarity_teach_data_pre.extractors.birefnet_vehicle_segmenter import (
                BiRefNetVehicleSegmenter,
            )
            from similarity_teach_data_pre.extractors.composite_feature_encoder import (
                CompositeFeatureEncoder,
            )
            from similarity_teach_data_pre.extractors.yolo_sam_part_segmenter import (
                YoloSamPartSegmenter,
            )
            from similarity_teach_data_pre.pipeline.preprocess import run_preprocess
            from similarity_teach_data_pre.schemas import ImageArtifact
            from similarity_teach_train.config import ModelConfig
            from similarity_teach_train.model import StructuredProjectionModel
        except ImportError as exc:
            raise RuntimeError(
                "cannot import V4 preprocessing/training runtime from "
                f"{source_root}: {exc}"
            ) from exc

        gallery = V4GalleryBundle.load(self.config.gallery_root)
        checkpoint_path = self.config.gallery_root / "indexes" / "v4" / "best.pt"
        device = _resolve_device(torch, self.config.device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_payload = dict(checkpoint["training_config"]["model"])
        model_config = ModelConfig(**model_payload)
        model = StructuredProjectionModel(model_config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        preprocess_config = self.config.preprocess_config.resolve()
        if not preprocess_config.is_file():
            raise FileNotFoundError(f"V4 preprocess config not found: {preprocess_config}")
        preprocess_settings = load_preprocess_settings(preprocess_config)
        self.config.query_store.mkdir(parents=True, exist_ok=True)

        self._gallery = gallery
        self._torch = torch
        self._device = device
        self._model = model
        self._model_config = model_config
        self._part_segmenter = YoloSamPartSegmenter.from_settings(preprocess_settings)
        self._vehicle_segmenter = BiRefNetVehicleSegmenter.from_settings(preprocess_settings)
        self._feature_encoder = CompositeFeatureEncoder.from_settings(preprocess_settings)
        self._run_preprocess = run_preprocess
        self._image_artifact_type = ImageArtifact

    def compare(self, query_image_path: str | Path, *, topk: int) -> dict[str, Any]:
        self.initialize()
        if topk <= 0:
            raise ValueError("topk must be positive")
        query_path = Path(query_image_path).resolve()
        if not query_path.is_file():
            raise FileNotFoundError(str(query_path))

        run_id = "v4-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_dir = self.config.result_root / "v4" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        suffix = query_path.suffix or ".jpg"
        staged_query = run_dir / f"query{suffix}"
        shutil.copy2(query_path, staged_query)

        stage_values: list[tuple[str, str, float]] = []
        total_started = time.perf_counter()
        started = time.perf_counter()
        prediction = predict_vehicle_view(staged_query, model_path=DEFAULT_VIEW_CLS_WEIGHT)
        stage_values.append(("view_classification", "视角识别", time.perf_counter() - started))
        if prediction.view != "front":
            raise V4UnsupportedViewError(
                f"V4 currently supports front images only; predicted={prediction.view}"
            )

        with self._inference_lock:
            started = time.perf_counter()
            artifact_path = self._preprocess_query(staged_query)
            features, valid_mask = self._artifact_features(artifact_path)
            stage_values.append(("query_preprocess", "查询图预处理", time.perf_counter() - started))

            started = time.perf_counter()
            projection = self._project(features, valid_mask)
            stage_values.append(("v4_projection", "V4向量化", time.perf_counter() - started))

            started = time.perf_counter()
            gallery = self._require_gallery()
            search_count = max(topk, self.config.candidate_k)
            distances, indexes = gallery.search(projection["vector_score"], search_count)
            stage_values.append(("faiss_search", "FAISS检索", time.perf_counter() - started))

        started = time.perf_counter()
        query_url = _result_url(staged_query, self.config.result_root)
        results, report_results = self._build_results(
            distances=distances,
            indexes=indexes,
            topk=topk,
            query_vector=projection["vector_score"],
            query_valid=valid_mask,
        )
        stage_values.append(("result_build", "结果整理", time.perf_counter() - started))
        timings = _timing_payload(stage_values, time.perf_counter() - total_started)
        model_version = str(gallery.manifest.get("model_fingerprint") or "asset1-v004")

        report = {
            "run_id": run_id,
            "retrieval_mode": "v4",
            "model_version": model_version,
            "view": prediction.view,
            "view_label": prediction.raw_label,
            "query": {
                "id": "query",
                "path": str(staged_query),
                "staged_path": str(staged_query),
                "url": query_url,
                "artifact_path": str(artifact_path),
            },
            "timings": timings,
            "results": report_results,
        }
        self._write_report(report)
        return {
            "run_id": run_id,
            "retrieval_mode": "v4",
            "model_version": model_version,
            "query_name": "上传比对图片",
            "query_staged_path": query_url,
            "query_annotation_url": "",
            "predicted_view": prediction.view,
            "predicted_view_label": prediction.raw_label,
            "timings": timings,
            "results": results,
            "_query_image_path": str(staged_query),
        }

    def _preprocess_query(self, query_path: Path) -> Path:
        result = self._run_preprocess(
            input_dir=str(query_path),
            store_dir=str(self.config.query_store),
            config_path=str(self.config.preprocess_config),
            dry_run=False,
            resume=True,
            part_segmenter=self._part_segmenter,
            vehicle_segmenter=self._vehicle_segmenter,
            feature_encoder=self._feature_encoder,
        )
        if result.failed:
            raise RuntimeError("V4 query preprocessing failed")
        content_sha256 = _sha256(query_path)
        candidates = sorted(
            self.config.query_store.glob(
                f"artifacts/{content_sha256[:2]}/{content_sha256}/*/artifact.json"
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError("V4 query Artifact was not created")
        return candidates[0]

    def _artifact_features(self, artifact_path: Path) -> tuple[np.ndarray, np.ndarray]:
        artifact = self._image_artifact_type.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
        input_dim = int(self._model_config.input_dim)
        features = np.zeros((len(COMPONENTS), input_dim), dtype=np.float32)
        valid_mask = np.zeros(len(COMPONENTS), dtype=np.bool_)
        for index, component_name in enumerate(COMPONENTS):
            component = artifact.vehicle if component_name == "vehicle" else artifact.parts[component_name]
            status = getattr(component.status, "value", str(component.status))
            feature_ref = component.features.dino
            if status not in TRAINABLE_STATUSES or feature_ref is None:
                continue
            feature_path = Path(feature_ref.path)
            if not feature_path.is_absolute():
                feature_path = artifact_path.parent / feature_path
            vector = _load_npz_vector(feature_path, feature_ref.key)
            if vector.size != input_dim or not np.isfinite(vector).all():
                raise ValueError(
                    f"invalid query DINO feature for {component_name}: shape={vector.shape}"
                )
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError(f"zero-norm query DINO feature for {component_name}")
            features[index] = vector / norm
            valid_mask[index] = True
        if not valid_mask[0]:
            raise ValueError("query vehicle DINO feature is unavailable")
        return features, valid_mask

    def _project(self, features: np.ndarray, valid_mask: np.ndarray) -> dict[str, np.ndarray]:
        torch = self._torch
        feature_tensor = torch.from_numpy(features[None]).to(self._device)
        valid_tensor = torch.from_numpy(valid_mask[None].copy()).to(self._device)
        with torch.no_grad():
            output = self._model(feature_tensor, valid_tensor)
        return {
            "vector_score": output.vector_score[0].float().cpu().numpy(),
            "vector_index": output.vector_index[0].float().cpu().numpy(),
        }

    def _build_results(
        self,
        *,
        distances: np.ndarray,
        indexes: np.ndarray,
        topk: int,
        query_vector: np.ndarray,
        query_valid: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        gallery = self._require_gallery()
        api_results: list[dict[str, Any]] = []
        report_results: list[dict[str, Any]] = []
        for raw_score, faiss_id in zip(distances, indexes, strict=True):
            if len(api_results) >= topk:
                break
            faiss_id = int(faiss_id)
            if faiss_id < 0:
                continue
            row = gallery.rows.iloc[faiss_id]
            display_path = gallery.resolve_path(row["display_image_path"])
            source_path = gallery.resolve_path(row["source_image_path"])
            component_scores = self._component_scores(
                query_vector=query_vector,
                query_valid=query_valid,
                gallery_vector=np.asarray(gallery.vectors[faiss_id], dtype=np.float32),
                gallery_valid=np.asarray(gallery.valid_mask[faiss_id], dtype=np.bool_),
            )
            calibrated = gallery.calibrate(float(raw_score))
            final_score = round(calibrated * 100.0, 2)
            vehicle_score = component_scores["vehicle"]
            part_score = _weighted_part_score(
                component_scores, gallery.component_weights
            )
            analysis = _analysis_points(final_score, component_scores)
            candidate_id = str(row["candidate_id"])
            candidate_url = gallery.asset_url(display_path)
            result = {
                "candidate_id": candidate_id,
                "candidate_name": str(row["candidate_name"]),
                "candidate_path": candidate_url,
                "final_score": final_score,
                "vector_score": float(raw_score),
                "contour_score": vehicle_score,
                "part_score": part_score,
                "component_scores": component_scores,
                "contour_diff_image": None,
                "analysis": analysis,
            }
            api_results.append(result)
            report_results.append(
                {
                    **result,
                    "candidate_path": str(source_path),
                    "candidate_url": candidate_url,
                    "artifact_path": str(gallery.resolve_path(row["artifact_path"])),
                    "faiss_id": faiss_id,
                    "image_id": str(row["image_id"]),
                }
            )
        return api_results, report_results

    def _component_scores(
        self,
        *,
        query_vector: np.ndarray,
        query_valid: np.ndarray,
        gallery_vector: np.ndarray,
        gallery_valid: np.ndarray,
    ) -> dict[str, float | None]:
        gallery = self._require_gallery()
        scores: dict[str, float | None] = {}
        for index, component_name in enumerate(COMPONENTS):
            if not query_valid[index] or not gallery_valid[index]:
                scores[component_name] = None
                continue
            start, stop = gallery.component_slices[component_name]
            weight = gallery.component_weights[component_name]
            weighted_dot = float(
                np.dot(query_vector[start:stop], gallery_vector[start:stop])
            )
            score = weighted_dot / max(weight, 1e-8)
            scores[component_name] = round(float(np.clip(score, 0.0, 1.0)) * 100.0, 2)
        return scores

    def _write_report(self, report: dict[str, Any]) -> None:
        reports_dir = self.config.result_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        report_path = reports_dir / f"{report['run_id']}.json"
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, report_path)
        latest = self.config.result_root / "latest_report.json"
        latest_temporary = latest.with_suffix(".tmp")
        latest_temporary.write_text(encoded, encoding="utf-8")
        os.replace(latest_temporary, latest)

    def _require_gallery(self) -> V4GalleryBundle:
        if self._gallery is None:
            raise RuntimeError("V4 gallery is not initialized")
        return self._gallery


def _resolve_device(torch: Any, configured: str):
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SIM_V4_DEVICE requests CUDA but CUDA is unavailable")
    return device


def _load_npz_vector(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    payload = np.load(path, allow_pickle=False)
    try:
        if key not in payload:
            raise KeyError(f"feature key not found: {key}")
        return np.asarray(payload[key], dtype=np.float32).reshape(-1)
    finally:
        close = getattr(payload, "close", None)
        if callable(close):
            close()


def _analysis_points(
    final_score: float, component_scores: dict[str, float | None]
) -> list[str]:
    valid = [(name, value) for name, value in component_scores.items() if value is not None]
    ordered = sorted(valid, key=lambda item: float(item[1]), reverse=True)
    points = [f"V4 综合向量相似度为 {final_score:.1f} 分。"]
    if ordered:
        name, value = ordered[0]
        points.append(f"最相似组件为{COMPONENT_LABELS[name]}，相似度 {value:.1f} 分。")
    if len(ordered) > 1:
        name, value = ordered[-1]
        points.append(f"差异最明显组件为{COMPONENT_LABELS[name]}，相似度 {value:.1f} 分。")
    return points


def _weighted_part_score(
    component_scores: dict[str, float | None],
    component_weights: dict[str, float],
) -> float | None:
    weighted_total = 0.0
    available_weight = 0.0
    for component_name, score in component_scores.items():
        if component_name == "vehicle" or score is None:
            continue
        weight = float(component_weights.get(component_name, 0.0))
        if weight <= 0.0:
            continue
        weighted_total += float(score) * weight
        available_weight += weight
    if available_weight <= 0.0:
        return None
    return round(weighted_total / available_weight, 2)


def _result_url(path: Path, result_root: Path) -> str:
    relative = path.resolve().relative_to(result_root.resolve())
    encoded = "/".join(quote(part) for part in relative.parts)
    return f"/assets/result/{encoded}"


def _timing_payload(
    stages: list[tuple[str, str, float]], total_seconds: float
) -> dict[str, Any]:
    denominator = max(total_seconds, 1e-9)
    values = [
        {
            "name": name,
            "label": label,
            "seconds": round(float(seconds), 4),
            "percent": round(float(seconds) / denominator * 100.0, 2),
            "index": index,
        }
        for index, (name, label, seconds) in enumerate(stages, start=1)
    ]
    bottleneck = max(values, key=lambda item: item["seconds"]) if values else None
    return {
        "total_seconds": round(float(total_seconds), 4),
        "stages": values,
        "bottleneck": bottleneck,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "V4GalleryBundle",
    "V4RetrievalRuntime",
    "V4RuntimeConfig",
    "V4UnsupportedViewError",
]
