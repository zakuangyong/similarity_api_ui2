from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    img_root: Path
    result_root: Path
    cors_allow_origins: list[str]
    retrieval_mode: str
    v4_gallery_root: Path
    v4_project_root: Path
    v4_preprocess_config: Path
    v4_query_store: Path
    v4_device: str
    v4_candidate_k: int
    v4_fallback_on_error: bool


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[1]
    img_root = Path(os.environ.get("SIM_IMG_ROOT", str(repo_root / "img"))).expanduser()
    if not img_root.is_absolute():
        img_root = (repo_root / img_root).resolve()

    result_root = Path(os.environ.get("SIM_RESULT_ROOT", str(repo_root / "result"))).expanduser()
    if not result_root.is_absolute():
        result_root = (repo_root / result_root).resolve()

    cors = os.environ.get("SIM_CORS_ALLOW_ORIGINS", "")
    allow_origins = [x.strip() for x in cors.split(",") if x.strip()]
    if not allow_origins:
        allow_origins = [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]

    retrieval_mode = os.environ.get("SIM_RETRIEVAL_MODE", "legacy").strip().lower()
    if retrieval_mode not in {"legacy", "v4", "shadow"}:
        raise ValueError("SIM_RETRIEVAL_MODE must be legacy, v4, or shadow")

    v4_project_root = _resolve_path(
        os.environ.get("SIM_V4_PROJECT_ROOT"),
        repo_root.parent / "similarity_teach_data_pre",
        repo_root,
    )
    v4_gallery_root = _resolve_path(
        os.environ.get("SIM_V4_GALLERY_ROOT"),
        repo_root / "gallery_store",
        repo_root,
    )
    v4_preprocess_config = _resolve_path(
        os.environ.get("SIM_V4_PREPROCESS_CONFIG"),
        v4_project_root / "configs" / "default.yaml",
        repo_root,
    )
    v4_query_store = _resolve_path(
        os.environ.get("SIM_V4_QUERY_STORE"),
        result_root / "v4-query-cache",
        repo_root,
    )
    try:
        v4_candidate_k = int(os.environ.get("SIM_V4_CANDIDATE_K", "100"))
    except ValueError as exc:
        raise ValueError("SIM_V4_CANDIDATE_K must be an integer") from exc
    if v4_candidate_k <= 0:
        raise ValueError("SIM_V4_CANDIDATE_K must be positive")

    return Settings(
        repo_root=repo_root,
        img_root=img_root,
        result_root=result_root,
        cors_allow_origins=allow_origins,
        retrieval_mode=retrieval_mode,
        v4_gallery_root=v4_gallery_root,
        v4_project_root=v4_project_root,
        v4_preprocess_config=v4_preprocess_config,
        v4_query_store=v4_query_store,
        v4_device=os.environ.get("SIM_V4_DEVICE", "auto").strip() or "auto",
        v4_candidate_k=v4_candidate_k,
        v4_fallback_on_error=_env_bool("SIM_V4_FALLBACK_ON_ERROR", True),
    )


def _resolve_path(value: str | None, default: Path, repo_root: Path) -> Path:
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")

