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

    return Settings(repo_root=repo_root, img_root=img_root, result_root=result_root, cors_allow_origins=allow_origins)

