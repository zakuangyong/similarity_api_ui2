from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEW_CLS_WEIGHT = ROOT / "models" / "yolo-view-cls" / "yolo11m-cls-for-car-view-train7.pt"

VIEW_ALIAS_TO_FAMILY = {
    "front": "front",
    "front_left_side45": "front",
    "front_right_side45": "front",
    "back": "back",
    "back_left_side45": "back",
    "back_right_side45": "back",
    "left_side": "left_side",
    "right_side": "right_side",
    "rear": "back",
}

VIEW_FAMILY_FALLBACKS = {
    "left_side": "front",
    "right_side": "front",
}


@dataclass(frozen=True)
class ViewPrediction:
    raw_label: str
    view: str
    confidence: float | None
    model_path: Path
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_label": self.raw_label,
            "view": self.view,
            "confidence": self.confidence,
            "model_path": str(self.model_path),
            "fallback_used": self.fallback_used,
        }


def _default_device() -> str | int:
    return 0 if torch.cuda.is_available() else "cpu"


def normalize_view_family(value: str | None) -> str:
    label = (value or "").strip().lower()
    if not label:
        return "front"
    if label in VIEW_ALIAS_TO_FAMILY:
        return VIEW_ALIAS_TO_FAMILY[label]
    if "front" in label:
        return "front"
    if "back" in label or "rear" in label:
        return "back"
    if "left" in label:
        return "left_side"
    if "right" in label:
        return "right_side"
    return label


def resolve_view_family(value: str | None, available_dirs: set[str] | None = None) -> tuple[str, bool]:
    family = normalize_view_family(value)
    if not available_dirs:
        return family, False

    available = {x.strip().lower() for x in available_dirs if x}
    if family in available:
        return family, False

    fallback = VIEW_FAMILY_FALLBACKS.get(family)
    if fallback and fallback in available:
        return fallback, True
    if "front" in available:
        return "front", True
    if "back" in available:
        return "back", True
    return family, True


def predict_vehicle_view(
    image_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_VIEW_CLS_WEIGHT,
    imgsz: int = 224,
    device: str | int | None = None,
) -> ViewPrediction:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(str(model_path))

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(str(image_path))

    if device is None:
        device = _default_device()

    model = _load_model(str(model_path.resolve()))
    results = model.predict(source=str(image_path), imgsz=int(imgsz), device=device, verbose=False)
    if not results:
        return ViewPrediction(raw_label="front", view="front", confidence=None, model_path=model_path)

    result = results[0]
    probs = getattr(result, "probs", None)
    names = getattr(model, "names", {}) or {}
    top1 = int(getattr(probs, "top1", -1)) if probs is not None else -1
    confidence = None
    if probs is not None:
        try:
            confidence = float(getattr(probs, "top1conf", None))
        except Exception:
            confidence = None
        if confidence is None:
            try:
                confidence = float(probs.data[top1]) if top1 >= 0 else None
            except Exception:
                confidence = None

    if isinstance(names, dict):
        raw_label = str(names.get(top1, f"class_{top1}" if top1 >= 0 else "unknown"))
    else:
        try:
            raw_label = str(names[top1]) if top1 >= 0 else "unknown"
        except Exception:
            raw_label = f"class_{top1}" if top1 >= 0 else "unknown"

    view = normalize_view_family(raw_label)
    return ViewPrediction(
        raw_label=raw_label,
        view=view,
        confidence=confidence,
        model_path=model_path,
        fallback_used=False,
    )


@lru_cache(maxsize=4)
def _load_model(model_path: str):
    return YOLO(model_path)
