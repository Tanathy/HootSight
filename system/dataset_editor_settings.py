from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from system.coordinator_settings import ROOT, SETTINGS

def _coerce_int_list(values: Sequence[int] | None, fallback: Sequence[int]) -> List[int]:
    if not values:
        return list(fallback)
    cleaned: List[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            cleaned.append(number)
    return cleaned or list(fallback)

def _coerce_presets(presets: Dict[str, Sequence[int]] | None, fallback: Dict[str, Sequence[int]]) -> Dict[str, List[int]]:
    if not isinstance(presets, dict):
        return {name: list(values) for name, values in fallback.items()}
    result: Dict[str, List[int]] = {}
    for name, values in presets.items():
        cleaned = _coerce_int_list(values, fallback.get(name) or [])
        if cleaned:
            result[name] = cleaned
    if not result:
        return {name: list(values) for name, values in fallback.items()}
    return result

_DATASET_EDITOR_DEFAULTS = {
    "page_sizes": [25, 50, 100, 200, 500, 750],
    "default_page_size": 50,
    "size_presets": {
        "Generic": [64, 128, 224, 256, 320, 384, 512, 640, 768, 1024],
        "YOLO": [320, 416, 512, 608, 640, 768, 896, 1024],
        "ResNet": [160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024],
        "ResNeXt": [160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024],
        "EfficientNet": [224, 240, 260, 300, 380, 456, 528, 600, 672, 800, 1024],
        "MobileNet": [96, 128, 160, 192, 224, 256, 320, 384, 512],
        "SqueezeNet": [128, 192, 224, 256, 320, 384, 512],
        "ShuffleNet": [128, 160, 192, 224, 256, 320, 384, 512],
    },
    "default_size": 1024,
    "build_workers": 4,
    "discovery_workers": 4,
}

@dataclass(frozen=True)
class DatasetEditorSettings:

    projects_dir: Path
    page_sizes: List[int]
    default_page_size: int
    build_workers: int
    discovery_workers: int
    dataset_size_options: Dict[str, List[int]]
    default_dataset_size: int
    image_extensions: List[str]

    @property
    def projects_root(self) -> Path:
        return ROOT / self.projects_dir

_SETTINGS_CACHE: DatasetEditorSettings | None = None

def _resolve_workers(value: int | str | None, fallback: int) -> int:
    if isinstance(value, str) and value.lower() == "auto":
        return max(2, (os.cpu_count() or fallback) - 1)
    if value is None:
        return fallback
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback

def get_dataset_editor_settings() -> DatasetEditorSettings:
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is not None:
        return _SETTINGS_CACHE

    dataset_config = SETTINGS.get("dataset_editor", {}) if isinstance(SETTINGS, dict) else {}
    paths_config = SETTINGS.get("paths", {}) if isinstance(SETTINGS, dict) else {}
    dataset_defaults = SETTINGS.get("dataset", {}) if isinstance(SETTINGS, dict) else {}

    projects_dir_value = paths_config.get("projects_dir", "projects")
    image_extensions = dataset_defaults.get("image_extensions", [])
    if not isinstance(image_extensions, list) or not image_extensions:
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".tiff",
            ".webp",
        ]

    page_sizes = _coerce_int_list(dataset_config.get("page_sizes"), _DATASET_EDITOR_DEFAULTS["page_sizes"])
    default_page_size = int(dataset_config.get("default_page_size", _DATASET_EDITOR_DEFAULTS["default_page_size"]))
    if default_page_size not in page_sizes:
        default_page_size = page_sizes[0]

    dataset_size_options = _coerce_presets(
        dataset_config.get("size_presets"),
        _DATASET_EDITOR_DEFAULTS["size_presets"],
    )
    default_dataset_size = int(dataset_config.get("default_size", _DATASET_EDITOR_DEFAULTS["default_size"]))
    if default_dataset_size <= 0:
        default_dataset_size = _DATASET_EDITOR_DEFAULTS["default_size"]

    build_workers = _resolve_workers(dataset_config.get("build_workers"), _DATASET_EDITOR_DEFAULTS["build_workers"])
    discovery_workers = _resolve_workers(dataset_config.get("discovery_workers"), _DATASET_EDITOR_DEFAULTS["discovery_workers"])

    _SETTINGS_CACHE = DatasetEditorSettings(
        projects_dir=Path(projects_dir_value),
        page_sizes=page_sizes,
        default_page_size=default_page_size,
        build_workers=build_workers,
        discovery_workers=discovery_workers,
        dataset_size_options=dataset_size_options,
        default_dataset_size=default_dataset_size,
        image_extensions=[ext.lower() for ext in image_extensions],
    )
    return _SETTINGS_CACHE

__all__ = ["DatasetEditorSettings", "get_dataset_editor_settings"]
