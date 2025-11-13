"""Dataset Editor utilities for project-local curation.

Provides data access, metadata persistence, tag management, cropping, and
build helpers powering the Dataset Editor UI.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from system.coordinator_settings import SETTINGS, lang
from system.dataset_discovery import get_image_files
from system.log import error, info, warning

PROJECTS_DIR = SETTINGS.get("paths", {}).get("projects_dir", "projects")
IMAGE_EXTENSIONS = SETTINGS.get("dataset", {}).get(
    "image_extensions",
    [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tiff",
        ".webp",
    ],
)
PAGE_SIZE_OPTIONS = [25, 50, 100, 250, 500]
DEFAULT_PAGE_SIZE = 100
MIN_CROP_SIZE = 0.2
MAX_CROP_SIZE = 1.0
STATE_VERSION = 1


@dataclass
class CropBox:
    center_x: float = 0.5
    center_y: float = 0.5
    size: float = 1.0

    def clamp(self) -> "CropBox":
        self.center_x = min(max(self.center_x, 0.0), 1.0)
        self.center_y = min(max(self.center_y, 0.0), 1.0)
        self.size = min(max(self.size, MIN_CROP_SIZE), MAX_CROP_SIZE)
        return self

    def to_dict(self) -> Dict[str, float]:
        return {
            "center_x": round(self.center_x, 6),
            "center_y": round(self.center_y, 6),
            "size": round(self.size, 6),
        }


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_root(project_name: str) -> Path:
    base = _root_dir() / PROJECTS_DIR
    project_path = (base / project_name).resolve()
    try:
        project_path.relative_to(base.resolve())
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(lang("dataset_editor.api.invalid_project")) from exc
    if not project_path.exists():
        raise FileNotFoundError(lang("dataset_editor.api.project_missing", project=project_name))
    return project_path


def _data_source_dir(project_name: str) -> Path:
    project_root = _project_root(project_name)
    data_source = project_root / "data_source"
    if not data_source.is_dir():
        raise FileNotFoundError(lang("dataset_editor.api.data_source_missing", project=project_name))
    return data_source


def _dataset_dir(project_name: str) -> Path:
    project_root = _project_root(project_name)
    dataset_dir = project_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def _state_path(project_name: str) -> Path:
    return _project_root(project_name) / "editor.json"


def _load_state(project_name: str) -> Dict[str, Any]:
    path = _state_path(project_name)
    if not path.exists():
        return {"version": STATE_VERSION, "images": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        warning(f"Editor state corrupted for {project_name}: {exc}")
        return {"version": STATE_VERSION, "images": {}}
    if not isinstance(data, dict):
        return {"version": STATE_VERSION, "images": {}}
    if "images" not in data or not isinstance(data["images"], dict):
        data["images"] = {}
    data["version"] = STATE_VERSION
    return data


def _save_state(project_name: str, state: Dict[str, Any]) -> None:
    path = _state_path(project_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def _default_crop() -> Dict[str, float]:
    return CropBox().to_dict()


def _normalize_crop(raw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return _default_crop()
    crop = CropBox(
        center_x=float(raw.get("center_x", 0.5)),
        center_y=float(raw.get("center_y", 0.5)),
        size=float(raw.get("size", 1.0)),
    )
    return crop.clamp().to_dict()


def _normalize_tags(tags: Optional[Iterable[str]]) -> List[str]:
    normalized: List[str] = []
    if not tags:
        return normalized
    seen = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        cleaned = tag.strip()
        if not cleaned:
            continue
        if len(cleaned) > 64:
            cleaned = cleaned[:64]
        if cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        normalized.append(cleaned)
    return normalized


def _infer_default_tags(rel_path: str) -> List[str]:
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) > 1:
        # Use deepest folder name as default tag
        return [parts[-2]]
    return []


def _relative_path(abs_path: Path, base: Path) -> str:
    return abs_path.relative_to(base).as_posix()


def _image_dimensions(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        width, height = img.size
    return int(width), int(height)


def _ensure_entry(state: Dict[str, Any], rel_path: str, abs_path: Path) -> Tuple[Dict[str, Any], bool]:
    images = state.setdefault("images", {})
    entry = images.setdefault(rel_path, {})
    touched = False
    if "crop" not in entry:
        entry["crop"] = _default_crop()
        touched = True
    else:
        entry["crop"] = _normalize_crop(entry["crop"])
    if "tags" not in entry:
        entry["tags"] = _infer_default_tags(rel_path)
        touched = True
    else:
        entry["tags"] = _normalize_tags(entry["tags"])
    if "dimensions" not in entry or not isinstance(entry["dimensions"], dict):
        width, height = _image_dimensions(abs_path)
        entry["dimensions"] = {"width": width, "height": height}
        touched = True
    else:
        width = int(entry["dimensions"].get("width", 0))
        height = int(entry["dimensions"].get("height", 0))
        if width <= 0 or height <= 0:
            width, height = _image_dimensions(abs_path)
            entry["dimensions"] = {"width": width, "height": height}
            touched = True
    if touched:
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    return entry, touched


def _purge_missing_entries(state: Dict[str, Any], available: Iterable[str]) -> bool:
    available_set = set(available)
    images = state.setdefault("images", {})
    missing = [key for key in list(images.keys()) if key not in available_set]
    for key in missing:
        images.pop(key, None)
    return bool(missing)


def _folder_counts(rel_paths: List[str]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {"": len(rel_paths)}
    for rel_path in rel_paths:
        parts = rel_path.split("/")
        if len(parts) <= 1:
            continue
        folder_path = "/".join(parts[:-1])
        counts[folder_path] = counts.get(folder_path, 0) + 1
    result = [{"path": key, "count": value} for key, value in counts.items()]
    result.sort(key=lambda item: item["path"] or "")
    return result


def _tag_counts(state: Dict[str, Any], rel_paths: Iterable[str]) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for rel_path in rel_paths:
        entry = state.get("images", {}).get(rel_path) or {}
        tags = _normalize_tags(entry.get("tags")) or _infer_default_tags(rel_path)
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    result = [{"tag": tag, "count": count} for tag, count in counts.items()]
    result.sort(key=lambda item: item["tag"].lower())
    return result


def _apply_filters(
    rel_paths: List[str],
    state: Dict[str, Any],
    folder: Optional[str],
    tags: Optional[List[str]],
    search: Optional[str],
) -> List[str]:
    folder_filter = (folder or "").strip().strip("/")
    tag_filter = _normalize_tags(tags)
    query = (search or "").strip().lower()

    filtered: List[str] = []
    for rel_path in rel_paths:
        if folder_filter and not rel_path.startswith(folder_filter + "/") and folder_filter != rel_path:
            continue
        entry = state.get("images", {}).get(rel_path) or {}
        image_tags = _normalize_tags(entry.get("tags")) or _infer_default_tags(rel_path)
        image_tags_lower = [tag.lower() for tag in image_tags]
        if tag_filter and not all(tag.lower() in image_tags_lower for tag in tag_filter):
            continue
        if query and query not in rel_path.lower():
            continue
        filtered.append(rel_path)
    return filtered


def query_editor_images(
    project_name: str,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    folder: Optional[str] = None,
    tags: Optional[List[str]] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    data_source = _data_source_dir(project_name)
    state = _load_state(project_name)

    abs_paths = [Path(p) for p in get_image_files(str(data_source))]
    rel_paths = [_relative_path(path, data_source) for path in abs_paths]
    rel_paths.sort()

    dirty = _purge_missing_entries(state, rel_paths)

    folder_meta = _folder_counts(rel_paths)
    tag_meta = _tag_counts(state, rel_paths)

    if page_size not in PAGE_SIZE_OPTIONS:
        page_size = DEFAULT_PAGE_SIZE

    filtered = _apply_filters(rel_paths, state, folder, tags, search)
    total = len(filtered)
    total_pages = max((total - 1) // page_size + 1, 1)
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    page_items = filtered[start:end]

    images_payload: List[Dict[str, Any]] = []
    for rel_path in page_items:
        abs_path = data_source / rel_path
        if not abs_path.exists():
            continue
        entry, touched = _ensure_entry(state, rel_path, abs_path)
        dirty = dirty or touched
        crop = _normalize_crop(entry.get("crop"))
        tags_list = _normalize_tags(entry.get("tags"))
        dims = entry.get("dimensions", {})
        width = int(dims.get("width", 0))
        height = int(dims.get("height", 0))
        images_payload.append(
            {
                "path": rel_path,
                "name": os.path.basename(rel_path),
                "folder": os.path.dirname(rel_path),
                "tags": tags_list,
                "crop": crop,
                "dimensions": {"width": width, "height": height},
                "size_bytes": abs_path.stat().st_size,
                "modified_at": datetime.fromtimestamp(abs_path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    if dirty:
        _save_state(project_name, state)

    return {
        "status": "success",
        "project": project_name,
        "page": {
            "index": page,
            "size": page_size,
            "total": total,
            "total_pages": total_pages,
            "options": PAGE_SIZE_OPTIONS,
        },
        "filters": {
            "folder": folder or "",
            "tags": _normalize_tags(tags),
            "search": search or "",
        },
        "folders": folder_meta,
        "tags": tag_meta,
        "images": images_payload,
    }


def update_image_crop(project_name: str, rel_path: str, crop_payload: Dict[str, Any]) -> Dict[str, Any]:
    data_source = _data_source_dir(project_name)
    target = data_source / rel_path
    if not target.exists():
        raise FileNotFoundError(lang("dataset_editor.api.image_missing", image=rel_path))

    state = _load_state(project_name)
    entry, _ = _ensure_entry(state, rel_path, target)
    entry["crop"] = _normalize_crop(crop_payload)
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(project_name, state)
    info(lang("dataset_editor.log.crop_saved", project=project_name, image=rel_path))
    return {"status": "success", "crop": entry["crop"]}


def update_image_tags(
    project_name: str,
    rel_paths: List[str],
    tags: List[str],
    mode: str,
) -> Dict[str, Any]:
    if not rel_paths:
        raise ValueError(lang("dataset_editor.api.no_images_selected"))

    normalized_tags = _normalize_tags(tags)
    if mode not in {"add", "replace", "remove"}:
        mode = "add"

    state = _load_state(project_name)
    data_source = _data_source_dir(project_name)

    updated = 0
    for rel_path in rel_paths:
        target = data_source / rel_path
        if not target.exists():
            continue
        entry, _ = _ensure_entry(state, rel_path, target)
        current_tags = _normalize_tags(entry.get("tags"))
        if mode == "replace":
            entry["tags"] = normalized_tags
        elif mode == "remove":
            removal = {tag.lower() for tag in normalized_tags}
            entry["tags"] = [tag for tag in current_tags if tag.lower() not in removal]
        else:  # add
            combined = current_tags + normalized_tags
            entry["tags"] = _normalize_tags(combined)
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated += 1

    _save_state(project_name, state)
    info(lang("dataset_editor.log.tags_updated", project=project_name, count=updated))
    return {"status": "success", "updated": updated}


def delete_images(project_name: str, rel_paths: List[str]) -> Dict[str, Any]:
    if not rel_paths:
        raise ValueError(lang("dataset_editor.api.no_images_selected"))

    data_source = _data_source_dir(project_name)
    dataset_dir = _dataset_dir(project_name)
    state = _load_state(project_name)

    removed = 0
    skipped = []
    dataset_variants = [dataset_dir, dataset_dir / "train", dataset_dir / "val"]

    for rel_path in rel_paths:
        target = data_source / rel_path
        if target.exists():
            try:
                target.unlink()
                removed += 1
            except Exception as exc:  # noqa: BLE001
                error(f"Failed to delete {target}: {exc}")
                skipped.append(rel_path)
                continue
        else:
            skipped.append(rel_path)
            continue

        txt_source = target.with_suffix(".txt")
        if txt_source.exists():
            txt_source.unlink()

        for variant in dataset_variants:
            variant_path = variant / rel_path
            if variant_path.exists():
                variant_path.unlink()
            variant_txt = variant_path.with_suffix(".txt")
            if variant_txt.exists():
                variant_txt.unlink()

        state.get("images", {}).pop(rel_path, None)

    _save_state(project_name, state)
    info(lang("dataset_editor.log.images_deleted", project=project_name, count=removed))
    return {"status": "success", "removed": removed, "skipped": skipped}


def _purge_directory(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def _compute_crop_box(width: int, height: int, crop: Dict[str, float]) -> Tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    min_side = min(width, height)
    side = max(int(min_side * crop.get("size", 1.0)), 1)
    center_x = int(width * crop.get("center_x", 0.5))
    center_y = int(height * crop.get("center_y", 0.5))
    half = side // 2
    left = max(center_x - half, 0)
    top = max(center_y - half, 0)
    if left + side > width:
        left = width - side
    if top + side > height:
        top = height - side
    left = max(left, 0)
    top = max(top, 0)
    return left, top, left + side, top + side


def build_dataset(project_name: str) -> Dict[str, Any]:
    data_source = _data_source_dir(project_name)
    dataset_dir = _dataset_dir(project_name)
    state = _load_state(project_name)

    abs_paths = [Path(p) for p in get_image_files(str(data_source))]
    rel_paths = [_relative_path(path, data_source) for path in abs_paths]

    _purge_directory(dataset_dir)

    processed = 0
    failed: List[str] = []

    for rel_path, abs_path in zip(rel_paths, abs_paths):
        entry, touched = _ensure_entry(state, rel_path, abs_path)
        if touched:
            state["images"][rel_path] = entry
        crop = _normalize_crop(entry.get("crop"))
        tags = _normalize_tags(entry.get("tags")) or _infer_default_tags(rel_path)
        dims = entry.get("dimensions", {})
        width = int(dims.get("width", 0))
        height = int(dims.get("height", 0))
        if width <= 0 or height <= 0:
            width, height = _image_dimensions(abs_path)
            entry["dimensions"] = {"width": width, "height": height}
        try:
            left, top, right, bottom = _compute_crop_box(width, height, crop)
            with Image.open(abs_path) as img:
                cropped = img.convert("RGB").crop((left, top, right, bottom))
                target_path = dataset_dir / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                cropped.save(target_path)
                tag_path = target_path.with_suffix(".txt")
                if tags:
                    with tag_path.open("w", encoding="utf-8") as handle:
                        handle.write(", ".join(tags))
                elif tag_path.exists():
                    tag_path.unlink()
            processed += 1
        except Exception as exc:  # noqa: BLE001
            error(f"Failed to build image {rel_path}: {exc}")
            failed.append(rel_path)

    _save_state(project_name, state)
    info(lang("dataset_editor.log.build_complete", project=project_name, count=processed))
    message = lang("dataset_editor.api.build_success", count=processed, failed=len(failed))
    return {"status": "success", "processed": processed, "failed": failed, "message": message}


def resolve_image_path(project_name: str, rel_path: str) -> Path:
    data_source = _data_source_dir(project_name)
    candidate = (data_source / rel_path).resolve()
    try:
        candidate.relative_to(data_source.resolve())
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(lang("dataset_editor.api.invalid_image_path")) from exc
    if not candidate.exists():
        raise FileNotFoundError(lang("dataset_editor.api.image_missing", image=rel_path))
    return candidate


__all__ = [
    "query_editor_images",
    "update_image_crop",
    "update_image_tags",
    "delete_images",
    "build_dataset",
    "resolve_image_path",
    "PAGE_SIZE_OPTIONS",
    "DEFAULT_PAGE_SIZE",
]
