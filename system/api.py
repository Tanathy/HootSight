from __future__ import annotations

import os
import re
import json
import random
import base64
import shutil
import urllib.parse
from io import BytesIO
from typing import Any, Dict, Optional, List, NoReturn
from pathlib import Path

from fastapi import FastAPI, Body, Query, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

import torch
from PIL import Image
from torchvision import transforms as tv_transforms

from system.log import info, success, error, warning
from system.coordinator_settings import SETTINGS
from system.dataset_discovery import discover_projects, get_project_info, get_image_files, compute_project_stats
from system.common.checkpoint import find_checkpoint
from system.project_db import (
    get_cached_stats,
    save_stats,
    stats_exist,
    clear_stats,
    set_metadata,
    get_metadata,
    set_project_config_value,
    delete_project_config_value,
    delete_metadata,
    get_project_config_overrides,
    flat_to_nested,
    nested_to_flat,
)
from system.training import start_training, stop_training, get_training_status, get_training_status_all
from system.heatmap import generate_heatmap, evaluate_with_heatmap
from system.coordinator_settings import (
    load_language_data,
    switch_language,
    available_languages,
    lang,
    get_current_language,
)
from system.augmentation import DataAugmentationFactory
from system.updates import check_for_updates, apply_updates
from system.dataset_editor_dataset_builder import build_manager as dataset_build_manager
from system.dataset_editor_discovery import discovery_manager as dataset_discovery_manager
from system.dataset_editor_models import (
    AnnotationUpdate,
    BoundsModel,
    BoundsUpdate,
    BuildOptions,
    BuildRequest,
    BuildStatus,
    BulkTagRequest,
    BulkTagResponse,
    DeleteItemsRequest,
    DeleteItemsResponse,
    DiscoveryStatus,
    DuplicateScanResponse,
    FolderCreateRequest,
    FolderDeleteRequest,
    FolderNode,
    FolderOperationResponse,
    FolderRenameRequest,
    ImageItem,
    PaginatedItems,
    ProjectSummary,
    StatsSummary,
    UploadResponse,
)
from system.dataset_editor_project_service import (
    repository as dataset_repository,
    ImageNotFoundError,
    ProjectNotFoundError,
    ProjectIndex,
)
from system.dataset_editor_settings import get_dataset_editor_settings


def _resolve_doc_path(docs_root: Path, requested: str) -> Optional[Path]:
    if not requested:
        return None

    cleaned = requested.strip()
    if not cleaned:
        return None

    normalized = cleaned.replace("\\", "/")
    normalized = normalized.lstrip("/")
    if normalized.lower().startswith("docs/"):
        normalized = normalized[5:]

    if not normalized.lower().endswith(".md"):
        normalized = f"{normalized}.md"

    candidate = (docs_root / normalized).resolve()
    try:
        candidate.relative_to(docs_root.resolve())
    except ValueError:
        return None

    return candidate


def _derive_doc_title(doc_path: Path) -> str:
    fallback = doc_path.stem.replace('_', ' ').replace('-', ' ').title()
    try:
        with doc_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("#"):
                    title = re.sub(r"^#+\\s*", "", stripped).strip()
                    return title or fallback
    except Exception:
        return fallback
    return fallback


def _list_doc_entries(docs_root: Path) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if not docs_root.is_dir():
        return entries

    for path in docs_root.rglob("*.md"):
        if not path.is_file():
            continue
        relative = path.relative_to(docs_root)
        parts = list(relative.parts)
        if any(part.startswith(".") for part in parts):
            continue
        rel_path = relative.as_posix()
        entries.append({
            "path": rel_path,
            "title": _derive_doc_title(path)
        })

    entries.sort(key=lambda item: item["path"].lower())
    return entries


_augmentation_preview_cache: Dict[str, Image.Image] = {}


def _encode_image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _candidate_dataset_dirs(project_name: str, phase: str) -> List[str]:
    """Get candidate directories for preview images, including data_source fallback."""
    resolved = []
    
    # First try dataset folder
    dataset_root = os.path.join("projects", project_name, "dataset")
    if os.path.isdir(dataset_root):
        phase_dirs: List[str] = []
        normalized_phase = (phase or "").lower()
        if normalized_phase == "train":
            phase_dirs.extend(["train", "training"])
        elif normalized_phase == "val":
            phase_dirs.extend(["val", "validation", "eval"])

        for suffix in phase_dirs:
            candidate = os.path.join(dataset_root, suffix)
            if os.path.isdir(candidate):
                resolved.append(candidate)

        resolved.append(dataset_root)
    
    # Fallback to data_source folder if no images found in dataset
    data_source_root = os.path.join("projects", project_name, "data_source")
    if os.path.isdir(data_source_root):
        resolved.append(data_source_root)
    
    return resolved


def _select_preview_image(project_name: str, phase: str) -> Optional[str]:
    for directory in _candidate_dataset_dirs(project_name, phase):
        images = get_image_files(directory)
        if images:
            return random.choice(images)
    return None


def _detect_checkpoint(model_dir: str, best_filename: str, model_filename: str) -> tuple[bool, Optional[str], str]:
    """Wrapper for centralized checkpoint detection."""
    found, path, reason = find_checkpoint(model_dir, best_filename, model_filename)
    if found and path:
        info(f"Checkpoint detected: {path.name} ({reason})")
    return found, str(path) if path else None, reason


def _apply_augmentation_preview(image: Image.Image, transform_configs: List[Dict[str, Any]]) -> Image.Image:
    if not transform_configs:
        return image.copy()

    transforms_to_apply = []
    normalize_transform = None

    for config in transform_configs:
        if not isinstance(config, dict):
            continue
        try:
            transform = DataAugmentationFactory.create_from_config(config)
        except Exception as exc:  # noqa: BLE001 - propagate as runtime error with context
            raise RuntimeError(str(exc)) from exc

        transforms_to_apply.append(transform)
        if isinstance(transform, tv_transforms.Normalize):
            normalize_transform = transform

    if not transforms_to_apply:
        return image.copy()

    result: Any = image.copy()
    for transform in transforms_to_apply:
        result = transform(result)

    if isinstance(result, torch.Tensor):
        tensor = result.detach().clone()
        if normalize_transform is not None:
            mean_tensor = torch.tensor(normalize_transform.mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
            std_tensor = torch.tensor(normalize_transform.std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
            tensor = tensor * std_tensor + mean_tensor
        tensor = torch.clamp(tensor, 0.0, 1.0).cpu()
        to_pil = tv_transforms.ToPILImage()
        return to_pil(tensor)

    if isinstance(result, Image.Image):
        return result.copy()

    raise TypeError(f"Unsupported augmentation output type: {type(result)}")


DATASET_EDITOR_SETTINGS = get_dataset_editor_settings()


def _dataset_editor_allowed_build_sizes() -> set[int]:
    sizes: set[int] = set()
    for values in DATASET_EDITOR_SETTINGS.dataset_size_options.values():
        sizes.update(values)
    return sizes


def _get_dataset_editor_project(project_name: str) -> ProjectIndex:
    try:
        return dataset_repository.get_index(project_name)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"messageKey": "dataset_editor.api.project_not_found", "messageParams": {"project": project_name}}) from exc


def _raise_image_not_found(project_name: str, exc: ImageNotFoundError) -> NoReturn:
    raise HTTPException(status_code=404, detail={"messageKey": "dataset_editor.api.image_not_found", "messageParams": {"project": project_name}}) from exc


def _validate_dataset_editor_page_size(page_size: int) -> None:
    if page_size not in DATASET_EDITOR_SETTINGS.page_sizes:
        allowed = ", ".join(str(size) for size in DATASET_EDITOR_SETTINGS.page_sizes)
        raise HTTPException(status_code=400, detail={"messageKey": "dataset_editor.api.invalid_page_size", "messageParams": {"allowed": allowed}})


def _validate_dataset_editor_build_size(size: int) -> None:
    allowed = _dataset_editor_allowed_build_sizes()
    if size not in allowed:
        allowed_label = ", ".join(str(value) for value in sorted(allowed))
        raise HTTPException(status_code=400, detail={"messageKey": "dataset_editor.api.invalid_size", "messageParams": {"allowed": allowed_label}})


def create_app() -> FastAPI:
    app = FastAPI(
        title="Hootsight API",
        version="1.0.0",
        description="Minimal API for config-driven ResNet training"
    )

    try:
        paths_cfg = SETTINGS['paths']
    except Exception:
        raise ValueError("Missing required 'paths' section in config/config.json")
    if 'ui_dir' not in paths_cfg:
        raise ValueError("paths.ui_dir must be defined in config/config.json")
    ui_dir = paths_cfg['ui_dir']
    if os.path.isdir(ui_dir):
        app.mount("/static", StaticFiles(directory=ui_dir), name="static")

    root_path = Path(__file__).resolve().parents[1]
    if 'docs_dir' not in paths_cfg:
        raise ValueError("paths.docs_dir must be defined in config/config.json")
    docs_dir_config = paths_cfg['docs_dir']
    docs_root = (root_path / docs_dir_config).resolve()
    if docs_root.is_dir():
        app.mount("/docs/assets", StaticFiles(directory=str(docs_root)), name="docs_assets")
    else:
        docs_root = None

    @app.get("/")
    async def serve_ui_root():
        ui_file = os.path.join(ui_dir, "index.html")
        if os.path.isfile(ui_file):
            return FileResponse(ui_file)
        return {"status": "ok", "message": "Hootsight API running"}

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        return {"status": "healthy"}

    @app.get("/system/stats")
    def get_system_stats_endpoint() -> Dict[str, Any]:
        """Get current system resource usage for monitoring."""
        from system.sensors import get_system_stats
        return get_system_stats()

    @app.get("/config")
    def get_config() -> Dict[str, Any]:
        return {"config": SETTINGS}

    @app.get("/config/schema")
    def get_config_schema() -> Dict[str, Any]:
        try:
            schema_path = os.path.join("config", "config_schema.json")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            # Return raw schema - frontend handles localization via lang keys
            return {"schema": schema}
        except Exception as ex:
            error(f"Schema loading failed: {ex}")
            return {"error": str(ex)}

    @app.get("/localization")
    def get_localization() -> Dict[str, Any]:
        try:
            data = load_language_data()
            return {
                "localization": data,
                "languages": available_languages(),
                "active": get_current_language(),
            }
        except Exception as ex:
            error(f"Localization load failed: {ex}")
            return {"error": str(ex)}

    @app.post("/localization/switch")
    def localization_switch(lang_code: str = Body(..., embed=True)) -> Dict[str, Any]:
        try:
            data = switch_language(lang_code)
            return {"localization": data, "active": lang_code}
        except Exception as ex:
            error(f"Localization switch failed: {ex}")
            return {"error": str(ex)}

    @app.get("/presets")
    def get_presets() -> Dict[str, Any]:
        try:
            presets_dir = root_path / "config" / "presets"
            if not presets_dir.exists():
                return {"presets": []}

            def _infer_preset_categories(preset_data: Dict[str, Any]) -> List[str]:
                categories: set[str] = set()

                task = preset_data.get("task")
                if isinstance(task, str) and task:
                    categories.add(task)

                dataset_types = preset_data.get("dataset_types", []) or []
                if "multi_label" in dataset_types:
                    categories.add("multi_label")
                if "folder_classification" in dataset_types or not dataset_types:
                    categories.add("classification")

                configs = preset_data.get("configs", {}) or {}
                for size_data in configs.values():
                    training_cfg = (size_data or {}).get("config", {}).get("training", {})
                    task_value = training_cfg.get("task")
                    if isinstance(task_value, str) and task_value:
                        categories.add(task_value)

                return sorted(categories) if categories else ["classification"]

            presets = []
            for preset_file in presets_dir.glob("*.json"):
                try:
                    with preset_file.open("r", encoding="utf-8") as f:
                        preset_data = json.load(f)
                        
                        # Extract size variants info from configs
                        size_variants = []
                        configs = preset_data.get("configs", {})
                        for size_key, size_data in configs.items():
                            size_variants.append({
                                "key": size_key,
                                "description": size_data.get("description", ""),
                                "range": size_data.get("range", [0, None])
                            })
                        
                        presets.append({
                            "id": preset_file.stem,
                            "name": preset_data.get("name", preset_file.stem),
                            "description": preset_data.get("description", ""),
                            "task": preset_data.get("task", "unknown"),
                            "recommended_models": preset_data.get("recommended_models", []),
                            "dataset_types": preset_data.get("dataset_types", []),
                            "size_variants": size_variants,
                            "categories": preset_data.get("categories", _infer_preset_categories(preset_data))
                        })
                except Exception as ex:
                    error(f"Failed to load preset {preset_file.name}: {ex}")
                    continue

            return {"presets": presets}
        except Exception as ex:
            error(f"Failed to load presets: {ex}")
            return {"error": str(ex)}

    @app.get("/presets/{preset_id}")
    def get_preset(preset_id: str) -> Dict[str, Any]:
        try:
            presets_dir = root_path / "config" / "presets"
            preset_file = presets_dir / f"{preset_id}.json"

            if not preset_file.exists():
                return {"status": "error", "message": f"Preset '{preset_id}' not found"}

            with preset_file.open("r", encoding="utf-8") as f:
                preset_data = json.load(f)

            return {"status": "success", "preset": preset_data}
        except Exception as ex:
            error(f"Failed to load preset '{preset_id}': {ex}")
            return {"status": "error", "message": str(ex)}

    @app.get("/docs/list")
    def docs_list() -> Dict[str, Any]:
        if docs_root is None:
            return {"docs": []}
        return {"docs": _list_doc_entries(docs_root)}

    @app.get("/docs/page")
    def docs_page(path: str = Query(..., min_length=1)) -> Dict[str, Any]:
        if docs_root is None:
            warning("Documentation directory is not configured; cannot serve documentation page")
            return {"status": "error", "messageKey": "docs.api.missing_root"}

        target = _resolve_doc_path(docs_root, path)
        if target is None or not target.is_file():
            warning(f"Documentation page '{path}' not found")
            return {"status": "error", "messageKey": "docs.api.not_found", "messageParams": {"path": path}}

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            error(f"Documentation decode failed for {target}: {exc}")
            return {"status": "error", "messageKey": "docs.api.decode_error", "messageParams": {"path": target.name}}
        except Exception as exc:
            error(f"Documentation read failed for {target}: {exc}")
            return {"status": "error", "messageKey": "docs.api.read_failed", "messageParams": {"error": str(exc)}}

        rel_path = target.relative_to(docs_root).as_posix()
        title = _derive_doc_title(target)
        return {"status": "success", "path": rel_path, "title": title, "content": content}

    @app.get("/projects/{project_name}/config")
    def get_project_config(project_name: str) -> Dict[str, Any]:
        """
        Get project-specific config overrides (nested structure).
        Returns ONLY the overrides stored in metadata, not merged with defaults.
        UI does its own deep merge with system defaults.
        """
        try:
            overrides_flat = get_project_config_overrides(project_name)
            overrides_nested = flat_to_nested(overrides_flat)
            return {"config": overrides_nested}
        except Exception as ex:
            error(f"Project config load failed: {ex}")
            return {"error": str(ex)}

    @app.post("/projects/{project_name}/config")
    def update_project_config(project_name: str, data: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """
        Save project config overrides.
        Accepts nested config, compares with defaults, flattens and saves only differences.
        """
        try:
            # Flatten incoming nested config
            incoming_flat = nested_to_flat(data)
            
            # Flatten default config for comparison
            default_flat = nested_to_flat(SETTINGS)
            
            # Get current overrides to know what to delete
            current_overrides = get_project_config_overrides(project_name)
            
            saved_count = 0
            deleted_count = 0
            
            # Find values that differ from defaults and save them
            for key, value in incoming_flat.items():
                default_value = default_flat.get(key)
                
                if value != default_value:
                    # Value differs from default - save it
                    set_project_config_value(project_name, key, value)
                    saved_count += 1
                elif key in current_overrides:
                    # Value equals default but was previously overridden - delete it
                    delete_metadata(project_name, key)
                    deleted_count += 1
            
            success(f"Project {project_name} config updated: {saved_count} saved, {deleted_count} reset to default")
            return {"status": "success", "saved": saved_count, "deleted": deleted_count}
        except Exception as ex:
            error(f"Project config update failed: {ex}")
            return {"status": "error", "message": str(ex)}

    @app.post("/projects/{project_name}/config/set")
    def set_single_config_value(project_name: str, data: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """
        Set a single project config value override.
        """
        try:
            key = data.get('key')
            value = data.get('value')
            
            if not key:
                return {"status": "error", "message": "Missing 'key' parameter"}
            
            set_project_config_value(project_name, key, value)
            info(f"Project {project_name} config set: {key} = {value}")
            return {"status": "success", "key": key, "value": value}
        except Exception as ex:
            error(f"Failed to set project config value: {ex}")
            return {"status": "error", "message": str(ex)}

    @app.get("/projects")
    def list_projects() -> Dict[str, Any]:
        """
        List all projects with basic info.
        Statistics are NOT computed here - only cached stats from DB are included.
        Use /projects/{name}/stats to compute and cache statistics.
        """
        try:
            # Lightweight discovery - no stats computation
            projects = discover_projects(compute_stats=False)
            project_dicts = []

            # Determine default checkpoint filenames once (project overrides can replace these per row)
            try:
                ckpt_cfg = SETTINGS['training']['checkpoint']
                default_ckpt_best = ckpt_cfg.get('best_model_filename', 'best_model.pth')
                default_ckpt_model = ckpt_cfg.get('model_filename', 'model.pth')
            except Exception:
                default_ckpt_best = 'best_model.pth'
                default_ckpt_model = 'model.pth'

            for project in projects:
                # Get user-set dataset_type from DB (not auto-detected)
                user_dataset_type = get_metadata(project.name, "dataset_type", "unknown")

                ckpt_best = get_metadata(project.name, "training.checkpoint.best_model_filename", default_ckpt_best) or default_ckpt_best
                ckpt_model = get_metadata(project.name, "training.checkpoint.model_filename", default_ckpt_model) or default_ckpt_model
                has_model, ckpt_path, ckpt_reason = _detect_checkpoint(project.model_path, ckpt_best, ckpt_model)

                data = {
                    "name": project.name,
                    "path": project.path,
                    "dataset_path": project.dataset_path,
                    "model_path": project.model_path,
                    "dataset_type": user_dataset_type,
                    "image_count": project.image_count,
                    "has_dataset": project.has_dataset,
                    "has_model_dir": project.has_model_dir,
                    "has_model": has_model,
                    "checkpoint_path": ckpt_path if has_model else None,
                    "checkpoint_reason": ckpt_reason,
                }
                
                # Include cached stats if available (from DB)
                cached = get_cached_stats(project.name)
                if cached:
                    data["balance_score"] = cached.get("balance_score", 0.0)
                    data["balance_status"] = cached.get("balance_status", "")
                    data["num_labels"] = cached.get("num_labels", 0)
                    data["stats_computed_at"] = cached.get("computed_at")
                    # Include lightweight balance_analysis (without heavy lists)
                    if "balance_analysis" in cached:
                        data["balance_analysis"] = cached["balance_analysis"]
                else:
                    data["stats_computed_at"] = None
                
                project_dicts.append(data)
            return {"projects": project_dicts}
        except Exception as ex:
            error(f"Project discovery failed: {ex}")
            return {"error": str(ex)}

    @app.post("/projects/create")
    def create_project(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        min_length = 3
        max_length = 64
        try:
            paths_cfg = SETTINGS['paths']
        except Exception:
            raise ValueError("Missing required 'paths' section in config/config.json")
        projects_dir = paths_cfg.get("projects_dir")
        if projects_dir is None:
            raise ValueError("paths.projects_dir must be defined in config/config.json")
        root_path = Path(__file__).resolve().parents[1]
        projects_root = (root_path / projects_dir).resolve()
        projects_root.mkdir(parents=True, exist_ok=True)

        name_value = ""
        if isinstance(payload, dict):
            name_candidate = payload.get("name")
            if isinstance(name_candidate, str):
                name_value = name_candidate.strip()
            elif name_candidate is not None:
                name_value = str(name_candidate).strip()

        if not name_value:
            return {"status": "error", "messageKey": "projects.api.create_missing"}

        if len(name_value) < min_length or len(name_value) > max_length:
            return {"status": "error", "messageKey": "projects.api.create_length", "messageParams": {"min": min_length, "max": max_length}}

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name_value):
            return {"status": "error", "messageKey": "projects.api.create_invalid"}

        project_path = (projects_root / name_value).resolve()
        try:
            project_path.relative_to(projects_root)
        except ValueError:
            return {"status": "error", "messageKey": "projects.api.create_invalid"}

        if project_path.exists():
            return {"status": "error", "messageKey": "projects.api.create_exists", "messageParams": {"name": name_value}}

        try:
            project_path.mkdir(parents=False, exist_ok=False)
            for subdir in ("dataset", "data_source", "model", "heatmaps", "validation"):
                (project_path / subdir).mkdir(parents=True, exist_ok=True)

            success(f"Project created: {name_value}")
        except Exception as ex:  # noqa: BLE001 - ensure structured error response
            error(f"Project creation failed for {name_value}: {ex}")
            return {"status": "error", "messageKey": "projects.api.create_error", "messageParams": {"error": str(ex)}}

        project_details = None
        try:
            for project in discover_projects(compute_stats=False):
                if project.name == name_value:
                    project_details = {
                        "name": project.name,
                        "path": project.path,
                        "dataset_path": project.dataset_path,
                        "model_path": project.model_path,
                        "has_dataset": project.has_dataset,
                        "has_model_dir": project.has_model_dir,
                    }
                    break
        except Exception as ex:  # noqa: BLE001 - project list retrieval should not fail the request
            warning(f"Post-create project discovery failed for {name_value}: {ex}")

        response: Dict[str, Any] = {
            "status": "success",
            "messageKey": "projects.api.create_success",
            "messageParams": {"name": name_value}
        }
        if project_details is not None:
            response["project"] = project_details
        else:
            response["project"] = {"name": name_value, "path": str(project_path)}

        return response

    @app.delete("/projects/{project_name}")
    def delete_project(project_name: str) -> Dict[str, Any]:
        """
        Delete a project and all its contents.
        This is a destructive operation - cannot be undone.
        """
        try:
            paths_cfg = SETTINGS['paths']
        except Exception:
            raise ValueError("Missing required 'paths' section in config/config.json")
        projects_dir = paths_cfg.get("projects_dir")
        if projects_dir is None:
            raise ValueError("paths.projects_dir must be defined in config/config.json")
        root_path = Path(__file__).resolve().parents[1]
        projects_root = (root_path / projects_dir).resolve()

        # Validate project name
        if not project_name or not project_name.strip():
            return {"status": "error", "messageKey": "projects.api.delete_invalid"}

        project_path = (projects_root / project_name).resolve()
        
        # Security check: ensure path is within projects directory
        try:
            project_path.relative_to(projects_root)
        except ValueError:
            return {"status": "error", "messageKey": "projects.api.delete_invalid"}

        if not project_path.exists():
            return {"status": "error", "messageKey": "projects.api.delete_not_found", "messageParams": {"name": project_name}}

        if not project_path.is_dir():
            return {"status": "error", "messageKey": "projects.api.delete_not_found", "messageParams": {"name": project_name}}

        try:
            # Remove the entire project directory
            shutil.rmtree(project_path)
            success(f"Project deleted: {project_name}")
            return {
                "status": "success",
                "messageKey": "projects.api.delete_success",
                "messageParams": {"name": project_name}
            }
        except Exception as ex:
            error(f"Project deletion failed for {project_name}: {ex}")
            return {"status": "error", "messageKey": "projects.api.delete_error", "messageParams": {"error": str(ex)}}

    @app.post("/projects/{project_name}/rename")
    def rename_project(project_name: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        """Rename a project directory."""
        min_length = 3
        max_length = 64
        try:
            paths_cfg = SETTINGS['paths']
        except Exception:
            raise ValueError("Missing required 'paths' section in config/config.json")
        projects_dir = paths_cfg.get("projects_dir")
        if projects_dir is None:
            raise ValueError("paths.projects_dir must be defined in config/config.json")
        root_path = Path(__file__).resolve().parents[1]
        projects_root = (root_path / projects_dir).resolve()

        # Validate current project exists
        old_path = (projects_root / project_name).resolve()
        try:
            old_path.relative_to(projects_root)
        except ValueError:
            return {"status": "error", "messageKey": "projects.api.rename_invalid"}

        if not old_path.exists() or not old_path.is_dir():
            return {"status": "error", "messageKey": "projects.api.rename_not_found", "messageParams": {"name": project_name}}

        # Get and validate new name
        new_name = ""
        if isinstance(payload, dict):
            name_candidate = payload.get("new_name")
            if isinstance(name_candidate, str):
                new_name = name_candidate.strip()
            elif name_candidate is not None:
                new_name = str(name_candidate).strip()

        if not new_name:
            return {"status": "error", "messageKey": "projects.api.rename_missing"}

        if len(new_name) < min_length or len(new_name) > max_length:
            return {"status": "error", "messageKey": "projects.api.create_length", "messageParams": {"min": min_length, "max": max_length}}

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", new_name):
            return {"status": "error", "messageKey": "projects.api.create_invalid"}

        new_path = (projects_root / new_name).resolve()
        try:
            new_path.relative_to(projects_root)
        except ValueError:
            return {"status": "error", "messageKey": "projects.api.rename_invalid"}

        if new_path.exists():
            return {"status": "error", "messageKey": "projects.api.create_exists", "messageParams": {"name": new_name}}

        try:
            old_path.rename(new_path)
            success(f"Project renamed: {project_name} -> {new_name}")
            return {
                "status": "success",
                "messageKey": "projects.api.rename_success",
                "messageParams": {"old": project_name, "new": new_name},
                "new_name": new_name
            }
        except Exception as ex:
            error(f"Project rename failed for {project_name}: {ex}")
            return {"status": "error", "messageKey": "projects.api.rename_error", "messageParams": {"error": str(ex)}}

    @app.get("/projects/{project_name}")
    def project_details(project_name: str) -> Dict[str, Any]:
        """
        Get project details without computing statistics.
        Cached stats from DB are included if available.
        Use /projects/{name}/stats to compute fresh statistics.
        """
        try:
            # Lightweight - no stats computation
            info_obj = get_project_info(project_name, compute_stats=False)
            if info_obj is None:
                return {"project": None}
            
            # Get user-set dataset_type from DB
            user_dataset_type = get_metadata(project_name, "dataset_type", "unknown")

            try:
                ckpt_cfg = SETTINGS['training']['checkpoint']
                default_ckpt_best = ckpt_cfg.get('best_model_filename', 'best_model.pth')
                default_ckpt_model = ckpt_cfg.get('model_filename', 'model.pth')
            except Exception:
                default_ckpt_best = 'best_model.pth'
                default_ckpt_model = 'model.pth'

            ckpt_best = get_metadata(project_name, "training.checkpoint.best_model_filename", default_ckpt_best) or default_ckpt_best
            ckpt_model = get_metadata(project_name, "training.checkpoint.model_filename", default_ckpt_model) or default_ckpt_model
            has_model, ckpt_path, ckpt_reason = _detect_checkpoint(info_obj.model_path, ckpt_best, ckpt_model)
            
            data = {
                "name": info_obj.name,
                "path": info_obj.path,
                "dataset_path": info_obj.dataset_path,
                "model_path": info_obj.model_path,
                "dataset_type": user_dataset_type,
                "image_count": info_obj.image_count,
                "has_dataset": info_obj.has_dataset,
                "has_model_dir": info_obj.has_model_dir,
                "has_model": has_model,
                "checkpoint_path": ckpt_path if has_model else None,
                "checkpoint_reason": ckpt_reason,
            }
            
            # Include cached stats if available
            cached = get_cached_stats(project_name)
            if cached:
                data["balance_score"] = cached.get("balance_score", 0.0)
                data["balance_status"] = cached.get("balance_status", "")
                data["num_labels"] = cached.get("num_labels", 0)
                data["stats_computed_at"] = cached.get("computed_at")
                if "balance_analysis" in cached:
                    data["balance_analysis"] = cached["balance_analysis"]
            else:
                data["stats_computed_at"] = None
            
            return {"project": data}
        except Exception as ex:
            error(f"Project info failed for {project_name}: {ex}")
            return {"error": str(ex)}

    @app.post("/projects/{project_name}/stats")
    def compute_project_statistics(project_name: str) -> Dict[str, Any]:
        """
        Compute and cache full statistics for a project.
        This is the expensive operation - use sparingly.
        Results are saved to project.db for future quick access.
        """
        try:
            # Full stats computation
            stats = compute_project_stats(project_name)
            if stats is None:
                return {"status": "error", "message": f"Project '{project_name}' not found"}
            
            # Save to database
            save_stats(project_name, stats)
            
            # Get user-set dataset_type from DB
            user_dataset_type = get_metadata(project_name, "dataset_type", "unknown")
            
            # Get balance_analysis which contains total_images, num_labels, min_images, max_images
            balance_analysis = stats.get("balance_analysis", {})
            
            # Return stats - pull values from balance_analysis (already computed there)
            result = {
                "status": "success",
                "project": project_name,
                "computed_at": stats.get("computed_at"),
                "dataset_type": user_dataset_type,
                "total_images": balance_analysis.get("total_images", stats.get("image_count", 0)),
                "num_labels": balance_analysis.get("num_labels", len(stats.get("labels", []))),
                "min_images": balance_analysis.get("min_images", 0),
                "max_images": balance_analysis.get("max_images", 0),
                "balance_score": stats.get("balance_score", 0.0),
                "balance_status": stats.get("balance_status", ""),
                "has_dataset": stats.get("has_dataset", False),
                "has_model_dir": stats.get("has_model_dir", False),
            }
            
            info(f"Computed and cached statistics for project {project_name}")
            return result
        except Exception as ex:
            error(f"Statistics computation failed for {project_name}: {ex}")
            return {"status": "error", "message": str(ex)}

    @app.get("/dataset/editor/projects", response_model=list[ProjectSummary])
    def dataset_editor_list_projects() -> list[ProjectSummary]:
        return dataset_repository.list_projects()

    @app.get("/dataset/editor/projects/{project_name}/items", response_model=PaginatedItems)
    def dataset_editor_list_items(
        project_name: str,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=DATASET_EDITOR_SETTINGS.default_page_size),
        search: str | None = Query(default=None),
        category: str | None = Query(default=None),
        folder: str | None = Query(default=None, description="Relative folder filter"),
    ) -> PaginatedItems:
        _validate_dataset_editor_page_size(page_size)
        project = _get_dataset_editor_project(project_name)
        return project.list_items(page=page, page_size=page_size, search=search, category=category, folder=folder)

    @app.post("/dataset/editor/projects/{project_name}/items/{item_path:path}/annotation", response_model=ImageItem)
    def dataset_editor_save_annotation(project_name: str, item_path: str, payload: AnnotationUpdate = Body(...)) -> ImageItem:
        project = _get_dataset_editor_project(project_name)
        try:
            return project.update_annotation(item_path, payload.content)
        except ImageNotFoundError as exc:
            _raise_image_not_found(project_name, exc)

    @app.post("/dataset/editor/projects/{project_name}/items/{item_path:path}/bounds", response_model=ImageItem)
    def dataset_editor_save_bounds(project_name: str, item_path: str, payload: BoundsUpdate = Body(...)) -> ImageItem:
        project = _get_dataset_editor_project(project_name)
        try:
            bounds = BoundsModel(**payload.model_dump())
            return project.update_bounds(item_path, bounds)
        except ImageNotFoundError as exc:
            _raise_image_not_found(project_name, exc)

    @app.post("/dataset/editor/projects/{project_name}/items/delete", response_model=DeleteItemsResponse)
    def dataset_editor_delete_items(project_name: str, payload: DeleteItemsRequest = Body(...)) -> DeleteItemsResponse:
        project = _get_dataset_editor_project(project_name)
        return project.delete_items(payload.items)

    @app.get("/dataset/editor/projects/{project_name}/stats", response_model=StatsSummary)
    def dataset_editor_project_stats(project_name: str) -> StatsSummary:
        project = _get_dataset_editor_project(project_name)
        return project.stats()

    @app.post("/dataset/editor/projects/{project_name}/stats/refresh")
    def dataset_editor_refresh_stats(project_name: str) -> Dict[str, Any]:
        """Recompute and save project statistics to project.db, return updated stats"""
        project = _get_dataset_editor_project(project_name)
        # Compute stats once and persist; also return live values to UI
        stats_model = project.stats()
        persisted = project.save_computed_stats()

        result: Dict[str, Any] = {
            "status": "success" if persisted else "error",
            "project": project_name,
            "image_count": stats_model.total_images,
            "balance_score": stats_model.balance_score,
            "balance_status": stats_model.balance_status,
        }

        # Fallback to cached values if save succeeded but stats_model unexpectedly missing
        if persisted and stats_model.total_images is None:
            cached = get_cached_stats(project_name) or {}
            result["image_count"] = cached.get("image_count", 0)
            result["balance_score"] = cached.get("balance_score", 0.0)
            result["balance_status"] = cached.get("balance_status", "")

        return result

    @app.post("/dataset/editor/projects/{project_name}/dataset-type")
    def dataset_editor_set_dataset_type(project_name: str, payload: Dict[str, str] = Body(...)) -> Dict[str, Any]:
        """Save user-selected dataset type to project.db"""
        dataset_type = payload.get("type", "unknown")
        # Don't save "auto" - that's not a real type
        if dataset_type == "auto":
            dataset_type = "unknown"
        success = set_metadata(project_name, "dataset_type", dataset_type)
        return {"status": "success" if success else "error", "type": dataset_type}

    @app.get("/dataset/editor/projects/{project_name}/dataset-type")
    def dataset_editor_get_dataset_type(project_name: str) -> Dict[str, Any]:
        """Get user-selected dataset type from project.db"""
        dataset_type = get_metadata(project_name, "dataset_type", "unknown")
        return {"type": dataset_type}

    @app.get("/dataset/editor/projects/{project_name}/folders", response_model=FolderNode)
    def dataset_editor_project_folders(project_name: str) -> FolderNode:
        project = _get_dataset_editor_project(project_name)
        return project.folder_tree()

    @app.post("/dataset/editor/projects/{project_name}/folders", response_model=FolderOperationResponse)
    def dataset_editor_create_folder(project_name: str, payload: FolderCreateRequest = Body(...)) -> FolderOperationResponse:
        project = _get_dataset_editor_project(project_name)
        try:
            created_path = project.create_folder(payload.path)
            relative = str(created_path.relative_to(project.data_source_dir)).replace("\\", "/")
            return FolderOperationResponse(success=True, path=relative, message="Folder created")
        except ValueError as e:
            return FolderOperationResponse(success=False, path=payload.path, message=str(e))

    @app.post("/dataset/editor/projects/{project_name}/folders/rename", response_model=FolderOperationResponse)
    def dataset_editor_rename_folder(project_name: str, payload: FolderRenameRequest = Body(...)) -> FolderOperationResponse:
        project = _get_dataset_editor_project(project_name)
        try:
            new_path = project.rename_folder(payload.old_path, payload.new_name)
            relative = str(new_path.relative_to(project.data_source_dir)).replace("\\", "/")
            return FolderOperationResponse(success=True, path=relative, message="Folder renamed")
        except ValueError as e:
            return FolderOperationResponse(success=False, path=payload.old_path, message=str(e))

    @app.post("/dataset/editor/projects/{project_name}/folders/delete", response_model=FolderOperationResponse)
    def dataset_editor_delete_folder(project_name: str, payload: FolderDeleteRequest = Body(...)) -> FolderOperationResponse:
        project = _get_dataset_editor_project(project_name)
        try:
            deleted_count = project.delete_folder(payload.path, payload.recursive)
            return FolderOperationResponse(success=True, path=payload.path, message=f"Folder deleted ({deleted_count} images removed)")
        except ValueError as e:
            return FolderOperationResponse(success=False, path=payload.path, message=str(e))

    @app.post("/dataset/editor/projects/{project_name}/upload", response_model=UploadResponse)
    async def dataset_editor_upload_images(
        project_name: str,
        folder: str = "",
        files: List[UploadFile] = File(...),
    ) -> UploadResponse:
        project = _get_dataset_editor_project(project_name)
        file_data = []
        for upload_file in files:
            content = await upload_file.read()
            file_data.append((upload_file.filename, content))
        result = project.upload_images(file_data, folder)
        return UploadResponse(
            uploaded=result["uploaded"],
            failed=result["failed"],
            files=result["files"],
            errors=result["errors"],
        )

    @app.post("/dataset/editor/projects/{project_name}/refresh", response_model=DiscoveryStatus)
    def dataset_editor_refresh(project_name: str) -> DiscoveryStatus:
        project = _get_dataset_editor_project(project_name)
        job = dataset_discovery_manager.start(project)
        return job.to_model()

    @app.get("/dataset/editor/projects/{project_name}/refresh/status", response_model=DiscoveryStatus)
    def dataset_editor_refresh_status(project_name: str) -> DiscoveryStatus:
        project = _get_dataset_editor_project(project_name)
        job = dataset_discovery_manager.get(project.name)
        return job.to_model()

    @app.post("/dataset/editor/projects/{project_name}/build", response_model=BuildStatus)
    def dataset_editor_build_project(
        project_name: str,
        payload: BuildRequest | None = Body(default=None),
    ) -> BuildStatus:
        project = _get_dataset_editor_project(project_name)
        size = payload.size if payload and payload.size else DATASET_EDITOR_SETTINGS.default_dataset_size
        _validate_dataset_editor_build_size(size)
        job = dataset_build_manager.start_build(project, size)
        return job.to_model()

    @app.get("/dataset/editor/projects/{project_name}/build/status", response_model=BuildStatus)
    def dataset_editor_build_status(project_name: str) -> BuildStatus:
        project = _get_dataset_editor_project(project_name)
        job = dataset_build_manager.get_status(project.name)
        return job.to_model()

    @app.get("/dataset/editor/projects/{project_name}/build/options", response_model=BuildOptions)
    def dataset_editor_build_options(project_name: str) -> BuildOptions:
        project = _get_dataset_editor_project(project_name)
        job = dataset_build_manager.get_status(project.name)
        return BuildOptions(
            presets=DATASET_EDITOR_SETTINGS.dataset_size_options,
            default_size=DATASET_EDITOR_SETTINGS.default_dataset_size,
            active_size=job.target_size,
        )

    @app.post("/dataset/editor/projects/{project_name}/bulk/tags", response_model=BulkTagResponse)
    def dataset_editor_bulk_tags(project_name: str, payload: BulkTagRequest = Body(...)) -> BulkTagResponse:
        project = _get_dataset_editor_project(project_name)
        return project.bulk_tags(payload)

    @app.post("/dataset/editor/projects/{project_name}/duplicates", response_model=DuplicateScanResponse)
    def dataset_editor_scan_duplicates(project_name: str) -> DuplicateScanResponse:
        project = _get_dataset_editor_project(project_name)
        return project.scan_duplicates()

    @app.get("/dataset/editor/projects/{project_name}/image")
    def dataset_editor_image(project_name: str, path: str = Query(..., min_length=1)) -> FileResponse:
        project = _get_dataset_editor_project(project_name)
        decoded = Path(urllib.parse.unquote(path))
        if decoded.is_absolute():
            raise HTTPException(status_code=400, detail={"messageKey": "dataset_editor.api.invalid_image_path"})
        target = (project.data_source_dir / decoded).resolve()
        data_source_dir = project.data_source_dir.resolve()
        if data_source_dir not in target.parents and target != data_source_dir:
            raise HTTPException(status_code=400, detail={"messageKey": "dataset_editor.api.invalid_image_path"})
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail={"messageKey": "dataset_editor.api.image_missing"})
        return FileResponse(target)

    @app.post("/training/start")
    def training_start(
        project_name: str = Body(..., embed=True),
        model_type: str = Body("resnet", embed=True),
        model_name: str = Body("resnet50", embed=True),
        epochs: Optional[int] = Body(None, embed=True),
        resume: bool = Body(False, embed=True)
    ) -> Dict[str, Any]:
        try:
            return start_training(project_name=project_name, model_type=model_type, model_name=model_name, epochs=epochs, resume=resume)
        except Exception as ex:
            error(f"Training start failed: {ex}")
            return {"started": False, "error": str(ex)}

    @app.post("/training/stop")
    def training_stop(training_id: str = Body(..., embed=True)) -> Dict[str, Any]:
        try:
            return stop_training(training_id)
        except Exception as ex:
            error(f"Training stop failed: {ex}")
            return {"stopped": False, "error": str(ex)}

    @app.get("/training/status")
    def training_status(training_id: Optional[str] = Query(None)) -> Dict[str, Any]:
        try:
            from system.sensors import get_system_stats
            status = get_training_status(training_id)
            # Include system stats in the response
            status['system_stats'] = get_system_stats()
            return status
        except Exception as ex:
            error(f"Training status failed: {ex}")
            return {"error": str(ex)}

    @app.get("/training/status/all")
    def training_status_all(training_id: Optional[str] = Query(None)) -> Dict[str, Any]:
        try:
            return get_training_status_all(training_id)
        except Exception as ex:
            error(f"Training status history failed: {ex}")
            return {"error": str(ex)}

    @app.get("/system/updates/check")
    def system_updates_check() -> Dict[str, Any]:
        try:
            return check_for_updates()
        except Exception as ex:  # noqa: BLE001 - ensure JSON response
            error(f"System update check failed: {ex}")
            return {"status": "error", "messageKey": "updates.api.check_failed", "messageParams": {"error": str(ex)}}

    @app.post("/system/updates/apply")
    def system_updates_apply(payload: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
        selected_paths = None
        if payload and isinstance(payload, dict):
            raw_paths = payload.get("paths")
            if isinstance(raw_paths, list):
                selected_paths = [str(item) for item in raw_paths if isinstance(item, (str, bytes))]
        try:
            return apply_updates(selected_paths)
        except Exception as ex:  # noqa: BLE001 - ensure JSON response
            error(f"System update apply failed: {ex}")
            return {"status": "error", "messageKey": "updates.api.apply_failed", "messageParams": {"error": str(ex)}}

    @app.get("/projects/{project_name}/heatmap")
    def project_heatmap(
        project_name: str,
        image_path: Optional[str] = Query(None),
        class_index: Optional[int] = Query(None),
        alpha: float = Query(0.5),
        multi_label: bool = Query(False)
    ):
        try:
            data, meta = generate_heatmap(project_name, image_path=image_path, target_class=class_index, alpha=alpha, multi_label=multi_label)
            return Response(content=data, media_type="image/png")
        except Exception as ex:
            error(f"Heatmap generation failed for {project_name}: {ex}")
            return {"error": str(ex)}

    @app.get("/projects/{project_name}/evaluate")
    def project_evaluate(
        project_name: str,
        image_path: Optional[str] = Query(None),
        checkpoint_path: Optional[str] = Query(None),
        use_live_model: bool = Query(False),
        multi_label: bool = Query(False)
    ):
        try:
            result = evaluate_with_heatmap(
                project_name, 
                image_path=image_path, 
                checkpoint_path=checkpoint_path,
                use_live_model=use_live_model,
                multi_label=multi_label
            )
            return result
        except Exception as ex:
            error(f"Evaluation failed for {project_name}: {ex}")
            return {"error": str(ex)}

    @app.post("/projects/{project_name}/evaluate/upload")
    async def project_evaluate_upload(
        project_name: str,
        file: UploadFile = File(...),
        multi_label: bool = Query(False)
    ):
        """Evaluate an uploaded image with heatmap generation."""
        try:
            # Get project paths
            proj = get_project_info(project_name)
            if not proj:
                raise HTTPException(status_code=404, detail=f"Project {project_name} not found")
            
            # Save uploaded file temporarily to validation folder
            validation_dir = os.path.join(proj.path, 'validation')
            os.makedirs(validation_dir, exist_ok=True)
            
            # Generate unique filename
            import uuid
            ext = os.path.splitext(file.filename or 'image.png')[1] or '.png'
            temp_filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
            temp_path = os.path.join(validation_dir, temp_filename)
            
            # Save the file
            contents = await file.read()
            with open(temp_path, 'wb') as f:
                f.write(contents)
            
            info(f"Saved uploaded image to {temp_path}")
            
            # Run evaluation
            result = evaluate_with_heatmap(project_name, image_path=temp_path, multi_label=multi_label)
            
            # Optionally clean up temp file (or keep for reference)
            # os.remove(temp_path)
            
            return result
        except HTTPException:
            raise
        except Exception as ex:
            error(f"Upload evaluation failed for {project_name}: {ex}")
            return {"error": str(ex)}

    @app.post("/projects/{project_name}/augmentation/preview")
    def augmentation_preview(
        project_name: str,
        payload: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        phase = str(payload.get("phase", "")).lower()
        if phase not in {"train", "val"}:
            warning(f"Invalid augmentation preview phase '{phase}' requested for project {project_name}")
            return {"status": "error", "messageKey": "augmentation.preview_invalid_phase", "messageParams": {"phase": phase or "?"}}

        transforms_cfg = payload.get("transforms")
        if not isinstance(transforms_cfg, list):
            transforms_cfg = []

        force_new_image = payload.get("new_image", False)
        if force_new_image:
            _augmentation_preview_cache.clear()

        provided_path = payload.get("image_path")
        if provided_path:
            image_path = os.path.join("projects", project_name, provided_path)
            if not os.path.isfile(image_path):
                image_path = None
        else:
            image_path = None
        
        if not image_path:
            image_path = _select_preview_image(project_name, phase)
        
        if not image_path:
            warning(f"No preview images available for project {project_name}")
            return {"status": "error", "messageKey": "augmentation.preview_no_images", "messageParams": {"project": project_name}}

        try:
            cache_key = image_path
            
            if cache_key in _augmentation_preview_cache:
                original_image = _augmentation_preview_cache[cache_key]
            else:
                _augmentation_preview_cache.clear()
                
                with Image.open(image_path) as src:
                    loaded_image = src.convert("RGB")
                    
                    w, h = loaded_image.size
                    min_dim = min(w, h)
                    left = (w - min_dim) // 2
                    top = (h - min_dim) // 2
                    cropped_image = loaded_image.crop((left, top, left + min_dim, top + min_dim))
                    
                    preview_size = 512
                    if min_dim > preview_size:
                        cropped_image = cropped_image.resize((preview_size, preview_size), Image.Resampling.LANCZOS)
                    
                    _augmentation_preview_cache[cache_key] = cropped_image
                    original_image = cropped_image
            
            original_base64 = _encode_image_to_base64(original_image)
            augmented_image = _apply_augmentation_preview(original_image, transforms_cfg)
            augmented_base64 = _encode_image_to_base64(augmented_image)
        except Exception as exc:
            error(f"Augmentation preview failed for {project_name}: {exc}")
            return {"status": "error", "messageKey": "augmentation.preview_failed", "messageParams": {"error": str(exc)}}

        rel_path = os.path.relpath(image_path, os.path.join("projects", project_name)).replace("\\", "/")
        info(f"Augmentation preview generated for {project_name} ({phase})")
        return {
            "status": "success",
            "phase": phase,
            "image_path": rel_path,
            "original_image": original_base64,
            "augmented_image": augmented_base64
        }

    info("FastAPI app created")
    return app
