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

from fastapi import FastAPI, Body, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

import torch
from PIL import Image
from torchvision import transforms as tv_transforms

from system.log import info, success, error, warning
from system.coordinator_settings import SETTINGS
from system.dataset_discovery import discover_projects, get_project_info, get_image_files
from system.training import start_training, stop_training, get_training_status, get_training_status_all
from system.heatmap import generate_project_heatmap, evaluate_with_heatmap
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
    FolderNode,
    ImageItem,
    PaginatedItems,
    ProjectSummary,
    StatsSummary,
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


def _localize_schema(obj: Any) -> Any:
    if isinstance(obj, str):
        if obj.startswith("schema."):
            return lang(obj)
        return obj
    elif isinstance(obj, dict):
        return {key: _localize_schema(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_localize_schema(item) for item in obj]
    else:
        return obj


def _encode_image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _candidate_dataset_dirs(project_name: str, phase: str) -> List[str]:
    dataset_root = os.path.join("projects", project_name, "dataset")
    if not os.path.isdir(dataset_root):
        return []

    phase_dirs: List[str] = []
    normalized_phase = (phase or "").lower()
    if normalized_phase == "train":
        phase_dirs.extend(["train", "training"])
    elif normalized_phase == "val":
        phase_dirs.extend(["val", "validation", "eval"])

    resolved = []
    for suffix in phase_dirs:
        candidate = os.path.join(dataset_root, suffix)
        if os.path.isdir(candidate):
            resolved.append(candidate)

    resolved.append(dataset_root)
    return resolved


def _select_preview_image(project_name: str, phase: str) -> Optional[str]:
    for directory in _candidate_dataset_dirs(project_name, phase):
        images = get_image_files(directory)
        if images:
            return random.choice(images)
    return None


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
        message = lang("dataset_editor.api.project_not_found", project=project_name)
        raise HTTPException(status_code=404, detail=message) from exc


def _raise_image_not_found(project_name: str, exc: ImageNotFoundError) -> NoReturn:
    message = lang("dataset_editor.api.image_not_found", project=project_name)
    raise HTTPException(status_code=404, detail=message) from exc


def _validate_dataset_editor_page_size(page_size: int) -> None:
    if page_size not in DATASET_EDITOR_SETTINGS.page_sizes:
        allowed = ", ".join(str(size) for size in DATASET_EDITOR_SETTINGS.page_sizes)
        message = lang("dataset_editor.api.invalid_page_size", allowed=allowed)
        raise HTTPException(status_code=400, detail=message)


def _validate_dataset_editor_build_size(size: int) -> None:
    allowed = _dataset_editor_allowed_build_sizes()
    if size not in allowed:
        allowed_label = ", ".join(str(value) for value in sorted(allowed))
        message = lang("dataset_editor.api.invalid_size", allowed=allowed_label)
        raise HTTPException(status_code=400, detail=message)


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

    @app.get("/config")
    def get_config() -> Dict[str, Any]:
        return {"config": SETTINGS}

    @app.get("/config/schema")
    def get_config_schema() -> Dict[str, Any]:
        try:
            schema_path = os.path.join("config", "config_schema.json")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            localized_schema = _localize_schema(schema)
            return {"schema": localized_schema}
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

    @app.get("/docs/list")
    def docs_list() -> Dict[str, Any]:
        if docs_root is None:
            return {"docs": []}
        return {"docs": _list_doc_entries(docs_root)}

    @app.get("/docs/page")
    def docs_page(path: str = Query(..., min_length=1)) -> Dict[str, Any]:
        if docs_root is None:
            message = lang("docs.api.missing_root")
            warning("Documentation directory is not configured; cannot serve documentation page")
            return {"status": "error", "message": message}

        target = _resolve_doc_path(docs_root, path)
        if target is None or not target.is_file():
            message = lang("docs.api.not_found", path=path)
            warning(f"Documentation page '{path}' not found")
            return {"status": "error", "message": message}

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            message = lang("docs.api.decode_error", path=target.name)
            error(f"Documentation decode failed for {target}: {exc}")
            return {"status": "error", "message": message}
        except Exception as exc:
            message = lang("docs.api.read_failed", error=str(exc))
            error(f"Documentation read failed for {target}: {exc}")
            return {"status": "error", "message": message}

        rel_path = target.relative_to(docs_root).as_posix()
        title = _derive_doc_title(target)
        return {"status": "success", "path": rel_path, "title": title, "content": content}

    @app.post("/config")
    def update_config(config: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            cfg_path = os.path.join("config", "config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            success("System configuration updated")
            return {"status": "success"}
        except Exception as ex:
            error(f"Config update failed: {ex}")
            return {"status": "error", "message": str(ex)}

    @app.get("/projects/{project_name}/config")
    def get_project_config(project_name: str) -> Dict[str, Any]:
        try:
            cfg_path = os.path.join("projects", project_name, "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                return {"config": config}
            else:
                return {"error": "Project config not found"}
        except Exception as ex:
            error(f"Project config load failed: {ex}")
            return {"error": str(ex)}

    @app.post("/projects/{project_name}/config")
    def update_project_config(project_name: str, config: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            cfg_path = os.path.join("projects", project_name, "config.json")
            os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            success(f"Project {project_name} configuration updated")
            return {"status": "success"}
        except Exception as ex:
            error(f"Project config update failed: {ex}")
            return {"status": "error", "message": str(ex)}

    @app.get("/projects")
    def list_projects() -> Dict[str, Any]:
        try:
            projects = discover_projects()
            project_dicts = [project.to_dict() for project in projects]
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
            message = lang("projects.api.create_missing")
            return {"status": "error", "message": message}

        if len(name_value) < min_length or len(name_value) > max_length:
            message = lang("projects.api.create_length", min=min_length, max=max_length)
            return {"status": "error", "message": message}

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name_value):
            message = lang("projects.api.create_invalid")
            return {"status": "error", "message": message}

        project_path = (projects_root / name_value).resolve()
        try:
            project_path.relative_to(projects_root)
        except ValueError:
            message = lang("projects.api.create_invalid")
            return {"status": "error", "message": message}

        if project_path.exists():
            message = lang("projects.api.create_exists", name=name_value)
            return {"status": "error", "message": message}

        try:
            project_path.mkdir(parents=False, exist_ok=False)
            for subdir in ("dataset", "data_source", "model", "heatmaps", "validation"):
                (project_path / subdir).mkdir(parents=True, exist_ok=True)

            config_src = root_path / "config" / "config.json"
            config_dst = project_path / "config.json"
            if config_src.is_file():
                shutil.copy2(config_src, config_dst)
            else:
                with config_dst.open("w", encoding="utf-8") as cfg_file:
                    json.dump({}, cfg_file, indent=4, ensure_ascii=False)

            success(lang("projects.api.create_success", name=name_value))
        except Exception as ex:  # noqa: BLE001 - ensure structured error response
            error(f"Project creation failed for {name_value}: {ex}")
            message = lang("projects.api.create_error", error=str(ex))
            return {"status": "error", "message": message}

        project_details = None
        try:
            for project in discover_projects():
                if project.name == name_value:
                    project_details = project.to_dict()
                    break
        except Exception as ex:  # noqa: BLE001 - project list retrieval should not fail the request
            warning(f"Post-create project discovery failed for {name_value}: {ex}")

        response: Dict[str, Any] = {
            "status": "success",
            "message": lang("projects.api.create_success", name=name_value)
        }
        if project_details is not None:
            response["project"] = project_details
        else:
            response["project"] = {"name": name_value, "path": str(project_path)}

        return response

    @app.get("/projects/{project_name}")
    def project_details(project_name: str) -> Dict[str, Any]:
        try:
            info_obj = get_project_info(project_name)
            return {"project": info_obj.to_dict() if info_obj else None}
        except Exception as ex:
            error(f"Project info failed for {project_name}: {ex}")
            return {"error": str(ex)}

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

    @app.get("/dataset/editor/projects/{project_name}/folders", response_model=FolderNode)
    def dataset_editor_project_folders(project_name: str) -> FolderNode:
        project = _get_dataset_editor_project(project_name)
        return project.folder_tree()

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

    @app.get("/dataset/editor/projects/{project_name}/image")
    def dataset_editor_image(project_name: str, path: str = Query(..., min_length=1)) -> FileResponse:
        project = _get_dataset_editor_project(project_name)
        decoded = Path(urllib.parse.unquote(path))
        if decoded.is_absolute():
            message = lang("dataset_editor.api.invalid_image_path")
            raise HTTPException(status_code=400, detail=message)
        target = (project.data_source_dir / decoded).resolve()
        data_source_dir = project.data_source_dir.resolve()
        if data_source_dir not in target.parents and target != data_source_dir:
            message = lang("dataset_editor.api.invalid_image_path")
            raise HTTPException(status_code=400, detail=message)
        if not target.exists() or not target.is_file():
            message = lang("dataset_editor.api.image_missing")
            raise HTTPException(status_code=404, detail=message)
        return FileResponse(target)

    @app.post("/training/start")
    def training_start(
        project_name: str = Body(..., embed=True),
        model_name: str = Body("resnet50", embed=True),
        epochs: Optional[int] = Body(None, embed=True)
    ) -> Dict[str, Any]:
        try:
            return start_training(project_name=project_name, model_type="resnet", model_name=model_name, epochs=epochs)
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
            return get_training_status(training_id)
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
            return {"status": "error", "message": lang("updates.api.check_failed", error=str(ex))}

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
            return {"status": "error", "message": lang("updates.api.apply_failed", error=str(ex))}

    @app.get("/projects/{project_name}/heatmap")
    def project_heatmap(
        project_name: str,
        image_path: Optional[str] = Query(None),
        class_index: Optional[int] = Query(None),
        alpha: float = Query(0.5)
    ):
        try:
            data, meta = generate_project_heatmap(project_name, image_path=image_path, target_class=class_index, alpha=alpha)
            return Response(content=data, media_type="image/png")
        except Exception as ex:
            error(f"Heatmap generation failed for {project_name}: {ex}")
            return {"error": str(ex)}

    @app.get("/projects/{project_name}/evaluate")
    def project_evaluate(
        project_name: str,
        image_path: Optional[str] = Query(None),
        checkpoint_path: Optional[str] = Query(None)
    ):
        try:
            result = evaluate_with_heatmap(project_name, image_path=image_path, checkpoint_path=checkpoint_path)
            return result
        except Exception as ex:
            error(f"Evaluation failed for {project_name}: {ex}")
            return {"error": str(ex)}

    @app.post("/projects/{project_name}/augmentation/preview")
    def augmentation_preview(
        project_name: str,
        payload: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        phase = str(payload.get("phase", "")).lower()
        if phase not in {"train", "val"}:
            message = lang("augmentation.preview_invalid_phase", phase=phase or "?")
            warning(f"Invalid augmentation preview phase '{phase}' requested for project {project_name}")
            return {"status": "error", "message": message}

        transforms_cfg = payload.get("transforms")
        if not isinstance(transforms_cfg, list):
            transforms_cfg = []

        image_path = _select_preview_image(project_name, phase)
        if not image_path:
            message = lang("augmentation.preview_no_images", project=project_name)
            warning(f"No preview images available for project {project_name}")
            return {"status": "error", "message": message}

        try:
            with Image.open(image_path) as src:
                original_image = src.convert("RGB")
                original_base64 = _encode_image_to_base64(original_image)
                augmented_image = _apply_augmentation_preview(original_image, transforms_cfg)
                augmented_base64 = _encode_image_to_base64(augmented_image)
        except Exception as exc:  # noqa: BLE001 - surface error to caller
            error(f"Augmentation preview failed for {project_name}: {exc}")
            message = lang("augmentation.preview_failed", error=str(exc))
            return {"status": "error", "message": message}

        rel_path = os.path.relpath(image_path, os.path.join("projects", project_name)).replace("\\", "/")
        info(lang("augmentation.preview_generated", project=project_name, phase=phase))
        return {
            "status": "success",
            "phase": phase,
            "image_path": rel_path,
            "original_image": original_base64,
            "augmented_image": augmented_base64
        }

    info("FastAPI app created")
    return app
