from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from system.coordinator_settings import SETTINGS
from system.log import info, warning

LABEL_MANIFEST = "labels.json"
ROOT_DIR = Path(__file__).resolve().parent.parent


def _as_absolute(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _resolve_projects_dir(override: Optional[str] = None) -> Optional[Path]:
    if override:
        return _as_absolute(override)

    try:
        paths_cfg = SETTINGS['paths']
    except Exception:
        raise ValueError("Missing required 'paths' section in config/config.json")

    base_rel = paths_cfg.get('projects_dir') or paths_cfg.get('models_dir')
    if not base_rel:
        warning("projects_dir or models_dir must be defined in config/config.json to persist labels")
        return None
    return _as_absolute(base_rel)


def _manifest_path(project_name: str, base_dir: Path) -> Path:
    return base_dir / project_name / 'model' / LABEL_MANIFEST


def persist_project_labels(project_name: str, labels: Dict[int, str], task: str,
                           base_override: Optional[str] = None) -> Optional[Path]:
    """Persist deterministic class labels for a project.
    
    Args:
        labels: Dict mapping index to label name, e.g. {0: "cat", 1: "dog"}
    """
    if not labels:
        warning(f"Skipping label persistence for {project_name}: labels dict is empty")
        return None

    base_dir = _resolve_projects_dir(base_override)
    if base_dir is None:
        warning(f"Cannot persist labels for {project_name}: unresolved projects directory")
        return None

    manifest_path = _manifest_path(project_name, base_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert int keys to string for JSON compatibility
    labels_json = {str(k): v for k, v in labels.items()}

    payload: Dict[str, Any] = {
        "project": project_name,
        "task": task,
        "labels": labels_json,
        "num_classes": len(labels),
        "generated_at": datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    }

    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    info(f"Persisted {len(labels)} labels for {project_name} -> {manifest_path}")
    return manifest_path


def load_project_labels(project_name: str, base_override: Optional[str] = None) -> Dict[int, str]:
    """Load persisted project labels if available.
    
    Returns:
        Dict mapping index to label name, e.g. {0: "cat", 1: "dog"}
    """
    try:
        base_dir = _resolve_projects_dir(base_override)
    except ValueError:
        return {}
    if base_dir is None:
        return {}

    manifest_path = _manifest_path(project_name, base_dir)
    if not manifest_path.exists():
        return {}

    try:
        with manifest_path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
    except Exception as exc:
        warning(f"Failed to read label manifest for {project_name}: {exc}")
        return {}

    labels = data.get('labels') or {}
    
    # Handle both dict and legacy list format
    if isinstance(labels, dict):
        # Convert string keys back to int
        return {int(k): str(v) for k, v in labels.items()}
    elif isinstance(labels, list):
        # Legacy list format - convert to dict
        return {idx: str(name) for idx, name in enumerate(labels)}
    else:
        warning(f"Invalid label manifest format for {project_name}, ignoring")
        return {}
