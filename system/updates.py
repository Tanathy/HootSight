from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

from system.log import error, info, success, warning
from system.coordinator_settings import lang

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REMOTE_BASE_URL = "https://raw.githubusercontent.com/Tanathy/HootSight/refs/heads/main/"
REMOTE_CONFIG_PATH = "config/config.json"
LOCAL_CHECKSUM_PATH = os.path.join(ROOT_DIR, "config", "checksum.json")
HTTP_TIMEOUT = 20
SKIP_PATHS: Set[str] = {"config/config.json"}
DEFAULT_HEADERS = {"User-Agent": "Hootsight-Updater/1.0"}


class UpdateError(RuntimeError):
    """Raised when the update subsystem encounters an unrecoverable error."""


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _safe_join(path: str) -> str:
    normalized = _normalize_path(path)
    full_path = os.path.abspath(os.path.join(ROOT_DIR, normalized))
    if not full_path.startswith(ROOT_DIR):
        raise UpdateError(lang("updates.log.path_escape", path=normalized))
    return full_path


def _load_local_checksums() -> Dict[str, str]:
    try:
        with open(LOCAL_CHECKSUM_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise UpdateError(lang("updates.log.local_checksum_invalid", error="not a mapping"))
        cleaned = {_normalize_path(key): str(value) for key, value in data.items()}
        return cleaned
    except FileNotFoundError:
        warning(lang("updates.log.local_checksum_missing"))
        return {}
    except json.JSONDecodeError as exc:
        raise UpdateError(lang("updates.log.local_checksum_invalid", error=str(exc))) from exc


def _write_local_checksums(entries: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(LOCAL_CHECKSUM_PATH), exist_ok=True)
    with open(LOCAL_CHECKSUM_PATH, "w", encoding="utf-8") as handle:
        json.dump(dict(sorted(entries.items())), handle, indent=2, sort_keys=True)


def _http_get(url: str) -> bytes:
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:  # noqa: BLE001 - surface full context
        raise UpdateError(str(exc)) from exc


def _http_get_json(url: str) -> Dict[str, Any]:
    payload = _http_get(url)
    try:
        data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise UpdateError(str(exc)) from exc
    if not isinstance(data, dict):
        raise UpdateError(lang("updates.log.remote_payload_invalid", url=url))
    return data


def _fetch_remote_config() -> Dict[str, Any]:
    url = f"{REMOTE_BASE_URL}{REMOTE_CONFIG_PATH}"
    try:
        return _http_get_json(url)
    except UpdateError as exc:
        raise UpdateError(lang("updates.log.remote_config_failed", error=str(exc))) from exc


def _resolve_remote_checksum_path(remote_config: Dict[str, Any]) -> str:
    config_dir = remote_config.get("paths", {}).get("config_dir", "config")
    checksum_rel = _normalize_path(f"{config_dir.rstrip('/')}/checksum.json")
    return checksum_rel


def _fetch_remote_manifest(remote_config: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    checksum_rel = _resolve_remote_checksum_path(remote_config)
    checksum_url = f"{REMOTE_BASE_URL}{checksum_rel}"
    try:
        payload = _http_get_json(checksum_url)
    except UpdateError as exc:
        raise UpdateError(lang("updates.log.remote_checksum_failed", error=str(exc))) from exc

    manifest: Dict[str, str] = {}
    for key, value in payload.items():
        normalized = _normalize_path(key)
        manifest[normalized] = str(value)
    return manifest, checksum_url


def _build_plan(
    selected_paths: Optional[Iterable[str]] = None
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str], List[Dict[str, Any]], List[str], List[str], str]:
    remote_config = _fetch_remote_config()
    remote_manifest, checksum_url = _fetch_remote_manifest(remote_config)
    local_manifest = _load_local_checksums()

    skip_set = {_normalize_path(item) for item in SKIP_PATHS}
    normalized_selection: Optional[Set[str]] = None
    if selected_paths is not None:
        normalized_selection = {_normalize_path(path) for path in selected_paths}

    plan_entries: List[Dict[str, Any]] = []
    for path, remote_hash in remote_manifest.items():
        if path in skip_set:
            continue
        if normalized_selection is not None and path not in normalized_selection:
            continue
        local_hash = local_manifest.get(path)
        if local_hash == remote_hash:
            continue
        status_key = "updates.status.missing" if local_hash is None else "updates.status.outdated"
        plan_entries.append(
            {
                "path": path,
                "status": "missing" if local_hash is None else "outdated",
                "status_label": lang(status_key),
                "local_checksum": local_hash,
                "remote_checksum": remote_hash,
            }
        )

    orphaned = sorted(path for path in local_manifest.keys() if path not in remote_manifest and path not in skip_set)
    missing_targets: List[str] = []
    if normalized_selection is not None:
        missing_targets = sorted(path for path in normalized_selection if path not in remote_manifest)

    return (
        remote_config,
        remote_manifest,
        local_manifest,
        plan_entries,
        orphaned,
        missing_targets,
        checksum_url,
    )


def check_for_updates(selected_paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    info(lang("updates.log.check_started"))
    timestamp = datetime.utcnow().isoformat() + "Z"

    try:
        (
            remote_config,
            remote_manifest,
            local_manifest,
            entries,
            orphaned,
            missing_targets,
            checksum_url,
        ) = _build_plan(selected_paths)
    except UpdateError as exc:
        error(lang("updates.log.check_failed", error=str(exc)))
        return {
            "status": "error",
            "message": lang("updates.api.check_failed", error=str(exc)),
            "checked_at": timestamp,
        }

    count = len(entries)
    info(lang("updates.log.check_complete", count=count))

    if count:
        message = lang("updates.api.check_success", count=count)
    else:
        message = lang("updates.api.check_no_updates")

    return {
        "status": "success",
        "message": message,
        "has_updates": bool(count),
        "files": entries,
        "orphaned": orphaned,
        "missing_targets": missing_targets,
        "skipped": sorted(SKIP_PATHS),
        "reference": {
            "base_url": REMOTE_BASE_URL,
            "config_url": f"{REMOTE_BASE_URL}{REMOTE_CONFIG_PATH}",
            "checksum_url": checksum_url,
        },
        "checked_at": timestamp,
        "remote_manifest_size": len(remote_manifest),
        "local_manifest_size": len(local_manifest),
        "remote_version_hint": remote_config.get("version"),
    }


def _download_and_store(path: str, remote_manifest: Dict[str, str], local_manifest: Dict[str, str]) -> None:
    url = f"{REMOTE_BASE_URL}{path}"
    payload = _http_get(url)
    target_path = _safe_join(path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as handle:
        handle.write(payload)
    local_manifest[path] = remote_manifest[path]


def apply_updates(selected_paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    info(lang("updates.log.apply_started"))
    timestamp = datetime.utcnow().isoformat() + "Z"

    try:
        (
            remote_config,
            remote_manifest,
            local_manifest,
            entries,
            orphaned,
            missing_targets,
            checksum_url,
        ) = _build_plan(selected_paths)
    except UpdateError as exc:
        error(lang("updates.log.apply_failed", error=str(exc)))
        return {
            "status": "error",
            "message": lang("updates.api.apply_failed", error=str(exc)),
            "checked_at": timestamp,
        }

    if not entries:
        info(lang("updates.log.apply_nothing"))
        return {
            "status": "noop",
            "message": lang("updates.api.apply_nothing"),
            "updated": [],
            "failed": [],
            "orphaned": orphaned,
            "missing_targets": missing_targets,
            "skipped": sorted(SKIP_PATHS),
            "reference": {
                "base_url": REMOTE_BASE_URL,
                "config_url": f"{REMOTE_BASE_URL}{REMOTE_CONFIG_PATH}",
                "checksum_url": checksum_url,
            },
            "checked_at": timestamp,
        }

    updated: List[str] = []
    failed: List[Dict[str, Any]] = []

    for entry in entries:
        path = entry["path"]
        try:
            _download_and_store(path, remote_manifest, local_manifest)
            updated.append(path)
            success(lang("updates.log.apply_file_success", path=path))
        except UpdateError as exc:
            failed.append({"path": path, "error": str(exc)})
            error(lang("updates.log.apply_file_failed", path=path, error=str(exc)))
        except OSError as exc:
            failed.append({"path": path, "error": str(exc)})
            error(lang("updates.log.apply_file_failed", path=path, error=str(exc)))

    if updated:
        _write_local_checksums(local_manifest)

    if failed:
        warning(lang("updates.log.apply_partial", updated=len(updated), failed=len(failed)))
        status = "partial"
        message = lang("updates.api.apply_partial", updated=len(updated), failed=len(failed))
    else:
        success(lang("updates.log.apply_complete", count=len(updated)))
        status = "success"
        message = lang("updates.api.apply_success", updated=len(updated))

    return {
        "status": status,
        "message": message,
        "updated": updated,
        "failed": failed,
        "orphaned": orphaned,
        "missing_targets": missing_targets,
        "skipped": sorted(SKIP_PATHS),
        "reference": {
            "base_url": REMOTE_BASE_URL,
            "config_url": f"{REMOTE_BASE_URL}{REMOTE_CONFIG_PATH}",
            "checksum_url": checksum_url,
        },
        "checked_at": timestamp,
    }
