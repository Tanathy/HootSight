from __future__ import annotations

"""
Dataset Editor Metadata Store - Legacy compatibility wrapper.

This module wraps the centralized characteristics_db for dataset editor operations.
The actual storage is now in characteristics.db (SQLite) using:
- snapshots table: image metadata with hash-based IDs
- build_cache table: incremental build data

Legacy methods are preserved for backward compatibility but delegate to characteristics_db.
"""

import json
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, Optional

from system.log import warning
from system import characteristics_db as cdb


def _dump_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _convert_to_new_format(key: str, old_data: dict) -> dict:
    """Convert old snapshot format to new normalized format."""
    return {
        'relative_path': key,
        'relative_dir': old_data.get('relative_dir', ''),
        'filename': old_data.get('filename', ''),
        'extension': old_data.get('extension', ''),
        'category': old_data.get('category', ''),
        'width': old_data.get('width', 0),
        'height': old_data.get('height', 0),
        'min_dimension': old_data.get('min_dimension', 0),
        'tags': old_data.get('tags', []),
        'crop': old_data.get('bounds', old_data.get('crop', {})),
    }


def _convert_from_new_format(new_data: dict) -> dict:
    """Convert new normalized format back to old format for compatibility."""
    crop = new_data.get('crop', {})
    return {
        'id': new_data.get('relative_path', ''),
        'relative_dir': new_data.get('relative_dir', ''),
        'relative_path': new_data.get('relative_path', ''),
        'filename': new_data.get('filename', ''),
        'extension': new_data.get('extension', ''),
        'category': new_data.get('category', ''),
        'width': new_data.get('width', 0),
        'height': new_data.get('height', 0),
        'min_dimension': new_data.get('min_dimension', 0),
        'annotation': ', '.join(new_data.get('tags', [])),
        'tags': new_data.get('tags', []),
        'bounds': {
            'zoom': crop.get('zoom', 1.0),
            'center_x': crop.get('center_x', 0.5),
            'center_y': crop.get('center_y', 0.5),
        },
        'has_custom_bounds': crop.get('zoom', 1.0) != 1.0 or crop.get('center_x', 0.5) != 0.5 or crop.get('center_y', 0.5) != 0.5,
    }


class EditorStore:
    """
    Dataset Editor Store - now backed by characteristics.db.
    
    This class provides the same interface as before but delegates
    all storage operations to the centralized characteristics_db module.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.project_name = project_root.name
        self._lock = RLock()
        
        # Migrate old database if needed
        cdb.migrate_from_dataset_index(self.project_name)
        
        # Run legacy migrations
        self._migrate_legacy_store_files()
        self._migrate_legacy_json()
        self._migrate_snapshot_cache()

    def bulk_upsert(self, payloads: Dict[str, dict]) -> None:
        """Insert or update multiple records."""
        if not payloads:
            return
        
        records = {key: _convert_to_new_format(key, data) for key, data in payloads.items()}
        with self._lock:
            cdb.snapshot_upsert(self.project_name, records)

    def upsert(self, key: str, payload: dict) -> None:
        """Insert or update a single record."""
        if not key:
            return
        self.bulk_upsert({key: payload})

    def prune(self, valid_keys: Iterable[str]) -> None:
        """Remove all records not in valid_keys."""
        with self._lock:
            cdb.snapshot_prune(self.project_name, valid_keys)

    def remove(self, keys: Iterable[str]) -> None:
        """Remove specific records."""
        with self._lock:
            cdb.snapshot_remove(self.project_name, keys)

    def get(self, key: str) -> Optional[dict]:
        """Get a single record by key (relative_path)."""
        if not key:
            return None
        with self._lock:
            data = cdb.snapshot_get(self.project_name, key)
            if data:
                return _convert_from_new_format(data)
            return None

    def items(self) -> Dict[str, dict]:
        """Get all records."""
        with self._lock:
            all_data = cdb.snapshot_load_all(self.project_name)
            return {key: _convert_from_new_format(data) for key, data in all_data.items()}

    def keys(self) -> list[str]:
        """Get all record keys (relative_paths)."""
        with self._lock:
            all_data = cdb.snapshot_load_all(self.project_name)
            return list(all_data.keys())

    def count(self) -> int:
        """Get count of records."""
        return cdb.snapshot_count(self.project_name)

    def load_snapshot(self) -> Dict[str, dict]:
        """Load all snapshot records."""
        with self._lock:
            all_data = cdb.snapshot_load_all(self.project_name)
            return {key: _convert_from_new_format(data) for key, data in all_data.items()}

    def snapshot_count(self) -> int:
        """Get count of snapshots."""
        return cdb.snapshot_count(self.project_name)

    def write_snapshot(self, records: Dict[str, dict]) -> None:
        """Write snapshot records (replaces existing)."""
        with self._lock:
            # Clear and rewrite
            cdb.snapshot_prune(self.project_name, [])  # Clear all
            if records:
                converted = {key: _convert_to_new_format(key, data) for key, data in records.items()}
                cdb.snapshot_upsert(self.project_name, converted)

    def update_snapshot_entry(self, key: str, data: dict) -> None:
        """Update a single snapshot entry."""
        with self._lock:
            converted = _convert_to_new_format(key, data)
            cdb.snapshot_upsert(self.project_name, {key: converted})

    def remove_snapshot_entries(self, keys: Iterable[str]) -> None:
        """Remove snapshot entries."""
        with self._lock:
            cdb.snapshot_remove(self.project_name, keys)

    def load_build_cache(self) -> Dict[str, dict]:
        """Load build cache."""
        return cdb.build_cache_load(self.project_name)

    def update_build_cache(self, updates: Dict[str, dict], deletions: Iterable[str]) -> None:
        """Update build cache."""
        cdb.build_cache_update(self.project_name, updates, deletions)

    # =========================================================================
    # Legacy migration methods
    # =========================================================================

    def _migrate_legacy_store_files(self) -> None:
        """Migrate old .editor_store files to dataset_editor naming."""
        legacy_pattern = ".editor_store"
        for child in self.project_root.glob(f"{legacy_pattern}*"):
            replacement = child.name.replace(legacy_pattern, "dataset_editor", 1)
            target = child.with_name(replacement)
            if target.exists():
                continue
            try:
                child.rename(target)
            except OSError as exc:
                warning(f"Failed to rename legacy editor store file {child}: {exc}")

    def _migrate_legacy_json(self) -> None:
        """Migrate old editor.json format."""
        json_path = self.project_root / "editor.json"
        if not json_path.exists():
            return
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            self._archive_legacy_file(json_path)
            return
        images = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(images, dict):
            self._archive_legacy_file(json_path)
            return
        entries: Dict[str, dict] = {}
        for key, value in images.items():
            if isinstance(value, dict):
                entries[str(key)] = value
        if entries:
            try:
                self.bulk_upsert(entries)
            except Exception as exc:
                warning(f"Failed to migrate legacy editor.json payload: {exc}")
        self._archive_legacy_file(json_path)

    def _archive_legacy_file(self, path: Path) -> None:
        """Archive a legacy file."""
        target = path.with_suffix(path.suffix + ".bak")
        counter = 1
        while target.exists():
            counter += 1
            target = path.with_suffix(f"{path.suffix}.bak{counter}")
        try:
            path.rename(target)
        except OSError as exc:
            warning(f"Unable to archive legacy file {path}: {exc}")

    def _migrate_snapshot_cache(self) -> None:
        """Migrate old .project_index_cache.json format."""
        cache_path = self.project_root / ".project_index_cache.json"
        if not cache_path.exists():
            return
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        records = {}
        items = payload.get("items") if isinstance(payload, dict) else None
        if isinstance(items, list):
            for entry in items:
                record_id = entry.get("id") if isinstance(entry, dict) else None
                if record_id:
                    records[record_id] = entry
        if records:
            self.write_snapshot(records)
        backup = cache_path.with_suffix(".legacy")
        try:
            cache_path.rename(backup)
        except OSError as exc:
            warning(f"Unable to archive legacy dataset snapshot cache: {exc}")


__all__ = ["EditorStore"]
