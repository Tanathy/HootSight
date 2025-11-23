from __future__ import annotations

import dbm
import json
import pickle
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, Optional, Sequence

from system.log import warning

FINGERPRINT_FIELD = "_fingerprint"

def _dump_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

class EditorStore:

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._legacy_dbm_path = project_root / "dataset_editor"
        self._db_path = project_root / "dataset_index.db"
        self._lock = RLock()
        self._migrate_legacy_store_files()
        self._ensure_schema()
        self._migrate_legacy_json()
        self._migrate_from_dbm()
        self._migrate_snapshot_cache()

    def bulk_upsert(self, payloads: Dict[str, dict]) -> None:
        if not payloads:
            return
        rows = [(key, _dump_json(payload)) for key, payload in payloads.items()]
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO payloads(id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json
                """,
                rows,
            )
            self._ensure_status_rows(conn, list(payloads.keys()))

    def upsert(self, key: str, payload: dict) -> None:
        if not key:
            return
        self.bulk_upsert({key: payload})

    def prune(self, valid_keys: Iterable[str]) -> None:
        keep = set(valid_keys)
        with self._lock, self._connect() as conn:
            if keep:
                placeholders = ",".join(["?"] * len(keep))
                conn.execute(f"DELETE FROM payloads WHERE id NOT IN ({placeholders})", tuple(keep))
                conn.execute(f"DELETE FROM dataset_status WHERE id NOT IN ({placeholders})", tuple(keep))
            else:
                conn.execute("DELETE FROM payloads")
                conn.execute("DELETE FROM dataset_status")

    def remove(self, keys: Iterable[str]) -> None:
        target = list(keys)
        if not target:
            return
        with self._lock, self._connect() as conn:
            conn.executemany("DELETE FROM payloads WHERE id=?", [(key,) for key in target])
            conn.executemany("DELETE FROM dataset_status WHERE id=?", [(key,) for key in target])
            conn.executemany("DELETE FROM snapshots WHERE id=?", [(key,) for key in target])
            conn.executemany("DELETE FROM build_cache WHERE id=?", [(key,) for key in target])

    def get(self, key: str) -> Optional[dict]:
        if not key:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM payloads WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def items(self) -> Dict[str, dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id, payload_json FROM payloads").fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def keys(self) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id FROM payloads").fetchall()
        return [row[0] for row in rows]

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            (value,) = conn.execute("SELECT COUNT(*) FROM payloads").fetchone()
        return int(value)

    def should_update(self, key: str, fingerprint: dict) -> bool:
        if not key:
            return False
        existing = self.get(key)
        if not existing:
            return True
        return existing.get(FINGERPRINT_FIELD) != fingerprint

    def load_snapshot(self) -> Dict[str, dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id, data_json FROM snapshots").fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def snapshot_count(self) -> int:
        with self._lock, self._connect() as conn:
            (value,) = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()
        return int(value)

    def write_snapshot(self, records: Dict[str, dict]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM snapshots")
            rows = [(key, _dump_json(value)) for key, value in records.items()]
            conn.executemany("INSERT INTO snapshots(id, data_json) VALUES (?, ?)", rows)
            self._ensure_status_rows(conn, list(records.keys()))

    def update_snapshot_entry(self, key: str, data: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO snapshots(id, data_json)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET data_json=excluded.data_json
                """,
                (key, _dump_json(data)),
            )
            self._ensure_status_rows(conn, [key])

    def remove_snapshot_entries(self, keys: Iterable[str]) -> None:
        target = list(keys)
        if not target:
            return
        with self._lock, self._connect() as conn:
            conn.executemany("DELETE FROM snapshots WHERE id=?", [(key,) for key in target])

    def pending_dataset_ids(self) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM dataset_status WHERE in_dataset=0 OR needs_update=1",
            ).fetchall()
        return [row[0] for row in rows]

    def mark_dataset_built(self, record_id: str) -> None:
        self._set_status(record_id, in_dataset=1, needs_update=0)

    def mark_dataset_needs_update(self, record_id: str) -> None:
        with self._lock, self._connect() as conn:
            self._ensure_status_rows(conn, [record_id])
            conn.execute("UPDATE dataset_status SET needs_update=1 WHERE id=? AND in_dataset=1", (record_id,))

    def mark_dataset_not_built(self, record_id: str) -> None:
        self._set_status(record_id, in_dataset=0)

    def remove_status(self, record_ids: Iterable[str]) -> None:
        target = list(record_ids)
        if not target:
            return
        with self._lock, self._connect() as conn:
            conn.executemany("DELETE FROM dataset_status WHERE id=?", [(key,) for key in target])

    def _set_status(self, record_id: str, *, in_dataset: Optional[int] = None, needs_update: Optional[int] = None) -> None:
        with self._lock, self._connect() as conn:
            self._ensure_status_rows(conn, [record_id])
            assignments: list[str] = []
            params: list[int | str] = []
            if in_dataset is not None:
                assignments.append("in_dataset=?")
                params.append(int(bool(in_dataset)))
            if needs_update is not None:
                assignments.append("needs_update=?")
                params.append(int(bool(needs_update)))
            params.append(record_id)
            if assignments:
                conn.execute(f"UPDATE dataset_status SET {', '.join(assignments)} WHERE id=?", tuple(params))

    def load_build_cache(self) -> Dict[str, dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT id, data_json FROM build_cache").fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def update_build_cache(self, updates: Dict[str, dict], deletions: Iterable[str]) -> None:
        delete_rows = [(key,) for key in deletions]
        upsert_rows = [(key, _dump_json(value)) for key, value in updates.items()]
        with self._lock, self._connect() as conn:
            if delete_rows:
                conn.executemany("DELETE FROM build_cache WHERE id=?", delete_rows)
            if upsert_rows:
                conn.executemany(
                    """
                    INSERT INTO build_cache(id, data_json)
                    VALUES (?, ?)
                    ON CONFLICT(id) DO UPDATE SET data_json=excluded.data_json
                    """,
                    upsert_rows,
                )

    def _connect(self):
        return sqlite3.connect(self._db_path)

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS payloads (
                    id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_status (
                    id TEXT PRIMARY KEY,
                    in_dataset INTEGER NOT NULL DEFAULT 0,
                    needs_update INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS build_cache (
                    id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dataset_pending ON dataset_status(in_dataset, needs_update);
                """
            )

    def _ensure_status_rows(self, conn: sqlite3.Connection, keys: Sequence[str]) -> None:
        rows = [(key,) for key in keys]
        if not rows:
            return
        conn.executemany(
            """
            INSERT INTO dataset_status(id, in_dataset, needs_update)
            VALUES (?, 0, 0)
            ON CONFLICT(id) DO NOTHING
            """,
            rows,
        )

    def _migrate_legacy_store_files(self) -> None:
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
            if not self._legacy_store_exists():
                self._write_legacy_dbm(entries)
        self._archive_legacy_file(json_path)

    def _legacy_store_exists(self) -> bool:
        stem = self._legacy_dbm_path.name
        return any(child for child in self.project_root.glob(f"{stem}*") if child.is_file())

    def _write_legacy_dbm(self, entries: Dict[str, dict]) -> None:
        if not entries:
            return
        try:
            with dbm.open(str(self._legacy_dbm_path), "n") as db:
                for key, value in entries.items():
                    encoded = key.encode("utf-8") if not isinstance(key, (bytes, bytearray)) else key
                    db[encoded] = pickle.dumps(value)
        except Exception as exc:
            warning(f"Failed to persist legacy dataset editor DBM cache: {exc}")

    def _archive_legacy_file(self, path: Path) -> None:
        target = path.with_suffix(path.suffix + ".bak")
        counter = 1
        while target.exists():
            counter += 1
            target = path.with_suffix(f"{path.suffix}.bak{counter}")
        try:
            path.rename(target)
        except OSError as exc:
            warning(f"Unable to archive legacy file {path}: {exc}")

    def _migrate_from_dbm(self) -> None:
        if not self._legacy_store_exists():
            return
        entries: Dict[str, dict] = {}
        try:
            with dbm.open(str(self._legacy_dbm_path), "r") as db:
                for raw_key in db.keys():
                    key = raw_key.decode("utf-8") if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
                    entries[key] = pickle.loads(db[raw_key])
        except Exception as exc:
            warning(f"Failed to read legacy dataset editor DBM cache: {exc}")
            entries = {}
        if entries:
            self.bulk_upsert(entries)
        for child in self.project_root.glob("dataset_editor*"):
            backup = child.with_suffix(child.suffix + ".legacy")
            try:
                child.rename(backup)
            except OSError as exc:
                warning(f"Unable to archive legacy dataset editor file {child}: {exc}")

    def _migrate_snapshot_cache(self) -> None:
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

__all__ = ["EditorStore", "FINGERPRINT_FIELD"]
