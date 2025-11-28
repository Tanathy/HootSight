"""
Characteristics database for project metadata, statistics, and indexes.

This module provides a SQLite-based storage layer for:
- Project statistics (balance scores, analysis results)
- Dataset editor snapshots (image metadata with hash-based IDs)
- Dataset build status tracking
- Any computed characteristics that are expensive to calculate

Statistics are only computed on explicit request and cached for quick retrieval.
The database file is stored as 'characteristics.db' in each project folder.
"""

import os
import json
import sqlite3
import hashlib
import threading
from typing import Any, Dict, List, Optional, Iterable, Sequence
from pathlib import Path
from datetime import datetime

from system.log import info, warning, error
from system.coordinator_settings import SETTINGS


_DB_FILENAME = "characteristics.db"
_local = threading.local()


def _get_projects_root() -> Path:
    try:
        paths_cfg = SETTINGS['paths']
        projects_dir = paths_cfg.get('projects_dir', 'projects')
    except Exception:
        projects_dir = 'projects'
    root_path = Path(__file__).resolve().parents[1]
    return (root_path / projects_dir).resolve()


def _get_db_path(project_name: str) -> Path:
    return _get_projects_root() / project_name / _DB_FILENAME


def _get_connection(project_name: str) -> sqlite3.Connection:
    """Get or create a thread-local connection for the project database."""
    db_path = _get_db_path(project_name)
    cache_key = f"conn_{project_name}"
    
    conn = getattr(_local, cache_key, None)
    if conn is not None:
        return conn
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    setattr(_local, cache_key, conn)
    
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create database schema if not exists."""
    cursor = conn.cursor()
    cursor.executescript("""
        PRAGMA journal_mode=WAL;
        
        -- Project-level statistics and metadata
        CREATE TABLE IF NOT EXISTS project_stats (
            id INTEGER PRIMARY KEY,
            computed_at TEXT NOT NULL,
            dataset_type TEXT,
            image_count INTEGER DEFAULT 0,
            num_labels INTEGER DEFAULT 0,
            balance_score REAL DEFAULT 0.0,
            balance_status TEXT,
            balance_analysis_json TEXT,
            extra_json TEXT
        );
        
        -- Label/tag index - for quick lookups without full scan
        CREATE TABLE IF NOT EXISTS label_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL UNIQUE,
            count INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        
        -- Image snapshots for dataset editor (hash-based ID)
        CREATE TABLE IF NOT EXISTS snapshots (
            hash TEXT PRIMARY KEY,
            relative_path TEXT NOT NULL UNIQUE,
            relative_dir TEXT,
            filename TEXT NOT NULL,
            extension TEXT NOT NULL,
            category TEXT,
            width INTEGER DEFAULT 0,
            height INTEGER DEFAULT 0,
            min_dimension INTEGER DEFAULT 0,
            tags_json TEXT,
            crop_json TEXT,
            fingerprint_json TEXT,
            updated_at TEXT
        );
        
        -- Dataset build status tracking
        CREATE TABLE IF NOT EXISTS dataset_status (
            hash TEXT PRIMARY KEY,
            in_dataset INTEGER NOT NULL DEFAULT 0,
            needs_update INTEGER NOT NULL DEFAULT 0
        );
        
        -- Build cache for incremental builds
        CREATE TABLE IF NOT EXISTS build_cache (
            hash TEXT PRIMARY KEY,
            data_json TEXT NOT NULL
        );
        
        -- Generic key-value store for extensibility
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        
        CREATE INDEX IF NOT EXISTS idx_label_index_label ON label_index(label);
        CREATE INDEX IF NOT EXISTS idx_snapshots_relative_path ON snapshots(relative_path);
        CREATE INDEX IF NOT EXISTS idx_snapshots_category ON snapshots(category);
        CREATE INDEX IF NOT EXISTS idx_dataset_pending ON dataset_status(in_dataset, needs_update);
    """)
    conn.commit()


def compute_image_hash(relative_path: str) -> str:
    """
    Compute a stable hash for an image based on its relative path.
    This provides a consistent ID that survives file moves within the same relative location.
    """
    return hashlib.sha256(relative_path.encode('utf-8')).hexdigest()[:16]


# =============================================================================
# PROJECT STATISTICS (used by statistics module)
# =============================================================================

def get_cached_stats(project_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached project statistics from the database.
    Returns None if no stats have been computed yet.
    """
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return None
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT computed_at, dataset_type, image_count, num_labels,
                   balance_score, balance_status, balance_analysis_json, extra_json
            FROM project_stats
            WHERE id = 1
        """)
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        balance_analysis = {}
        if row['balance_analysis_json']:
            try:
                balance_analysis = json.loads(row['balance_analysis_json'])
            except json.JSONDecodeError:
                pass
        
        extra = {}
        if row['extra_json']:
            try:
                extra = json.loads(row['extra_json'])
            except json.JSONDecodeError:
                pass
        
        return {
            "computed_at": row['computed_at'],
            "dataset_type": row['dataset_type'],
            "image_count": row['image_count'],
            "num_labels": row['num_labels'],
            "balance_score": row['balance_score'],
            "balance_status": row['balance_status'],
            "balance_analysis": balance_analysis,
            **extra
        }
    except Exception as ex:
        warning(f"Failed to read cached stats for {project_name}: {ex}")
        return None


def save_stats(project_name: str, stats: Dict[str, Any]) -> bool:
    """
    Save computed statistics to the database.
    Overwrites any existing stats.
    """
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        
        computed_at = datetime.utcnow().isoformat()
        dataset_type = stats.get('dataset_type', '')
        image_count = stats.get('image_count', 0)
        num_labels = len(stats.get('labels', []))
        balance_score = stats.get('balance_score', 0.0)
        balance_status = stats.get('balance_status', '')
        
        balance_analysis = stats.get('balance_analysis', {})
        # Strip heavy data from balance_analysis before storing
        analysis_to_store = {
            k: v for k, v in balance_analysis.items()
            if k not in ('over_represented', 'under_represented', 'critical_shortage')
        }
        balance_analysis_json = json.dumps(analysis_to_store, ensure_ascii=False)
        
        # Store any extra lightweight fields
        extra = {
            'has_dataset': stats.get('has_dataset', False),
            'has_model_dir': stats.get('has_model_dir', False),
        }
        extra_json = json.dumps(extra, ensure_ascii=False)
        
        cursor.execute("DELETE FROM project_stats")
        cursor.execute("""
            INSERT INTO project_stats 
            (id, computed_at, dataset_type, image_count, num_labels, 
             balance_score, balance_status, balance_analysis_json, extra_json)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (computed_at, dataset_type, image_count, num_labels,
              balance_score, balance_status, balance_analysis_json, extra_json))
        
        conn.commit()
        info(f"Saved statistics for project {project_name}")
        return True
    except Exception as ex:
        error(f"Failed to save stats for {project_name}: {ex}")
        return False


def save_label_index(project_name: str, label_distribution: Dict[str, int]) -> bool:
    """Save label index to database for quick lookups."""
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        
        updated_at = datetime.utcnow().isoformat()
        
        cursor.execute("DELETE FROM label_index")
        
        for label, count in label_distribution.items():
            cursor.execute("""
                INSERT INTO label_index (label, count, updated_at)
                VALUES (?, ?, ?)
            """, (label, count, updated_at))
        
        conn.commit()
        return True
    except Exception as ex:
        error(f"Failed to save label index for {project_name}: {ex}")
        return False


def get_label_count(project_name: str) -> int:
    """Get the number of unique labels from the index."""
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return 0
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM label_index")
        row = cursor.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


def get_label_distribution(project_name: str) -> Dict[str, int]:
    """Get label distribution from the index."""
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return {}
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("SELECT label, count FROM label_index ORDER BY count DESC")
        return {row['label']: row['count'] for row in cursor.fetchall()}
    except Exception as ex:
        warning(f"Failed to read label distribution for {project_name}: {ex}")
        return {}


def clear_stats(project_name: str) -> bool:
    """Clear all cached statistics for a project."""
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM project_stats")
        cursor.execute("DELETE FROM label_index")
        conn.commit()
        info(f"Cleared statistics cache for project {project_name}")
        return True
    except Exception as ex:
        error(f"Failed to clear stats for {project_name}: {ex}")
        return False


def stats_exist(project_name: str) -> bool:
    """Check if statistics have been computed for a project."""
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return False
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM project_stats")
        row = cursor.fetchone()
        return row[0] > 0 if row else False
    except Exception:
        return False


# =============================================================================
# DATASET EDITOR SNAPSHOTS (used by dataset editor)
# =============================================================================

def _dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def snapshot_upsert(project_name: str, records: Dict[str, dict]) -> None:
    """
    Insert or update snapshot records.
    Key is the relative_path, value is the record data.
    """
    if not records:
        return
    
    conn = _get_connection(project_name)
    cursor = conn.cursor()
    updated_at = datetime.utcnow().isoformat()
    
    rows = []
    for relative_path, data in records.items():
        hash_id = compute_image_hash(relative_path)
        rows.append((
            hash_id,
            relative_path,
            data.get('relative_dir', ''),
            data.get('filename', ''),
            data.get('extension', ''),
            data.get('category', ''),
            data.get('width', 0),
            data.get('height', 0),
            data.get('min_dimension', 0),
            _dump_json(data.get('tags', [])),
            _dump_json(data.get('crop', {})),
            _dump_json(data.get('fingerprint', {})),
            data.get('updated_at', updated_at),
        ))
    
    cursor.executemany("""
        INSERT INTO snapshots (hash, relative_path, relative_dir, filename, extension,
                               category, width, height, min_dimension, tags_json,
                               crop_json, fingerprint_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(hash) DO UPDATE SET
            relative_path=excluded.relative_path,
            relative_dir=excluded.relative_dir,
            filename=excluded.filename,
            extension=excluded.extension,
            category=excluded.category,
            width=excluded.width,
            height=excluded.height,
            min_dimension=excluded.min_dimension,
            tags_json=excluded.tags_json,
            crop_json=excluded.crop_json,
            fingerprint_json=excluded.fingerprint_json,
            updated_at=excluded.updated_at
    """, rows)
    conn.commit()


def snapshot_load_all(project_name: str) -> Dict[str, dict]:
    """Load all snapshot records. Returns dict keyed by relative_path."""
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return {}
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT hash, relative_path, relative_dir, filename, extension,
                   category, width, height, min_dimension, tags_json,
                   crop_json, fingerprint_json, updated_at
            FROM snapshots
        """)
        
        result = {}
        for row in cursor.fetchall():
            tags = []
            crop = {}
            fingerprint = {}
            try:
                tags = json.loads(row['tags_json']) if row['tags_json'] else []
            except json.JSONDecodeError:
                pass
            try:
                crop = json.loads(row['crop_json']) if row['crop_json'] else {}
            except json.JSONDecodeError:
                pass
            try:
                fingerprint = json.loads(row['fingerprint_json']) if row['fingerprint_json'] else {}
            except json.JSONDecodeError:
                pass
            
            result[row['relative_path']] = {
                'hash': row['hash'],
                'relative_path': row['relative_path'],
                'relative_dir': row['relative_dir'],
                'filename': row['filename'],
                'extension': row['extension'],
                'category': row['category'],
                'width': row['width'],
                'height': row['height'],
                'min_dimension': row['min_dimension'],
                'tags': tags,
                'crop': crop,
                'fingerprint': fingerprint,
                'updated_at': row['updated_at'],
            }
        return result
    except Exception as ex:
        warning(f"Failed to load snapshots for {project_name}: {ex}")
        return {}


def snapshot_get(project_name: str, relative_path: str) -> Optional[dict]:
    """Get a single snapshot by relative_path."""
    hash_id = compute_image_hash(relative_path)
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return None
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT hash, relative_path, relative_dir, filename, extension,
                   category, width, height, min_dimension, tags_json,
                   crop_json, fingerprint_json, updated_at
            FROM snapshots WHERE hash = ?
        """, (hash_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        tags = []
        crop = {}
        fingerprint = {}
        try:
            tags = json.loads(row['tags_json']) if row['tags_json'] else []
        except json.JSONDecodeError:
            pass
        try:
            crop = json.loads(row['crop_json']) if row['crop_json'] else {}
        except json.JSONDecodeError:
            pass
        try:
            fingerprint = json.loads(row['fingerprint_json']) if row['fingerprint_json'] else {}
        except json.JSONDecodeError:
            pass
        
        return {
            'hash': row['hash'],
            'relative_path': row['relative_path'],
            'relative_dir': row['relative_dir'],
            'filename': row['filename'],
            'extension': row['extension'],
            'category': row['category'],
            'width': row['width'],
            'height': row['height'],
            'min_dimension': row['min_dimension'],
            'tags': tags,
            'crop': crop,
            'fingerprint': fingerprint,
            'updated_at': row['updated_at'],
        }
    except Exception as ex:
        warning(f"Failed to get snapshot for {relative_path}: {ex}")
        return None


def snapshot_remove(project_name: str, relative_paths: Iterable[str]) -> None:
    """Remove snapshots by relative_path."""
    paths = list(relative_paths)
    if not paths:
        return
    
    hashes = [(compute_image_hash(p),) for p in paths]
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.executemany("DELETE FROM snapshots WHERE hash = ?", hashes)
        cursor.executemany("DELETE FROM dataset_status WHERE hash = ?", hashes)
        cursor.executemany("DELETE FROM build_cache WHERE hash = ?", hashes)
        conn.commit()
    except Exception as ex:
        warning(f"Failed to remove snapshots: {ex}")


def snapshot_count(project_name: str) -> int:
    """Get count of snapshots."""
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return 0
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM snapshots")
        row = cursor.fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


def snapshot_prune(project_name: str, valid_paths: Iterable[str]) -> None:
    """Remove all snapshots not in valid_paths."""
    keep = set(valid_paths)
    if not keep:
        # Clear all
        try:
            conn = _get_connection(project_name)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM snapshots")
            cursor.execute("DELETE FROM dataset_status")
            cursor.execute("DELETE FROM build_cache")
            conn.commit()
        except Exception as ex:
            warning(f"Failed to clear snapshots: {ex}")
        return
    
    valid_hashes = {compute_image_hash(p) for p in keep}
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        
        # Get all current hashes
        cursor.execute("SELECT hash FROM snapshots")
        all_hashes = {row[0] for row in cursor.fetchall()}
        
        # Find hashes to remove
        to_remove = all_hashes - valid_hashes
        if to_remove:
            cursor.executemany("DELETE FROM snapshots WHERE hash = ?", [(h,) for h in to_remove])
            cursor.executemany("DELETE FROM dataset_status WHERE hash = ?", [(h,) for h in to_remove])
            cursor.executemany("DELETE FROM build_cache WHERE hash = ?", [(h,) for h in to_remove])
            conn.commit()
    except Exception as ex:
        warning(f"Failed to prune snapshots: {ex}")


def snapshot_should_update(project_name: str, relative_path: str, fingerprint: dict) -> bool:
    """Check if a snapshot needs to be updated based on fingerprint."""
    existing = snapshot_get(project_name, relative_path)
    if not existing:
        return True
    return existing.get('fingerprint') != fingerprint


# =============================================================================
# DATASET BUILD STATUS (used by dataset editor builder)
# =============================================================================

def status_ensure_rows(project_name: str, relative_paths: Sequence[str]) -> None:
    """Ensure status rows exist for given paths."""
    if not relative_paths:
        return
    
    rows = [(compute_image_hash(p),) for p in relative_paths]
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO dataset_status (hash, in_dataset, needs_update)
            VALUES (?, 0, 0)
            ON CONFLICT(hash) DO NOTHING
        """, rows)
        conn.commit()
    except Exception as ex:
        warning(f"Failed to ensure status rows: {ex}")


def status_get_pending(project_name: str) -> List[str]:
    """Get relative_paths of items pending build."""
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return []
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.relative_path
            FROM dataset_status ds
            JOIN snapshots s ON ds.hash = s.hash
            WHERE ds.in_dataset = 0 OR ds.needs_update = 1
        """)
        return [row[0] for row in cursor.fetchall()]
    except Exception as ex:
        warning(f"Failed to get pending status: {ex}")
        return []


def status_mark_built(project_name: str, relative_path: str) -> None:
    """Mark an item as built in dataset."""
    hash_id = compute_image_hash(relative_path)
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE dataset_status SET in_dataset = 1, needs_update = 0
            WHERE hash = ?
        """, (hash_id,))
        conn.commit()
    except Exception as ex:
        warning(f"Failed to mark built: {ex}")


def status_mark_needs_update(project_name: str, relative_path: str) -> None:
    """Mark an item as needing update."""
    hash_id = compute_image_hash(relative_path)
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE dataset_status SET needs_update = 1
            WHERE hash = ? AND in_dataset = 1
        """, (hash_id,))
        conn.commit()
    except Exception as ex:
        warning(f"Failed to mark needs update: {ex}")


def status_mark_not_built(project_name: str, relative_path: str) -> None:
    """Mark an item as not in dataset."""
    hash_id = compute_image_hash(relative_path)
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dataset_status (hash, in_dataset, needs_update)
            VALUES (?, 0, 0)
            ON CONFLICT(hash) DO UPDATE SET in_dataset = 0
        """, (hash_id,))
        conn.commit()
    except Exception as ex:
        warning(f"Failed to mark not built: {ex}")


# =============================================================================
# BUILD CACHE
# =============================================================================

def build_cache_load(project_name: str) -> Dict[str, dict]:
    """Load build cache."""
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return {}
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("SELECT hash, data_json FROM build_cache")
        result = {}
        for row in cursor.fetchall():
            try:
                result[row['hash']] = json.loads(row['data_json'])
            except json.JSONDecodeError:
                pass
        return result
    except Exception as ex:
        warning(f"Failed to load build cache: {ex}")
        return {}


def build_cache_update(project_name: str, updates: Dict[str, dict], deletions: Iterable[str]) -> None:
    """Update build cache."""
    delete_hashes = [(compute_image_hash(p),) for p in deletions]
    upsert_rows = [(compute_image_hash(k), _dump_json(v)) for k, v in updates.items()]
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        if delete_hashes:
            cursor.executemany("DELETE FROM build_cache WHERE hash = ?", delete_hashes)
        if upsert_rows:
            cursor.executemany("""
                INSERT INTO build_cache (hash, data_json)
                VALUES (?, ?)
                ON CONFLICT(hash) DO UPDATE SET data_json = excluded.data_json
            """, upsert_rows)
        conn.commit()
    except Exception as ex:
        warning(f"Failed to update build cache: {ex}")


# =============================================================================
# METADATA (generic key-value store)
# =============================================================================

def set_metadata(project_name: str, key: str, value: Any) -> bool:
    """Store arbitrary metadata in the key-value store."""
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        
        updated_at = datetime.utcnow().isoformat()
        value_json = json.dumps(value, ensure_ascii=False)
        
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value_json, updated_at)
            VALUES (?, ?, ?)
        """, (key, value_json, updated_at))
        
        conn.commit()
        return True
    except Exception as ex:
        error(f"Failed to set metadata {key} for {project_name}: {ex}")
        return False


def get_metadata(project_name: str, key: str, default: Any = None) -> Any:
    """Retrieve metadata from the key-value store."""
    db_path = _get_db_path(project_name)
    if not db_path.exists():
        return default
    
    try:
        conn = _get_connection(project_name)
        cursor = conn.cursor()
        cursor.execute("SELECT value_json FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        
        if row is None:
            return default
        
        return json.loads(row['value_json'])
    except Exception:
        return default


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def db_exists(project_name: str) -> bool:
    """Check if the characteristics database exists for a project."""
    return _get_db_path(project_name).exists()


def migrate_from_dataset_index(project_name: str) -> bool:
    """
    Migrate data from old dataset_index.db to new characteristics.db.
    Returns True if migration was performed.
    """
    old_path = _get_projects_root() / project_name / "dataset_index.db"
    new_path = _get_db_path(project_name)
    
    if not old_path.exists():
        return False
    
    if new_path.exists():
        # Already migrated, just rename old file
        backup = old_path.with_suffix('.db.legacy')
        try:
            old_path.rename(backup)
        except OSError:
            pass
        return False
    
    # Rename old to new
    try:
        old_path.rename(new_path)
        info(f"Migrated dataset_index.db to characteristics.db for {project_name}")
        return True
    except OSError as ex:
        warning(f"Failed to migrate database for {project_name}: {ex}")
        return False
