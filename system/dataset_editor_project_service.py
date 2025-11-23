from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from PIL import Image

from system.dataset_editor_metadata_store import EditorStore, FINGERPRINT_FIELD
from system.dataset_editor_models import (
    BoundsModel,
    BulkTagRequest,
    BulkTagResponse,
    DeleteItemsResponse,
    DiscoveryResponse,
    FolderNode,
    ImageItem,
    PaginatedItems,
    ProjectSummary,
    StatsSummary,
)
from system.dataset_editor_settings import get_dataset_editor_settings
from system.log import warning

if TYPE_CHECKING:
    from system.dataset_editor_discovery import DiscoveryJob

SETTINGS = get_dataset_editor_settings()

class ImageNotFoundError(ValueError):
    pass

class ProjectNotFoundError(ValueError):
    pass

class DatasetBuildError(ValueError):
    pass

TAG_SPLIT_PATTERN = re.compile(r"[,\s]+")

def parse_tags(annotation: str) -> List[str]:
    tokens = [token.strip().lower() for token in TAG_SPLIT_PATTERN.split(annotation)]
    return [token for token in tokens if token]

def _build_record_payload_task(args: tuple[str, str]) -> tuple[str, ImageRecord, dict, dict]:
    data_source_dir, relative_path_str = args
    data_source = Path(data_source_dir)
    relative_path = Path(relative_path_str)
    image_path = data_source / relative_path
    record, fingerprint, payload = ProjectIndex._build_record_entry(image_path, relative_path)
    return record.id, record, fingerprint, payload

@dataclass
class ImageRecord:
    id: str
    image_path: Path
    annotation_path: Path
    bounds_path: Path
    relative_dir: str
    relative_path: str
    filename: str
    extension: str
    category: str
    width: int
    height: int
    min_dimension: int
    annotation: str
    tags: List[str]
    bounds: BoundsModel
    has_custom_bounds: bool
    updated_at: Optional[datetime]

    def to_model(self, project_name: str) -> ImageItem:
        encoded_path = urllib.parse.quote(self.relative_path)
        image_url = f"/dataset/editor/projects/{project_name}/image?path={encoded_path}"
        return ImageItem(
            id=self.id,
            filename=self.filename,
            extension=self.extension,
            category=self.category,
            relative_dir=self.relative_dir,
            relative_path=self.relative_path,
            width=self.width,
            height=self.height,
            min_dimension=self.min_dimension,
            annotation=self.annotation,
            tags=self.tags,
            image_url=image_url,
            bounds=self.bounds,
            has_custom_bounds=self.has_custom_bounds,
            updated_at=self.updated_at,
        )

class ProjectIndex:
    def __init__(self, name: str, root: Path):
        self.name = name
        self.root = root
        self.data_source_dir = root / "data_source"
        self.dataset_dir = root / "dataset"
        self.items: Dict[str, ImageRecord] = {}
        self._lock = RLock()
        self._discovery_workers = max(1, SETTINGS.discovery_workers)
        self._loaded = False
        self._load_lock = RLock()
        self.editor_store = EditorStore(root)

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        snapshot = self.editor_store.load_snapshot()
        if snapshot:
            self._hydrate_from_snapshot(snapshot)
            return
        with self._load_lock:
            if self._loaded:
                return
            snapshot = self.editor_store.load_snapshot()
            if snapshot:
                self._hydrate_from_snapshot(snapshot)
                return
            self._perform_scan()

    def refresh(self) -> DiscoveryResponse:
        return self._perform_scan()

    def run_discovery(self, job: Optional["DiscoveryJob"] = None) -> DiscoveryResponse:
        return self._perform_scan(job)

    def _perform_scan(self, job: Optional["DiscoveryJob"] = None) -> DiscoveryResponse:
        if job:
            job.mark_running()

        start = time.perf_counter()
        previous_ids = self._previous_ids_snapshot()
        removed_ids = set(previous_ids)
        records: Dict[str, ImageRecord] = {}
        new_record_ids: set[str] = set()
        modified_record_ids: set[str] = set()
        editor_updates: Dict[str, dict] = {}

        files = self._collect_image_paths()
        total_candidates = len(files)
        processed = 0
        added = 0
        updated = 0
        skipped = 0

        if job:
            job.set_total(total_candidates)

        worker_count = min(total_candidates or 1, self._discovery_workers)

        def handle_result(
            record_id: str,
            record: ImageRecord,
            fingerprint: dict,
            payload: dict,
        ) -> None:
            nonlocal added, updated
            records[record_id] = record
            removed_ids.discard(record_id)
            is_new = record_id not in previous_ids
            needs_update = self.editor_store.should_update(record_id, fingerprint)
            if is_new:
                added += 1
                new_record_ids.add(record_id)
            elif needs_update:
                updated += 1
                modified_record_ids.add(record_id)
            if is_new or needs_update:
                editor_updates[record_id] = payload
            if job:
                job.set_current(record_id)
                job.advance(added=is_new, updated=(not is_new and needs_update))

        if worker_count <= 1:
            for image_path in files:
                processed += 1
                try:
                    relative_path = image_path.relative_to(self.data_source_dir)
                    record, fingerprint, payload = self._build_record_entry(image_path, relative_path)
                    record_id = record.id
                except Exception as exc:
                    warning(f"Dataset editor metadata read failed for {image_path}: {exc}")
                    skipped += 1
                    if job:
                        job.advance(skipped=True)
                    continue
                handle_result(record_id, record, fingerprint, payload)
        else:
            data_source_str = str(self.data_source_dir)
            tasks = [(data_source_str, str(path.relative_to(self.data_source_dir))) for path in files]
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=None) as executor:
                futures = {executor.submit(_build_record_payload_task, args): args for args in tasks}
                for future in as_completed(futures):
                    processed += 1
                    try:
                        record_id, record, fingerprint, payload = future.result()
                    except Exception as exc:
                        image_path = futures[future][1]
                        warning(f"Dataset editor metadata read failed for {image_path}: {exc}")
                        skipped += 1
                        if job:
                            job.advance(skipped=True)
                        continue
                    handle_result(record_id, record, fingerprint, payload)

        if editor_updates:
            self.editor_store.bulk_upsert(editor_updates)
            for record_id in new_record_ids:
                self.editor_store.mark_dataset_not_built(record_id)
            for record_id in modified_record_ids:
                self.editor_store.mark_dataset_needs_update(record_id)

        if removed_ids:
            self.editor_store.remove(removed_ids)
            for record_id in removed_ids:
                self._remove_dataset_outputs(record_id)

        removed_count = len(removed_ids)

        with self._lock:
            self.items = dict(sorted(records.items(), key=lambda item: item[0]))
            self._loaded = True
            snapshot = dict(self.items)

        self._write_cache(snapshot)
        duration = time.perf_counter() - start
        result = DiscoveryResponse(
            total_candidates=total_candidates,
            processed_items=processed,
            added_items=added,
            updated_items=updated,
            removed_items=removed_count,
            skipped_items=skipped,
            duration_seconds=duration,
            updated_at=datetime.now(timezone.utc),
        )

        if job:
            if removed_count:
                job.add_removed(removed_count)
            job.mark_success(result)

        return result

    @staticmethod
    def _load_bounds(bounds_path: Path) -> tuple[BoundsModel, bool]:
        if not bounds_path.exists():
            return BoundsModel(), False
        try:
            data = json.loads(bounds_path.read_text(encoding="utf-8"))
            return BoundsModel(**data), True
        except Exception:
            return BoundsModel(), False

    def _previous_ids_snapshot(self) -> set[str]:
        with self._lock:
            if self._loaded:
                return set(self.items.keys())
        return set(self.editor_store.keys())

    def _collect_image_paths(self) -> list[Path]:
        if not self.data_source_dir.exists():
            return []
        results: list[Path] = []
        for path in self.data_source_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SETTINGS.image_extensions:
                continue
            results.append(path)
        return sorted(results)

    @classmethod
    def _build_record_entry(cls, image_path: Path, relative_path: Path) -> tuple[ImageRecord, dict, dict]:
        with Image.open(image_path) as img:
            width, height = img.size

        annotation_path = image_path.with_suffix(".txt")
        annotation = annotation_path.read_text(encoding="utf-8") if annotation_path.exists() else ""
        tags = parse_tags(annotation)

        bounds_path = image_path.with_name(f"{image_path.stem}-bounds.json")
        bounds, has_custom = cls._load_bounds(bounds_path)

        relative_dir = str(relative_path.parent).replace("\\", "/")
        category = relative_dir.split("/")[0] if relative_dir else "root"
        record_id = str(relative_path).replace("\\", "/")
        updated_at = cls._annotation_timestamp(annotation_path)

        record = ImageRecord(
            id=record_id,
            image_path=image_path,
            annotation_path=annotation_path,
            bounds_path=bounds_path,
            relative_dir=relative_dir,
            relative_path=record_id,
            filename=image_path.name,
            extension=image_path.suffix.lower(),
            category=category,
            width=width,
            height=height,
            min_dimension=min(width, height),
            annotation=annotation,
            tags=tags,
            bounds=bounds,
            has_custom_bounds=has_custom,
            updated_at=updated_at,
        )

        fingerprint = cls._capture_fingerprint(image_path, annotation_path, bounds_path)
        payload = cls._build_editor_payload(record, fingerprint)
        return record, fingerprint, payload

    @staticmethod
    def _annotation_timestamp(annotation_path: Path) -> Optional[datetime]:
        if not annotation_path.exists():
            return None
        return datetime.fromtimestamp(annotation_path.stat().st_mtime, tz=timezone.utc)

    @staticmethod
    def _capture_fingerprint(image_path: Path, annotation_path: Path, bounds_path: Path) -> dict:
        image_stat = image_path.stat()
        annotation_stat = annotation_path.stat() if annotation_path.exists() else None
        bounds_stat = bounds_path.stat() if bounds_path.exists() else None
        return {
            "image_mtime": image_stat.st_mtime,
            "image_size": image_stat.st_size,
            "annotation_mtime": annotation_stat.st_mtime if annotation_stat else 0.0,
            "annotation_size": annotation_stat.st_size if annotation_stat else 0,
            "bounds_mtime": bounds_stat.st_mtime if bounds_stat else 0.0,
            "bounds_size": bounds_stat.st_size if bounds_stat else 0,
        }

    @staticmethod
    def _fingerprint_for_record(record: ImageRecord) -> dict:
        return ProjectIndex._capture_fingerprint(record.image_path, record.annotation_path, record.bounds_path)

    def _remove_dataset_outputs(self, record_id: str) -> None:
        if not record_id:
            return
        target_image = self.dataset_dir / Path(record_id)
        target_annotation = target_image.with_suffix(".txt")
        target_image.unlink(missing_ok=True)
        target_annotation.unlink(missing_ok=True)

    def list_items(
        self,
        page: int,
        page_size: int,
        search: Optional[str] = None,
        category: Optional[str] = None,
        folder: Optional[str] = None,
    ) -> PaginatedItems:
        self.ensure_loaded()
        records = self._filter_records(search=search, category=category, folder=folder)

        total = len(records)
        if total == 0:
            return PaginatedItems(items=[], page=page, page_size=page_size, total_items=0, total_pages=0)

        total_pages = math.ceil(total / page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        page_items = [record.to_model(self.name) for record in records[start:end]]

        return PaginatedItems(
            items=page_items,
            page=page,
            page_size=page_size,
            total_items=total,
            total_pages=total_pages,
        )

    def _filter_records(
        self,
        search: Optional[str] = None,
        category: Optional[str] = None,
        folder: Optional[str] = None,
    ) -> List[ImageRecord]:
        records = list(self.items.values())

        if search:
            query = search.lower()
            records = [record for record in records if query in record.annotation.lower() or query in record.filename.lower()]

        if category:
            records = [record for record in records if record.category == category]

        if folder is not None:
            folder_path = folder.replace("\\", "/").strip("/")
            if folder_path:
                folder_prefix = f"{folder_path}/"
                records = [
                    record
                    for record in records
                    if record.relative_dir == folder_path or record.relative_dir.startswith(folder_prefix)
                ]

        return records

    def get_record(self, item_id: str) -> ImageRecord:
        self.ensure_loaded()
        if item_id not in self.items:
            raise ImageNotFoundError(f"Image '{item_id}' not found in project {self.name}")
        return self.items[item_id]

    def update_annotation(self, item_id: str, content: str) -> ImageItem:
        record = self.get_record(item_id)
        record.annotation = content
        record.tags = parse_tags(content)
        record.annotation_path.write_text(content, encoding="utf-8")
        record.updated_at = datetime.now(timezone.utc)
        fingerprint = self._fingerprint_for_record(record)
        self.editor_store.upsert(record.id, self._build_editor_payload(record, fingerprint))
        self.editor_store.update_snapshot_entry(record.id, self._record_to_cache(record))
        self.editor_store.mark_dataset_needs_update(record.id)
        return record.to_model(self.name)

    def update_bounds(self, item_id: str, payload: BoundsModel) -> ImageItem:
        record = self.get_record(item_id)
        record.bounds = payload
        record.has_custom_bounds = True
        record.bounds_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        fingerprint = self._fingerprint_for_record(record)
        self.editor_store.upsert(record.id, self._build_editor_payload(record, fingerprint))
        self.editor_store.update_snapshot_entry(record.id, self._record_to_cache(record))
        self.editor_store.mark_dataset_needs_update(record.id)
        return record.to_model(self.name)

    def delete_items(self, item_ids: List[str]) -> DeleteItemsResponse:
        self.ensure_loaded()
        if not item_ids:
            return DeleteItemsResponse(deleted=0, missing=[])

        deleted_records: list[ImageRecord] = []
        missing: list[str] = []
        seen: set[str] = set()

        for raw_id in item_ids:
            normalized = raw_id.replace("\\", "/")
            if normalized in seen:
                continue
            seen.add(normalized)
            record = self.items.get(normalized)
            if not record:
                missing.append(normalized)
                continue
            deleted_records.append(record)

        if not deleted_records:
            return DeleteItemsResponse(deleted=0, missing=missing)

        for record in deleted_records:
            record.image_path.unlink(missing_ok=True)
            record.annotation_path.unlink(missing_ok=True)
            record.bounds_path.unlink(missing_ok=True)
            self._remove_dataset_outputs(record.id)

        removed_ids = {record.id for record in deleted_records}

        with self._lock:
            for record_id in removed_ids:
                self.items.pop(record_id, None)

        try:
            self.editor_store.remove(removed_ids)
        except Exception:
            pass
        self._write_cache_snapshot()

        return DeleteItemsResponse(deleted=len(removed_ids), missing=missing)

    def bulk_tags(self, request: BulkTagRequest) -> BulkTagResponse:
        self.ensure_loaded()
        current: List[ImageRecord]
        if request.items:
            deduped: dict[str, None] = {}
            current = []
            for item_id in request.items:
                normalized = item_id.replace("\\", "/")
                if normalized in deduped:
                    continue
                record = self.items.get(normalized)
                if record:
                    current.append(record)
                    deduped[normalized] = None
        else:
            current = self._filter_records(
                search=request.search,
                category=request.category,
                folder=request.folder,
            )

        add_tags = [tag.strip().lower() for tag in request.add if tag.strip()]
        remove_tags = [tag.strip().lower() for tag in request.remove if tag.strip()]

        if not current:
            return BulkTagResponse(
                updated=0,
                skipped=0,
                added_tags=add_tags,
                removed_tags=remove_tags,
                updated_at=datetime.now(timezone.utc),
            )

        updated = 0
        skipped = 0

        editor_updates: Dict[str, dict] = {}
        snapshot_updates: Dict[str, dict] = {}

        for record in current:
            original_tags = set(record.tags)
            new_tags = set(original_tags)
            new_tags.difference_update(remove_tags)
            new_tags.update(add_tags)

            if new_tags == original_tags:
                skipped += 1
                continue

            record.tags = sorted(new_tags)
            record.annotation = self._rewrite_annotation(record.annotation, record.tags)
            record.annotation_path.write_text(record.annotation, encoding="utf-8")
            record.updated_at = datetime.now(timezone.utc)
            updated += 1
            fingerprint = self._fingerprint_for_record(record)
            editor_updates[record.id] = self._build_editor_payload(record, fingerprint)
            snapshot_updates[record.id] = self._record_to_cache(record)

        if editor_updates:
            self.editor_store.bulk_upsert(editor_updates)
            for record_id, entry in snapshot_updates.items():
                self.editor_store.update_snapshot_entry(record_id, entry)
                self.editor_store.mark_dataset_needs_update(record_id)

        return BulkTagResponse(
            updated=updated,
            skipped=skipped,
            added_tags=add_tags,
            removed_tags=remove_tags,
            updated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _build_editor_payload(record: ImageRecord, fingerprint: Optional[dict] = None) -> dict:
        bounds = record.bounds or BoundsModel()
        zoom = bounds.zoom or 1.0
        size = round(1.0 / zoom, 6) if zoom else 1.0
        updated = record.updated_at or datetime.now(timezone.utc)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        else:
            updated = updated.astimezone(timezone.utc)
        payload = {
            "crop": {
                "center_x": bounds.center_x,
                "center_y": bounds.center_y,
                "size": size,
                "zoom": bounds.zoom,
            },
            "tags": list(record.tags),
            "dimensions": {"width": record.width, "height": record.height},
            "updated_at": updated.isoformat(),
        }
        payload[FINGERPRINT_FIELD] = fingerprint or ProjectIndex._fingerprint_for_record(record)
        return payload

    @staticmethod
    def _annotation_separator(annotation: str | None) -> str:
        if not annotation:
            return ", "
        if "\r\n" in annotation:
            return "\r\n"
        if "\n" in annotation:
            return "\n"
        if ", " in annotation:
            return ", "
        if "," in annotation:
            return ","
        if "; " in annotation:
            return "; "
        if ";" in annotation:
            return ";"
        if "|" in annotation:
            return "|"
        if " " in annotation:
            return " "
        return ", "

    @classmethod
    def _rewrite_annotation(cls, annotation: str, tags: List[str]) -> str:
        if not tags:
            return ""
        separator = cls._annotation_separator(annotation)
        return separator.join(tags)

    def _snapshot_items(self) -> Dict[str, ImageRecord]:
        with self._lock:
            return dict(self.items)

    def _hydrate_from_snapshot(self, snapshot: Dict[str, dict]) -> None:
        records: Dict[str, ImageRecord] = {}
        for record_id, payload in snapshot.items():
            entry = dict(payload or {})
            entry.setdefault("id", record_id)
            try:
                record = self._record_from_cache(entry)
            except Exception:
                continue
            records[record.id] = record

        with self._lock:
            self.items = dict(sorted(records.items(), key=lambda item: item[0]))
            self._loaded = True

    def _write_cache_snapshot(self) -> None:
        try:
            snapshot = self._snapshot_items()
        except Exception:
            return
        self._write_cache(snapshot)

    def _write_cache(self, records: Dict[str, ImageRecord]) -> None:
        try:
            snapshot_payload = {
                record_id: self._record_to_cache(record)
                for record_id, record in sorted(records.items(), key=lambda item: item[0])
            }
            self.editor_store.write_snapshot(snapshot_payload)
        except Exception as exc:
            warning(f"Dataset editor snapshot persistence failed for {self.name}: {exc}")

    def cached_image_count(self) -> int:
        if self._loaded:
            return len(self.items)
        cache_count = self._cached_count_from_disk()
        if cache_count is not None:
            return cache_count
        return self._editor_store_count()

    def _cached_count_from_disk(self) -> int | None:
        try:
            count = self.editor_store.snapshot_count()
        except Exception:
            return None
        return count if count else None

    def _editor_store_count(self) -> int:
        try:
            return self.editor_store.count()
        except Exception:
            return 0

    def _record_to_cache(self, record: ImageRecord) -> dict:
        return {
            "id": record.id,
            "image_path": str(record.image_path),
            "annotation_path": str(record.annotation_path),
            "bounds_path": str(record.bounds_path),
            "relative_dir": record.relative_dir,
            "relative_path": record.relative_path,
            "filename": record.filename,
            "extension": record.extension,
            "category": record.category,
            "width": record.width,
            "height": record.height,
            "min_dimension": record.min_dimension,
            "annotation": record.annotation,
            "tags": record.tags,
            "bounds": record.bounds.model_dump(),
            "has_custom_bounds": record.has_custom_bounds,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        }

    def _record_from_cache(self, data: dict) -> ImageRecord:
        updated_at = data.get("updated_at")
        if updated_at:
            try:
                updated_at = datetime.fromisoformat(updated_at)
            except Exception:
                updated_at = None
        else:
            updated_at = None

        bounds_payload = data.get("bounds") or {}
        bounds = BoundsModel(**bounds_payload)

        relative_path = data.get("relative_path", data["id"])
        image_path_value = data.get("image_path")
        image_path = Path(image_path_value) if image_path_value else (self.data_source_dir / relative_path)
        annotation_path_value = data.get("annotation_path")
        annotation_path = Path(annotation_path_value) if annotation_path_value else image_path.with_suffix(".txt")
        bounds_path_value = data.get("bounds_path")
        bounds_path = Path(bounds_path_value) if bounds_path_value else image_path.with_name(f"{image_path.stem}-bounds.json")

        width = int(data.get("width") or 0)
        height = int(data.get("height") or 0)
        default_min = min(width, height) if width and height else max(width, height, 0)
        min_value = data.get("min_dimension")
        min_dimension = int(min_value) if min_value is not None else default_min

        return ImageRecord(
            id=data["id"],
            image_path=image_path,
            annotation_path=annotation_path,
            bounds_path=bounds_path,
            relative_dir=data.get("relative_dir", ""),
            relative_path=relative_path,
            filename=data.get("filename", image_path.name),
            extension=data.get("extension", image_path.suffix),
            category=data.get("category", "root"),
            width=width,
            height=height,
            min_dimension=min_dimension,
            annotation=data.get("annotation", ""),
            tags=list(data.get("tags", [])),
            bounds=bounds,
            has_custom_bounds=bool(data.get("has_custom_bounds", False)),
            updated_at=updated_at,
        )

    def stats(self) -> StatsSummary:
        self.ensure_loaded()
        categories = Counter(record.category for record in self.items.values())
        tag_counter = Counter(tag for record in self.items.values() for tag in record.tags)

        least_common: List[str] = []
        if tag_counter:
            min_count = min(tag_counter.values())
            least_common = [tag for tag, count in tag_counter.items() if count == min_count]
            least_common = sorted(least_common)[:5]

        most_common = [tag for tag, _ in tag_counter.most_common(5)]

        recommendations: List[Dict[str, Any]] = []
        total_images = len(self.items)
        

        if total_images == 0:
            recommendations.append({
                "type": "dataset_empty",
                "severity": "critical",
                "data": {}
            })
        elif total_images < 50:
            recommendations.append({
                "type": "dataset_very_small",
                "severity": "critical",
                "data": {
                    "current_total": total_images,
                    "recommended_min": 100,
                    "recommended_ideal": 200
                }
            })
        elif total_images < 200:
            recommendations.append({
                "type": "dataset_small",
                "severity": "warning",
                "data": {
                    "current_total": total_images,
                    "recommended_min": 200
                }
            })
        

        if len(categories) > 1:
            cat_counts = list(categories.values())
            min_count = min(cat_counts)
            max_count = max(cat_counts)
            avg_count = sum(cat_counts) / len(cat_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else 0
            

            sorted_cats = sorted(categories.items(), key=lambda x: x[1])
            underrepresented = [{"label": c, "count": n} for c, n in sorted_cats if n < avg_count * 0.5]
            overrepresented = [{"label": c, "count": n} for c, n in sorted_cats if n > avg_count * 1.5]
            
            if imbalance_ratio > 5:
                recommendations.append({
                    "type": "severe_imbalance",
                    "severity": "critical",
                    "data": {
                        "ratio": round(imbalance_ratio, 1),
                        "min_count": min_count,
                        "max_count": max_count,
                        "avg_count": round(avg_count, 1),
                        "underrepresented": underrepresented[:3],
                        "overrepresented": overrepresented[-3:],
                        "total_categories": len(categories)
                    }
                })
            elif imbalance_ratio > 2:
                recommendations.append({
                    "type": "moderate_imbalance",
                    "severity": "warning",
                    "data": {
                        "ratio": round(imbalance_ratio, 1),
                        "min_count": min_count,
                        "max_count": max_count,
                        "avg_count": round(avg_count, 1),
                        "underrepresented": underrepresented[:2],
                        "overrepresented": overrepresented[-2:],
                        "total_categories": len(categories)
                    }
                })
        

        if tag_counter and len(tag_counter) > 1:
            tag_counts = list(tag_counter.values())
            min_tag = min(tag_counts)
            max_tag = max(tag_counts)
            avg_tag = sum(tag_counts) / len(tag_counts)
            tag_ratio = max_tag / min_tag if min_tag > 0 else 0
            
            if tag_ratio > 5:
                sorted_tags = sorted(tag_counter.items(), key=lambda x: x[1])
                rare_tags = [{"tag": t, "count": c} for t, c in sorted_tags[:3]]
                recommendations.append({
                    "type": "severe_tag_imbalance",
                    "severity": "critical",
                    "data": {
                        "ratio": round(tag_ratio, 1),
                        "rare_tags": rare_tags,
                        "total_tags": len(tag_counter)
                    }
                })

        balance_score = 1.0
        balance_status = "Excellent"
        balance_analysis = {}
        
        if len(categories) > 1 and len(self.items) > 0:

            counts = list(categories.values())
            min_count = min(counts)
            max_count = max(counts)
            avg_count = sum(counts) / len(counts)
            

            ratio = min_count / max_count if max_count > 0 else 1.0
            balance_score = (ratio ** 0.5) * 0.5 + 0.5
            

            if balance_score >= 0.9:
                balance_status = "Excellent"
            elif balance_score >= 0.7:
                balance_status = "Good"
            elif balance_score >= 0.5:
                balance_status = "Fair"
            elif balance_score >= 0.3:
                balance_status = "Poor"
            else:
                balance_status = "Critical"
            

            balance_analysis = {
                "ideal_per_label": round(avg_count, 1),
                "min_images": min_count,
                "max_images": max_count,
                "ratio_max_to_min": round(max_count / min_count if min_count > 0 else 0, 2),
                "categories_count": len(categories),
            }

        return StatsSummary(
            total_images=len(self.items),
            categories=dict(categories),
            tag_frequencies=dict(tag_counter.most_common(50)),
            least_common_tags=least_common,
            most_common_tags=most_common,
            recommendations=recommendations,

            label_distribution=dict(categories),
            balance_score=round(balance_score, 3),
            balance_status=balance_status,
            balance_analysis=balance_analysis,
        )

    def folder_tree(self) -> FolderNode:
        self.ensure_loaded()

        nodes: dict[str, FolderNode] = {}

        def ensure_node(path: str) -> FolderNode:
            if path in nodes:
                return nodes[path]
            if path == "":
                node = FolderNode(name="All images", path="", image_count=0, children=[])
                nodes[path] = node
                return node

            parent_path, _, name = path.rpartition("/")
            parent = ensure_node(parent_path)
            node = FolderNode(name=name or path, path=path, image_count=0, children=[])
            nodes[path] = node
            parent.children.append(node)
            return node

        counts: dict[str, int] = defaultdict(int)

        ensure_node("")

        for record in self.items.values():
            relative_dir = record.relative_dir
            ensure_node(relative_dir)
            parts = relative_dir.split("/") if relative_dir else []
            for depth in range(len(parts), -1, -1):
                sub_path = "/".join(parts[:depth])
                counts[sub_path] += 1

        counts.setdefault("", len(self.items))

        for path, node in nodes.items():
            node.image_count = counts.get(path, 0)

        def sort_children(node: FolderNode) -> None:
            node.children.sort(key=lambda child: child.name.lower())
            for child in node.children:
                sort_children(child)

        root = ensure_node("")
        sort_children(root)
        return root

class ProjectRepository:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._indexes: Dict[str, ProjectIndex] = {}
        self._lock = RLock()

    def list_projects(self) -> List[ProjectSummary]:
        summaries: List[ProjectSummary] = []
        if not self.base_dir.exists():
            return summaries
        for entry in sorted(self.base_dir.iterdir()):
            if not entry.is_dir():
                continue
            index = self._indexes.get(entry.name)
            if not index:
                index = ProjectIndex(entry.name, entry)
                self._indexes[entry.name] = index
            try:
                image_count = index.cached_image_count()
            except Exception:
                image_count = 0
            summaries.append(
                ProjectSummary(
                    name=entry.name,
                    image_count=image_count,
                    data_source_dir=str(index.data_source_dir),
                    dataset_dir=str(index.dataset_dir),
                )
            )
        return summaries

    def get_index(self, project_name: str) -> ProjectIndex:
        project_path = self.base_dir / project_name
        if not project_path.exists():
            raise ProjectNotFoundError(f"Project '{project_name}' not found")
        with self._lock:
            index = self._indexes.get(project_name)
            if not index:
                index = ProjectIndex(project_name, project_path)
                self._indexes[project_name] = index
        return index

repository = ProjectRepository(SETTINGS.projects_root)
