from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from PIL import Image

from system.dataset_editor_metadata_store import EditorStore
from system.dataset_editor_models import (
    BoundsModel,
    BulkTagRequest,
    BulkTagResponse,
    DeleteItemsResponse,
    DiscoveryResponse,
    DuplicateGroup,
    DuplicateScanResponse,
    FolderNode,
    ImageItem,
    PaginatedItems,
    ProjectSummary,
    StatsSummary,
)
from system.dataset_editor_settings import get_dataset_editor_settings
from system.duplicate_detection import scan_for_duplicates
from system.log import warning, info
from system.dataset_discovery import calculate_balance_score, classify_balance_status
from system import project_db

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

def _build_record_payload_task(args: tuple[str, str]) -> tuple[str, ImageRecord, dict]:
    data_source_dir, relative_path_str = args
    data_source = Path(data_source_dir)
    relative_path = Path(relative_path_str)
    image_path = data_source / relative_path
    record, payload = ProjectIndex._build_record_entry(image_path, relative_path)
    return record.id, record, payload

@dataclass
class ImageRecord:
    id: str
    image_path: Path
    annotation_path: Path
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
        """
        Load items from cache/snapshot if available.
        Does NOT run automatic discovery - user must explicitly sync.
        """
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
            # No snapshot - stay empty until user triggers sync
            self._loaded = True

    def sync(self, job: Optional["DiscoveryJob"] = None) -> DiscoveryResponse:
        """
        Sync/discover images from data_source folder.
        This is the single entry point for scanning - called explicitly by user.
        """
        return self._perform_scan(job)

    # Keep for backward compatibility
    def refresh(self) -> DiscoveryResponse:
        return self.sync()

    def run_discovery(self, job: Optional["DiscoveryJob"] = None) -> DiscoveryResponse:
        return self.sync(job)

    def _perform_scan(self, job: Optional["DiscoveryJob"] = None) -> DiscoveryResponse:
        if job:
            job.mark_running()

        start = time.perf_counter()
        previous_ids = self._previous_ids_snapshot()
        removed_ids = set(previous_ids)
        records: Dict[str, ImageRecord] = {}
        new_record_ids: set[str] = set()
        editor_updates: Dict[str, dict] = {}

        files = self._collect_image_paths()
        total_candidates = len(files)
        
        # Collect files to process - new files only (not in previous_ids)
        # Existing files are reused from memory/cache
        files_to_process = []
        unchanged_count = 0
        
        for image_path in files:
            relative_path = image_path.relative_to(self.data_source_dir)
            record_id = str(relative_path).replace("\\", "/")
            removed_ids.discard(record_id)
            
            if record_id in previous_ids and record_id in self.items:
                # Unchanged - reuse existing record from memory
                unchanged_count += 1
                records[record_id] = self.items[record_id]
            else:
                # New file - needs processing
                files_to_process.append((image_path, relative_path, record_id))
        
        processed = 0
        added = 0
        skipped = 0
        
        if job:
            job.set_total(len(files_to_process))

        worker_count = min(len(files_to_process) or 1, self._discovery_workers)

        def handle_result(
            record_id: str,
            record: ImageRecord,
            payload: dict,
        ) -> None:
            nonlocal added
            records[record_id] = record
            added += 1
            new_record_ids.add(record_id)
            editor_updates[record_id] = payload
            if job:
                job.set_current(record_id)
                job.advance(added=True)

        if worker_count <= 1 or len(files_to_process) <= 1:
            for image_path, relative_path, record_id in files_to_process:
                processed += 1
                try:
                    record, payload = self._build_record_entry(image_path, relative_path)
                except Exception as exc:
                    warning(f"Dataset editor metadata read failed for {image_path}: {exc}")
                    skipped += 1
                    if job:
                        job.advance(skipped=True)
                    continue
                handle_result(record_id, record, payload)
        else:
            data_source_str = str(self.data_source_dir)
            tasks = [(data_source_str, str(rp)) for _, rp, _ in files_to_process]
            
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=None) as executor:
                futures = {executor.submit(_build_record_payload_task, args): args for args in tasks}
                for future in as_completed(futures):
                    processed += 1
                    try:
                        record_id, record, payload = future.result()
                    except Exception as exc:
                        image_path = futures[future][1]
                        warning(f"Dataset editor metadata read failed for {image_path}: {exc}")
                        skipped += 1
                        if job:
                            job.advance(skipped=True)
                        continue
                    handle_result(record_id, record, payload)

        if editor_updates:
            self.editor_store.bulk_upsert(editor_updates)

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
            removed_items=removed_count,
            skipped_items=skipped,
            duration_seconds=duration,
        )

        # Save computed stats to project.db after discovery
        self.save_computed_stats()

        if job:
            if removed_count:
                job.add_removed(removed_count)
            job.mark_success(result)

        return result

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
    def _build_record_entry(cls, image_path: Path, relative_path: Path, bounds: Optional[BoundsModel] = None, has_custom_bounds: bool = False) -> tuple[ImageRecord, dict]:
        # Image.open only reads header for size - fast
        with Image.open(image_path) as img:
            width, height = img.size

        annotation_path = image_path.with_suffix(".txt")
        annotation = annotation_path.read_text(encoding="utf-8") if annotation_path.exists() else ""
        tags = parse_tags(annotation)

        # Use provided bounds or create default
        if bounds is None:
            bounds = BoundsModel(center_x=0.5, center_y=0.5, zoom=1.0)
            has_custom_bounds = False

        relative_dir = str(relative_path.parent).replace("\\", "/")
        category = relative_dir.split("/")[0] if relative_dir else "root"
        record_id = str(relative_path).replace("\\", "/")

        record = ImageRecord(
            id=record_id,
            image_path=image_path,
            annotation_path=annotation_path,
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
            has_custom_bounds=has_custom_bounds,
        )

        payload = cls._build_editor_payload(record)
        return record, payload

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
        self.editor_store.upsert(record.id, self._build_editor_payload(record))
        self.editor_store.update_snapshot_entry(record.id, self._record_to_cache(record))
        return record.to_model(self.name)

    def update_bounds(self, item_id: str, payload: BoundsModel) -> ImageItem:
        record = self.get_record(item_id)
        record.bounds = payload
        record.has_custom_bounds = True
        # Save to database via editor_store (snapshot_upsert includes crop data)
        self.editor_store.upsert(record.id, self._build_editor_payload(record))
        self.editor_store.update_snapshot_entry(record.id, self._record_to_cache(record))
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
            # Bounds are stored in DB, not files - no file deletion needed
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
            updated += 1
            editor_updates[record.id] = self._build_editor_payload(record)
            snapshot_updates[record.id] = self._record_to_cache(record)

        if editor_updates:
            self.editor_store.bulk_upsert(editor_updates)
            for record_id, entry in snapshot_updates.items():
                self.editor_store.update_snapshot_entry(record_id, entry)

        return BulkTagResponse(
            updated=updated,
            skipped=skipped,
            added_tags=add_tags,
            removed_tags=remove_tags,
        )

    def scan_duplicates(self) -> DuplicateScanResponse:
        """
        Scan all images in the project for duplicates based on content hash.
        Returns groups of duplicate images.
        """
        self.ensure_loaded()
        start_time = time.time()
        
        # Collect all image records
        records = [
            {"relative_path": record.relative_path, "record": record}
            for record in self.items.values()
        ]
        
        if not records:
            return DuplicateScanResponse(
                total_groups=0,
                total_duplicates=0,
                groups=[],
                scanned_images=0,
                duration_seconds=0.0,
            )
        
        # Scan for duplicates
        hash_map, duplicate_groups = scan_for_duplicates(
            [{"relative_path": r["relative_path"]} for r in records],
            self.data_source_dir,
            max_workers=4,
        )
        
        # Build response groups with full ImageItem data
        groups: List[DuplicateGroup] = []
        total_duplicates = 0
        
        for content_hash, paths in duplicate_groups.items():
            images: List[ImageItem] = []
            for path in paths:
                normalized = path.replace("\\", "/")
                record = self.items.get(normalized)
                if record:
                    images.append(record.to_model(self.name))
            
            if len(images) > 1:
                groups.append(DuplicateGroup(
                    content_hash=content_hash,
                    images=images,
                ))
                total_duplicates += len(images)
        
        duration = time.time() - start_time
        
        return DuplicateScanResponse(
            total_groups=len(groups),
            total_duplicates=total_duplicates,
            groups=groups,
            scanned_images=len(records),
            duration_seconds=round(duration, 2),
        )

    @staticmethod
    def _build_editor_payload(record: ImageRecord) -> dict:
        bounds = record.bounds or BoundsModel()
        zoom = bounds.zoom or 1.0
        size = round(1.0 / zoom, 6) if zoom else 1.0
        return {
            "crop": {
                "center_x": bounds.center_x,
                "center_y": bounds.center_y,
                "size": size,
                "zoom": bounds.zoom,
            },
            "tags": list(record.tags),
            "dimensions": {"width": record.width, "height": record.height},
        }

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
        }

    def _record_from_cache(self, data: dict) -> ImageRecord:
        bounds_payload = data.get("bounds") or {}
        bounds = BoundsModel(**bounds_payload)

        relative_path = data.get("relative_path", data["id"])
        image_path_value = data.get("image_path")
        image_path = Path(image_path_value) if image_path_value else (self.data_source_dir / relative_path)
        annotation_path_value = data.get("annotation_path")
        annotation_path = Path(annotation_path_value) if annotation_path_value else image_path.with_suffix(".txt")

        width = int(data.get("width") or 0)
        height = int(data.get("height") or 0)
        default_min = min(width, height) if width and height else max(width, height, 0)
        min_value = data.get("min_dimension")
        min_dimension = int(min_value) if min_value is not None else default_min

        return ImageRecord(
            id=data["id"],
            image_path=image_path,
            annotation_path=annotation_path,
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
        
        # Use tags for balance calculation if available (multi-label), otherwise use categories (folder classification)
        if tag_counter and len(tag_counter) > 1:
            # Multi-label dataset - use tag distribution
            distribution = dict(tag_counter)
            balance_score = calculate_balance_score(distribution)
            balance_status = classify_balance_status(balance_score)
            
            counts = list(distribution.values())
            balance_analysis = {
                "ideal_per_label": round(sum(counts) / len(counts), 1),
                "min_images": min(counts),
                "max_images": max(counts),
                "ratio_max_to_min": round(max(counts) / min(counts) if min(counts) > 0 else 0, 2),
                "categories_count": len(distribution),
            }
        elif len(categories) > 1 and len(self.items) > 0:
            # Folder classification - use category distribution
            distribution = dict(categories)
            balance_score = calculate_balance_score(distribution)
            balance_status = classify_balance_status(balance_score)
            
            counts = list(distribution.values())
            balance_analysis = {
                "ideal_per_label": round(sum(counts) / len(counts), 1),
                "min_images": min(counts),
                "max_images": max(counts),
                "ratio_max_to_min": round(max(counts) / min(counts) if min(counts) > 0 else 0, 2),
                "categories_count": len(distribution),
            }

        # Use tags for label_distribution if available, otherwise fall back to categories
        if tag_counter:
            label_dist = dict(tag_counter)
        else:
            label_dist = dict(categories)

        return StatsSummary(
            total_images=len(self.items),
            categories=dict(categories),
            tag_frequencies=dict(tag_counter.most_common(50)),
            least_common_tags=least_common,
            most_common_tags=most_common,
            recommendations=recommendations,

            label_distribution=label_dist,
            balance_score=round(balance_score, 3),
            balance_status=balance_status,
            balance_analysis=balance_analysis,
        )

    def save_computed_stats(self) -> bool:
        """
        Compute stats and save them to project.db for caching.
        Called after discovery/sync or on explicit refresh.
        """
        try:
            stats_model = self.stats()
            
            label_dist = stats_model.label_distribution or {}
            balance_anal = stats_model.balance_analysis or {}
            
            # Build stats dict for project_db.save_stats()
            stats_dict = {
                "dataset_type": "multi_label",  # or detect from project config
                "image_count": stats_model.total_images,
                "labels": list(label_dist.keys()),
                "label_distribution": label_dist,
                "balance_score": stats_model.balance_score,
                "balance_status": stats_model.balance_status,
                "balance_analysis": {
                    "total_images": stats_model.total_images,
                    "num_labels": len(label_dist),
                    **balance_anal,
                },
                "has_dataset": self.dataset_dir.exists() and any(self.dataset_dir.iterdir()) if self.dataset_dir.exists() else False,
                "has_model_dir": (self.root / "model").exists(),
            }
            
            # Save to project.db
            project_db.save_stats(self.name, stats_dict)
            
            info(f"Saved computed stats for project {self.name}: {stats_model.total_images} images, {len(label_dist)} labels")
            return True
        except Exception as ex:
            warning(f"Failed to save computed stats for {self.name}: {ex}")
            return False

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

        # First, scan filesystem for all directories (including empty ones)
        if self.data_source_dir.exists():
            for dir_path in self.data_source_dir.rglob("*"):
                if dir_path.is_dir():
                    try:
                        relative = dir_path.relative_to(self.data_source_dir)
                        relative_str = str(relative).replace("\\", "/")
                        ensure_node(relative_str)
                    except ValueError:
                        pass

        # Then count images per folder from loaded items
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

    def _validate_path_within_data_source(self, relative_path: str) -> Path:
        """Validate that a path is within data_source and return absolute path."""
        normalized = relative_path.replace("\\", "/").strip("/")
        if not normalized:
            raise ValueError("Path cannot be empty")
        
        # Check for path traversal attempts
        if ".." in normalized or normalized.startswith("/"):
            raise ValueError("Invalid path: path traversal not allowed")
        
        full_path = self.data_source_dir / normalized
        
        # Resolve and verify it's still within data_source
        try:
            resolved = full_path.resolve()
            data_source_resolved = self.data_source_dir.resolve()
            if not str(resolved).startswith(str(data_source_resolved)):
                raise ValueError("Path must be within data_source directory")
        except Exception:
            raise ValueError("Invalid path")
        
        return full_path

    def create_folder(self, relative_path: str) -> Path:
        """Create a new folder within data_source."""
        full_path = self._validate_path_within_data_source(relative_path)
        
        if full_path.exists():
            raise ValueError(f"Folder already exists: {relative_path}")
        
        full_path.mkdir(parents=True, exist_ok=False)
        return full_path

    def rename_folder(self, old_relative_path: str, new_name: str) -> Path:
        """Rename a folder within data_source."""
        if not new_name or "/" in new_name or "\\" in new_name:
            raise ValueError("Invalid folder name")
        
        old_path = self._validate_path_within_data_source(old_relative_path)
        
        if not old_path.exists():
            raise ValueError(f"Folder not found: {old_relative_path}")
        
        if not old_path.is_dir():
            raise ValueError(f"Not a folder: {old_relative_path}")
        
        new_path = old_path.parent / new_name
        
        if new_path.exists():
            raise ValueError(f"A folder with name '{new_name}' already exists")
        
        # Rename the folder
        old_path.rename(new_path)
        
        # Update internal records
        old_prefix = old_relative_path.replace("\\", "/").rstrip("/")
        new_prefix = str(new_path.relative_to(self.data_source_dir)).replace("\\", "/")
        
        with self._lock:
            updated_items: Dict[str, ImageRecord] = {}
            for record_id, record in list(self.items.items()):
                if record.relative_dir == old_prefix or record.relative_dir.startswith(old_prefix + "/"):
                    # Update record paths
                    new_relative_dir = new_prefix + record.relative_dir[len(old_prefix):]
                    new_relative_path = new_relative_dir + "/" + record.filename if new_relative_dir else record.filename
                    new_record_id = new_relative_path
                    
                    # Create updated record
                    new_image_path = self.data_source_dir / new_relative_path
                    updated_record = ImageRecord(
                        id=new_record_id,
                        image_path=new_image_path,
                        annotation_path=new_image_path.with_suffix(".txt"),
                        relative_dir=new_relative_dir,
                        relative_path=new_relative_path,
                        filename=record.filename,
                        extension=record.extension,
                        category=new_relative_dir.split("/")[0] if new_relative_dir else "root",
                        width=record.width,
                        height=record.height,
                        min_dimension=record.min_dimension,
                        annotation=record.annotation,
                        tags=record.tags,
                        bounds=record.bounds,
                        has_custom_bounds=record.has_custom_bounds,
                    )
                    updated_items[new_record_id] = updated_record
                    del self.items[record_id]
            
            self.items.update(updated_items)
        
        # Refresh to update editor store
        self._write_cache_snapshot()
        
        return new_path

    def delete_folder(self, relative_path: str, recursive: bool = False) -> int:
        """Delete a folder within data_source. Returns number of deleted images."""
        full_path = self._validate_path_within_data_source(relative_path)
        
        if not full_path.exists():
            raise ValueError(f"Folder not found: {relative_path}")
        
        if not full_path.is_dir():
            raise ValueError(f"Not a folder: {relative_path}")
        
        # Check if folder is empty or recursive delete is allowed
        contents = list(full_path.iterdir())
        if contents and not recursive:
            raise ValueError("Folder is not empty. Use recursive=true to delete non-empty folders.")
        
        # Collect all images to delete
        normalized_path = relative_path.replace("\\", "/").rstrip("/")
        images_to_delete = []
        
        with self._lock:
            for record_id, record in self.items.items():
                if record.relative_dir == normalized_path or record.relative_dir.startswith(normalized_path + "/"):
                    images_to_delete.append(record_id)
        
        # Delete images through existing method
        deleted_count = 0
        if images_to_delete:
            result = self.delete_items(images_to_delete)
            deleted_count = result.deleted
        
        # Delete the folder itself (and any remaining non-image files)
        import shutil
        if full_path.exists():
            shutil.rmtree(full_path)
        
        return deleted_count

    def upload_images(self, files: List[tuple], target_folder: str = "") -> dict:
        """
        Upload images to data_source.
        files: List of (filename, file_content_bytes) tuples
        target_folder: Relative path within data_source (empty string for root)
        Returns: { uploaded: int, failed: int, files: [], errors: [] }
        """
        result = {
            "uploaded": 0,
            "failed": 0,
            "files": [],
            "errors": []
        }
        
        # Validate target folder
        if target_folder:
            target_path = self._validate_path_within_data_source(target_folder)
            if not target_path.exists():
                target_path.mkdir(parents=True, exist_ok=True)
        else:
            target_path = self.data_source_dir
        
        allowed_extensions = SETTINGS.image_extensions
        
        for filename, content in files:
            try:
                # Sanitize filename
                safe_filename = Path(filename).name
                if not safe_filename:
                    result["errors"].append(f"Invalid filename: {filename}")
                    result["failed"] += 1
                    continue
                
                # Check extension
                ext = Path(safe_filename).suffix.lower()
                if ext not in allowed_extensions:
                    result["errors"].append(f"Invalid file type: {safe_filename} (allowed: {', '.join(allowed_extensions)})")
                    result["failed"] += 1
                    continue
                
                # Write file
                file_path = target_path / safe_filename
                
                # Handle duplicate filenames
                counter = 1
                original_stem = file_path.stem
                while file_path.exists():
                    file_path = target_path / f"{original_stem}_{counter}{ext}"
                    counter += 1
                
                file_path.write_bytes(content)
                
                relative_path = str(file_path.relative_to(self.data_source_dir)).replace("\\", "/")
                result["files"].append(relative_path)
                result["uploaded"] += 1
                
            except Exception as e:
                result["errors"].append(f"Failed to upload {filename}: {str(e)}")
                result["failed"] += 1
        
        return result

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
