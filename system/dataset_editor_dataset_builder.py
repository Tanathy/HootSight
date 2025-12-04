from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock, Thread
from typing import Dict, Iterable, Literal, Tuple

from PIL import Image

from system.dataset_editor_models import BoundsModel, BuildDatasetResponse, BuildStatus
from system.dataset_editor_project_service import ImageRecord, ProjectIndex
from system.dataset_editor_settings import get_dataset_editor_settings
from system.log import warning

SETTINGS = get_dataset_editor_settings()

# PIL version compatibility: newer versions use Image.Resampling.LANCZOS
if hasattr(Image, "Resampling"):
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:
    RESAMPLE_LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

def compute_crop_box(width: int, height: int, bounds: BoundsModel) -> Tuple[int, int, int, int]:
    zoom = max(bounds.zoom, 1.0)
    min_dimension = min(width, height)
    crop_size = max(1, int(round(min_dimension / zoom)))
    crop_size = min(crop_size, width, height)

    center_x = bounds.center_x * width
    center_y = bounds.center_y * height
    half = crop_size / 2

    left = int(round(center_x - half))
    top = int(round(center_y - half))
    left = max(0, min(left, width - crop_size))
    top = max(0, min(top, height - crop_size))

    right = left + crop_size
    bottom = top + crop_size
    return left, top, right, bottom

class BuildCache:
    """Build cache using project.db (SQLite) instead of shelve files."""
    def __init__(self, project: "ProjectIndex"):
        self._project_name = project.root.name

    def snapshot(self) -> Dict[str, dict]:
        from system import project_db as cdb
        return cdb.build_cache_load(self._project_name)

    def commit(self, updates: Dict[str, dict], deletions: Iterable[str]) -> None:
        if not updates and not deletions:
            return
        from system import project_db as cdb
        cdb.build_cache_update(self._project_name, updates, list(deletions))

class BuildJob:
    def __init__(self, project_name: str, target_size: int):
        self.project_name = project_name
        self.status: Literal["idle", "pending", "running", "success", "error"] = "idle"
        self.total_items = 0
        self.completed_items = 0
        self.built_images = 0
        self.skipped_images = 0
        self.deleted_images = 0
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.duration_seconds: float = 0.0
        self.last_error: str | None = None
        self.current_item: str | None = None
        self.result: BuildDatasetResponse | None = None
        self._perf_start: float | None = None
        self._lock = RLock()
        self.target_size = target_size

    def is_running(self) -> bool:
        with self._lock:
            return self.status == "running"

    def mark_running(self) -> None:
        with self._lock:
            self.status = "running"
            self.started_at = datetime.now(timezone.utc)
            self.finished_at = None
            self.completed_items = 0
            self.built_images = 0
            self.skipped_images = 0
            self.deleted_images = 0
            self.duration_seconds = 0.0
            self.last_error = None
            self.current_item = None
            self._perf_start = time.perf_counter()

    def mark_pending(self, target_size: int | None = None) -> None:
        with self._lock:
            self.status = "pending"
            self.started_at = None
            self.finished_at = None
            self.total_items = 0
            self.completed_items = 0
            self.built_images = 0
            self.skipped_images = 0
            self.deleted_images = 0
            self.duration_seconds = 0.0
            self.last_error = None
            self.current_item = None
            self.result = None
            self._perf_start = None
            if target_size is not None:
                self.target_size = target_size

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total_items = total

    def set_current(self, item_id: str | None) -> None:
        with self._lock:
            self.current_item = item_id

    def advance(self, *, success: bool, deleted: bool = False) -> None:
        with self._lock:
            self.completed_items += 1
            if deleted:
                self.deleted_images += 1
            elif success:
                self.built_images += 1
            else:
                self.skipped_images += 1
            self.current_item = None

    def mark_success(self, result: BuildDatasetResponse) -> None:
        with self._lock:
            self.status = "success"
            self.result = result
            self.finished_at = datetime.now(timezone.utc)
            self.current_item = None
            if self._perf_start is not None:
                self.duration_seconds = time.perf_counter() - self._perf_start

    def mark_failed(self, error: str) -> None:
        with self._lock:
            self.status = "error"
            self.last_error = error
            self.finished_at = datetime.now(timezone.utc)
            self.current_item = None
            if self._perf_start is not None:
                self.duration_seconds = time.perf_counter() - self._perf_start

    def eta_seconds(self) -> float | None:
        with self._lock:
            if not self._perf_start or self.completed_items == 0 or self.status != "running":
                return None
            elapsed = time.perf_counter() - self._perf_start
            remaining = max(self.total_items - self.completed_items, 0)
            if remaining == 0:
                return 0.0
            avg = elapsed / self.completed_items
            return remaining * avg

    def elapsed_seconds(self) -> float:
        with self._lock:
            if self._perf_start is None:
                return 0.0
            if self.status in {"success", "error"} and self.finished_at:
                return self.duration_seconds
            return time.perf_counter() - self._perf_start

    def to_model(self) -> BuildStatus:
        with self._lock:
            return BuildStatus(
                project_name=self.project_name,
                status=self.status,
                total_items=self.total_items,
                completed_items=self.completed_items,
                built_images=self.built_images,
                skipped_images=self.skipped_images,
                deleted_images=self.deleted_images,
                eta_seconds=self.eta_seconds(),
                elapsed_seconds=self.elapsed_seconds(),
                started_at=self.started_at,
                finished_at=self.finished_at,
                current_item=self.current_item,
                last_error=self.last_error,
                duration_seconds=self.duration_seconds,
                result=self.result,
                target_size=self.target_size,
            )

class BuildManager:
    def __init__(self):
        self._jobs: Dict[str, BuildJob] = {}
        self._lock = RLock()

    def start_build(self, project: ProjectIndex, target_size: int) -> BuildJob:
        with self._lock:
            existing = self._jobs.get(project.name)
            if existing and existing.is_running():
                return existing
            if existing:
                job = existing
                job.mark_pending(target_size)
            else:
                job = BuildJob(project.name, target_size)
                job.mark_pending(target_size)
            self._jobs[project.name] = job
            thread = Thread(target=self._run, args=(project, job), daemon=True)
            thread.start()
            return job

    def get_status(self, project_name: str) -> BuildJob:
        with self._lock:
            job = self._jobs.get(project_name)
            if job:
                return job
            job = BuildJob(project_name, SETTINGS.default_dataset_size)
            self._jobs[project_name] = job
            return job

    def _run(self, project: ProjectIndex, job: BuildJob) -> None:
        try:
            build_dataset(project, job=job, target_size=job.target_size)
        except Exception as exc:
            job.mark_failed(str(exc))

build_manager = BuildManager()

class IncrementalDatasetBuilder:
    def __init__(self, project: ProjectIndex, job: BuildJob | None, target_size: int):
        self.project = project
        self.job = job
        self.cache = BuildCache(project)
        self.max_workers = max(1, SETTINGS.build_workers)
        self.target_size = target_size

    def run(self) -> BuildDatasetResponse:
        self.project.ensure_loaded()
        output_dir = self.project.dataset_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self.cache.snapshot()
        current_ids = set(self.project.items.keys())
        deletions = sorted(set(snapshot.keys()) - current_ids)

        planned: list[tuple[str, ImageRecord, dict]] = []
        for record_id, record in self.project.items.items():
            meta = self._capture_metadata(record)
            cached = snapshot.get(record_id)
            if self._needs_rebuild(meta, cached, output_dir):
                planned.append((record_id, record, meta))

        total_tasks = len(planned) + len(deletions)
        start = time.perf_counter()
        if self.job:
            self.job.mark_running()
            self.job.set_total(total_tasks)

        processed = 0
        skipped = 0
        deleted_count = 0

        cache_updates: Dict[str, dict] = {}

        for record_id in deletions:
            meta = snapshot.get(record_id)
            self._remove_outputs(meta, output_dir)
            deleted_count += 1
            if self.job:
                self.job.set_current(record_id)
                self.job.advance(success=True, deleted=True)

        if planned:
            built, failures, updates = self._process_plan(planned, output_dir)
            processed += built
            skipped += failures
            cache_updates.update(updates)

        self.cache.commit(cache_updates, deletions)

        duration = time.perf_counter() - start
        result = BuildDatasetResponse(
            processed_images=processed,
            skipped_images=skipped,
            deleted_images=deleted_count,
            total_candidates=total_tasks,
            output_dir=str(output_dir),
            duration_seconds=duration,
            updated_at=datetime.now(timezone.utc),
            target_size=self.target_size,
        )

        if self.job:
            self.job.mark_success(result)

        return result

    def _process_plan(
        self,
        plan: list[tuple[str, ImageRecord, dict]],
        output_dir: Path,
    ) -> tuple[int, int, Dict[str, dict]]:
        if not plan:
            return 0, 0, {}

        worker_count = min(len(plan), self.max_workers)
        processed = 0
        skipped = 0
        updates: Dict[str, dict] = {}

        if worker_count <= 1:
            for record_id, record, meta in plan:
                success = self._build_record_task(record_id, record, output_dir)
                if success:
                    processed += 1
                    updates[record_id] = meta
                else:
                    skipped += 1
            return processed, skipped, updates

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="dataset-build") as executor:
            future_map = {
                executor.submit(self._build_record_task, record_id, record, output_dir): (record_id, meta)
                for record_id, record, meta in plan
            }
            for future in as_completed(future_map):
                record_id, meta = future_map[future]
                try:
                    success = future.result()
                except Exception as exc:
                    warning(f"Dataset build worker crashed for {record_id}: {exc}")
                    success = False
                if success:
                    processed += 1
                    updates[record_id] = meta
                else:
                    skipped += 1

        return processed, skipped, updates

    def _build_record_task(self, record_id: str, record: ImageRecord, output_dir: Path) -> bool:
        try:
            if self.job:
                self.job.set_current(record_id)
            self._write_record(record, output_dir)
        except Exception as exc:
            warning(f"Failed to build dataset asset for {record.filename}: {exc}")
            if self.job:
                self.job.advance(success=False, deleted=False)
            return False
        if self.job:
            self.job.advance(success=True, deleted=False)
        return True

    def _capture_metadata(self, record: ImageRecord) -> dict:
        image_stat = record.image_path.stat()
        annotation_stat = record.annotation_path.stat() if record.annotation_path.exists() else None
        # Bounds are stored in DB, not files - include bounds data directly for change detection
        bounds_data = record.bounds.model_dump() if record.bounds else {}
        return {
            "image_mtime": image_stat.st_mtime,
            "image_size": image_stat.st_size,
            "annotation_mtime": annotation_stat.st_mtime if annotation_stat else 0.0,
            "annotation_size": annotation_stat.st_size if annotation_stat else 0,
            "bounds_data": bounds_data,  # Direct bounds data instead of file stat
            "output_rel_path": record.relative_path,
            "target_size": self.target_size,
        }

    def _needs_rebuild(self, meta: dict, cached: dict | None, output_dir: Path) -> bool:
        if cached is None:
            return True
        if not meta:
            return True
        output_rel_path = meta.get("output_rel_path")
        if not output_rel_path:
            return True
        output_image = output_dir / Path(output_rel_path)
        if not output_image.exists():
            return True
        keys = [
            "image_mtime",
            "image_size",
            "annotation_mtime",
            "annotation_size",
            "bounds_data",
            "target_size",
        ]
        return any(cached.get(key) != meta.get(key) for key in keys)

    def _remove_outputs(self, meta: dict | None, output_dir: Path) -> None:
        if not meta:
            return
        rel = meta.get("output_rel_path")
        if not rel:
            return
        image_path = output_dir / Path(rel)
        annotation_path = image_path.with_suffix(".txt")
        image_path.unlink(missing_ok=True)
        annotation_path.unlink(missing_ok=True)

    def _write_record(self, record: ImageRecord, output_dir: Path) -> None:
        target_dir = output_dir / record.relative_dir if record.relative_dir else output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_image_path = target_dir / record.filename
        target_annotation_path = target_dir / f"{Path(record.filename).stem}.txt"

        with Image.open(record.image_path) as img:
            bounds = record.bounds if record.bounds else BoundsModel()
            left, top, right, bottom = compute_crop_box(img.width, img.height, bounds)
            cropped = img.crop((left, top, right, bottom))
            if cropped.size != (self.target_size, self.target_size):
                cropped = cropped.resize((self.target_size, self.target_size), RESAMPLE_LANCZOS)
            cropped.save(target_image_path)

        annotation_content = record.annotation
        if not annotation_content and record.annotation_path.exists():
            annotation_content = record.annotation_path.read_text(encoding="utf-8")
        target_annotation_path.write_text(annotation_content, encoding="utf-8")

def build_dataset(index: ProjectIndex, job: BuildJob | None = None, target_size: int | None = None) -> BuildDatasetResponse:
    size = target_size or getattr(job, "target_size", None) or SETTINGS.default_dataset_size
    builder = IncrementalDatasetBuilder(index, job, size)
    return builder.run()
