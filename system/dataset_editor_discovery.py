from __future__ import annotations

import time
from datetime import datetime, timezone
from threading import RLock, Thread
from typing import Dict

from system.dataset_editor_models import DiscoveryResponse, DiscoveryStatus
from system.dataset_editor_project_service import ProjectIndex
from system.log import error

class DiscoveryJob:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.status = "idle"
        self.total_items = 0
        self.processed_items = 0
        self.added_items = 0
        self.updated_items = 0
        self.removed_items = 0
        self.skipped_items = 0
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.duration_seconds: float = 0.0
        self.eta_seconds: float | None = None
        self.elapsed_seconds: float = 0.0
        self.current_item: str | None = None
        self.last_error: str | None = None
        self.result: DiscoveryResponse | None = None
        self._perf_start: float | None = None
        self._lock = RLock()

    def is_running(self) -> bool:
        with self._lock:
            return self.status in {"pending", "running"}

    def mark_pending(self) -> None:
        with self._lock:
            self.status = "pending"
            self.total_items = 0
            self.processed_items = 0
            self.added_items = 0
            self.updated_items = 0
            self.removed_items = 0
            self.skipped_items = 0
            self.started_at = None
            self.finished_at = None
            self.duration_seconds = 0.0
            self.elapsed_seconds = 0.0
            self.eta_seconds = None
            self.current_item = None
            self.last_error = None
            self.result = None
            self._perf_start = None

    def mark_running(self) -> None:
        with self._lock:
            self.status = "running"
            self.started_at = datetime.now(timezone.utc)
            self.finished_at = None
            self._perf_start = time.perf_counter()

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total_items = total

    def set_current(self, key: str | None) -> None:
        with self._lock:
            self.current_item = key

    def advance(self, *, added: bool = False, updated: bool = False, removed: bool = False, skipped: bool = False) -> None:
        with self._lock:
            self.processed_items += 1
            if added:
                self.added_items += 1
            if updated:
                self.updated_items += 1
            if removed:
                self.removed_items += 1
            if skipped:
                self.skipped_items += 1
            self.current_item = None
            self._update_eta()

    def add_removed(self, count: int) -> None:
        with self._lock:
            self.removed_items += count

    def mark_success(self, result: DiscoveryResponse) -> None:
        with self._lock:
            self.status = "success"
            self.result = result
            self.finished_at = datetime.now(timezone.utc)
            if self._perf_start is not None:
                self.duration_seconds = time.perf_counter() - self._perf_start
                self.elapsed_seconds = self.duration_seconds
            self.current_item = None
            self.eta_seconds = 0.0

    def mark_failed(self, error: str) -> None:
        with self._lock:
            self.status = "error"
            self.last_error = error
            self.finished_at = datetime.now(timezone.utc)
            if self._perf_start is not None:
                self.duration_seconds = time.perf_counter() - self._perf_start
                self.elapsed_seconds = self.duration_seconds
            self.current_item = None

    def _update_eta(self) -> None:
        if not self._perf_start or self.processed_items == 0:
            self.eta_seconds = None
            self.elapsed_seconds = 0.0 if not self._perf_start else time.perf_counter() - self._perf_start
            return
        elapsed = time.perf_counter() - self._perf_start
        self.elapsed_seconds = elapsed
        if self.total_items <= 0 or self.processed_items == 0:
            self.eta_seconds = None
            return
        remaining = max(self.total_items - self.processed_items, 0)
        avg = elapsed / max(self.processed_items, 1)
        self.eta_seconds = remaining * avg

    def to_model(self) -> DiscoveryStatus:
        with self._lock:
            return DiscoveryStatus(
                project_name=self.project_name,
                status=self.status,
                total_items=self.total_items,
                processed_items=self.processed_items,
                added_items=self.added_items,
                updated_items=self.updated_items,
                removed_items=self.removed_items,
                skipped_items=self.skipped_items,
                eta_seconds=self.eta_seconds,
                elapsed_seconds=self.elapsed_seconds,
                duration_seconds=self.duration_seconds,
                started_at=self.started_at,
                finished_at=self.finished_at,
                current_item=self.current_item,
                last_error=self.last_error,
                result=self.result,
            )

class DiscoveryManager:
    def __init__(self):
        self._jobs: Dict[str, DiscoveryJob] = {}
        self._lock = RLock()

    def start(self, project: ProjectIndex) -> DiscoveryJob:
        with self._lock:
            existing = self._jobs.get(project.name)
            if existing and existing.is_running():
                return existing
            job = DiscoveryJob(project.name)
            job.mark_pending()
            self._jobs[project.name] = job
            thread = Thread(target=self._run, args=(project, job), daemon=True)
            thread.start()
            return job

    def get(self, project_name: str) -> DiscoveryJob:
        with self._lock:
            job = self._jobs.get(project_name)
            if job:
                return job
            job = DiscoveryJob(project_name)
            self._jobs[project_name] = job
            return job

    def _run(self, project: ProjectIndex, job: DiscoveryJob) -> None:
        try:
            project.run_discovery(job)
        except Exception as exc:
            error(f"Dataset discovery failed for {project.name}: {exc}")
            job.mark_failed(str(exc))

discovery_manager = DiscoveryManager()
