from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

class BoundsModel(BaseModel):
    zoom: float = Field(default=1.0, ge=1.0, le=20.0)
    center_x: float = Field(default=0.5, ge=0.0, le=1.0)
    center_y: float = Field(default=0.5, ge=0.0, le=1.0)

class ProjectSummary(BaseModel):
    name: str
    image_count: int
    data_source_dir: str
    dataset_dir: str

class ImageItem(BaseModel):
    id: str
    filename: str
    extension: str
    category: str
    relative_dir: str
    relative_path: str
    width: int
    height: int
    min_dimension: int
    annotation: str
    tags: List[str]
    image_url: str
    bounds: BoundsModel
    has_custom_bounds: bool
    updated_at: Optional[datetime]

class PaginatedItems(BaseModel):
    items: List[ImageItem]
    page: int
    page_size: int
    total_items: int
    total_pages: int

class AnnotationUpdate(BaseModel):
    content: str = Field(min_length=0)

class BoundsUpdate(BaseModel):
    zoom: float
    center_x: float
    center_y: float

class StatsSummary(BaseModel):
    total_images: int
    categories: Dict[str, int]
    tag_frequencies: Dict[str, int]
    least_common_tags: List[str]
    most_common_tags: List[str]
    recommendations: List[Dict[str, Any]]

    label_distribution: Optional[Dict[str, int]] = None
    balance_score: Optional[float] = None
    balance_status: Optional[str] = None
    balance_analysis: Optional[Dict[str, Any]] = None

class BuildDatasetResponse(BaseModel):
    processed_images: int
    skipped_images: int
    deleted_images: int
    total_candidates: int
    output_dir: str
    duration_seconds: float
    updated_at: datetime
    target_size: int

class BuildStatus(BaseModel):
    project_name: str
    status: Literal["idle", "pending", "running", "success", "error"]
    total_items: int
    completed_items: int
    built_images: int
    skipped_images: int
    deleted_images: int
    eta_seconds: Optional[float]
    elapsed_seconds: float
    duration_seconds: float
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    current_item: Optional[str]
    last_error: Optional[str]
    result: Optional[BuildDatasetResponse]
    target_size: int

class DiscoveryResponse(BaseModel):
    total_candidates: int
    processed_items: int
    added_items: int
    updated_items: int
    removed_items: int
    skipped_items: int
    duration_seconds: float
    updated_at: datetime

class DiscoveryStatus(BaseModel):
    project_name: str
    status: Literal["idle", "pending", "running", "success", "error"]
    total_items: int
    processed_items: int
    added_items: int
    updated_items: int
    removed_items: int
    skipped_items: int
    eta_seconds: Optional[float]
    elapsed_seconds: float
    duration_seconds: float
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    current_item: Optional[str]
    last_error: Optional[str]
    result: Optional[DiscoveryResponse]

class FolderNode(BaseModel):
    name: str
    path: str
    image_count: int
    children: List["FolderNode"] = Field(default_factory=list)

FolderNode.model_rebuild()

class BulkTagRequest(BaseModel):
    add: List[str] = Field(default_factory=list)
    remove: List[str] = Field(default_factory=list)
    search: Optional[str] = Field(default=None)
    folder: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    items: Optional[List[str]] = Field(default=None)

class BulkTagResponse(BaseModel):
    updated: int
    skipped: int
    added_tags: List[str]
    removed_tags: List[str]
    updated_at: datetime

class BuildRequest(BaseModel):
    size: Optional[int] = Field(default=None, ge=32, le=8192)

class BuildOptions(BaseModel):
    presets: Dict[str, List[int]]
    default_size: int
    active_size: int

class DeleteItemsRequest(BaseModel):
    items: List[str] = Field(min_length=1)

class DeleteItemsResponse(BaseModel):
    deleted: int
    missing: List[str] = Field(default_factory=list)

class FolderCreateRequest(BaseModel):
    path: str = Field(min_length=1, description="Relative path within data_source")

class FolderRenameRequest(BaseModel):
    old_path: str = Field(min_length=1)
    new_name: str = Field(min_length=1)

class FolderDeleteRequest(BaseModel):
    path: str = Field(min_length=1)
    recursive: bool = Field(default=False)

class FolderOperationResponse(BaseModel):
    success: bool
    path: str
    message: str = ""

class UploadResponse(BaseModel):
    uploaded: int
    failed: int
    files: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class DuplicateGroup(BaseModel):
    content_hash: str
    images: List[ImageItem]


class DuplicateScanResponse(BaseModel):
    total_groups: int
    total_duplicates: int
    groups: List[DuplicateGroup]
    scanned_images: int
    duration_seconds: float


__all__ = [
    "AnnotationUpdate",
    "BoundsModel",
    "BoundsUpdate",
    "BuildDatasetResponse",
    "BuildOptions",
    "BuildRequest",
    "BuildStatus",
    "BulkTagRequest",
    "BulkTagResponse",
    "DeleteItemsRequest",
    "DeleteItemsResponse",
    "DiscoveryResponse",
    "DiscoveryStatus",
    "DuplicateGroup",
    "DuplicateScanResponse",
    "FolderCreateRequest",
    "FolderDeleteRequest",
    "FolderNode",
    "FolderOperationResponse",
    "FolderRenameRequest",
    "ImageItem",
    "PaginatedItems",
    "ProjectSummary",
    "StatsSummary",
    "UploadResponse",
]
