"""
Duplicate Detection Module

Provides content-based duplicate detection for images in the dataset.
Uses SHA256 hash of file content for exact duplicate detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from system.log import info, warning
from system.common.hash_utils import hash_file_sha256


CHUNK_SIZE = 65536  # 64KB chunks for efficient file reading


def compute_content_hash(file_path: Path) -> Optional[str]:
    """
    Compute SHA256 hash of file content.
    Returns None if file cannot be read.
    """
    if not file_path.exists():
        return None

    try:
        return hash_file_sha256(file_path, chunk_size=CHUNK_SIZE, suppress_errors=False)
    except (IOError, OSError) as exc:
        warning(f"Failed to hash file {file_path}: {exc}")
        return None


def compute_content_hash_batch(
    file_paths: List[Path],
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, str]:
    """
    Compute content hashes for multiple files in parallel.
    
    Args:
        file_paths: List of file paths to hash
        max_workers: Number of parallel workers
        progress_callback: Optional callback(completed, total) for progress
        
    Returns:
        Dict mapping relative_path -> content_hash
    """
    results = {}
    total = len(file_paths)
    completed = 0
    
    def hash_file(path: Path) -> Tuple[str, Optional[str]]:
        return str(path), compute_content_hash(path)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(hash_file, p): p for p in file_paths}
        
        for future in as_completed(futures):
            path_str, hash_val = future.result()
            if hash_val:
                results[path_str] = hash_val
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
    
    return results


def find_duplicates(hash_map: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Find duplicate files from a hash map.
    
    Args:
        hash_map: Dict mapping relative_path -> content_hash
        
    Returns:
        Dict mapping content_hash -> list of relative_paths (only groups with 2+ files)
    """
    # Invert the hash map: group paths by their hash
    hash_to_paths: Dict[str, List[str]] = defaultdict(list)
    
    for path, content_hash in hash_map.items():
        hash_to_paths[content_hash].append(path)
    
    # Filter to only groups with duplicates
    duplicates = {
        h: paths for h, paths in hash_to_paths.items()
        if len(paths) > 1
    }
    
    return duplicates


def scan_for_duplicates(
    image_records: List[dict],
    data_source_dir: Path,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Scan images for duplicates.
    
    Args:
        image_records: List of image record dicts with 'relative_path' key
        data_source_dir: Base directory for images
        max_workers: Number of parallel workers
        progress_callback: Optional callback(completed, total) for progress
        
    Returns:
        Tuple of (hash_map, duplicate_groups)
        - hash_map: Dict mapping relative_path -> content_hash
        - duplicate_groups: Dict mapping content_hash -> list of relative_paths
    """
    # Build list of full paths
    file_paths = []
    path_to_relative: Dict[str, str] = {}
    
    for record in image_records:
        relative_path = record.get('relative_path', '')
        if relative_path:
            full_path = data_source_dir / relative_path
            if full_path.exists():
                file_paths.append(full_path)
                path_to_relative[str(full_path)] = relative_path
    
    if not file_paths:
        return {}, {}
    
    info(f"Scanning {len(file_paths)} images for duplicates")
    
    # Compute hashes
    raw_hashes = compute_content_hash_batch(file_paths, max_workers, progress_callback)
    
    # Convert full paths back to relative paths
    hash_map = {
        path_to_relative[p]: h
        for p, h in raw_hashes.items()
        if p in path_to_relative
    }
    
    # Find duplicates
    duplicate_groups = find_duplicates(hash_map)
    
    if duplicate_groups:
        total_dupes = sum(len(paths) for paths in duplicate_groups.values())
        info(f"Found {len(duplicate_groups)} duplicate groups with {total_dupes} total files")
    else:
        info("No duplicates found")
    
    return hash_map, duplicate_groups
