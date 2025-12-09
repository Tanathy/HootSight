"""Hash utilities shared across the system.

Provides consistent SHA-256 hashing helpers for text, bytes, and files
with optional shortening. These helpers centralize hashing behavior to
avoid ad-hoc implementations across modules.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union


def hash_bytes_sha256(data: bytes, length: Optional[int] = None) -> str:
    """Return SHA-256 hex digest for a bytes payload.

    Args:
        data: Bytes to hash.
        length: Optional length to truncate the hex digest.

    Returns:
        Hex digest (possibly truncated).
    """
    digest = hashlib.sha256(data).hexdigest()
    return digest[:length] if length else digest


def hash_text_sha256(text: str, length: Optional[int] = None) -> str:
    """Hash UTF-8 text with SHA-256 and optional truncation."""
    return hash_bytes_sha256(text.encode("utf-8"), length)


def hash_file_sha256(
    path: Union[str, Path],
    *,
    chunk_size: int = 65536,
    missing_fallback: Optional[str] = None,
    shorten: Optional[int] = None,
    suppress_errors: bool = True,
) -> Optional[str]:
    """Hash file contents with SHA-256, optionally truncating the hex digest.

    Args:
        path: File path to hash.
        chunk_size: Chunk size for streaming reads.
        missing_fallback: Optional string to hash when the file is missing.
        shorten: If provided, truncate the hex digest to this many characters.

    Returns:
        Hex digest string or None when the file cannot be read and no fallback
        is supplied.
    """
    p = Path(path)
    if not p.exists():
        return hash_text_sha256(missing_fallback, shorten) if missing_fallback else None

    try:
        sha = hashlib.sha256()
        with p.open("rb") as fh:
            while chunk := fh.read(chunk_size):
                sha.update(chunk)
        digest = sha.hexdigest()
        return digest[:shorten] if shorten else digest
    except (OSError, IOError):
        if suppress_errors:
            return None
        raise


__all__ = ["hash_bytes_sha256", "hash_text_sha256", "hash_file_sha256"]
