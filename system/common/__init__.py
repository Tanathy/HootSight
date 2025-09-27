"""Common utility functions for Hootsight system.

This module contains shared functions used across the system.
"""
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from system.log import info, warning, error


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure

    Returns:
        Path object of the ensured directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file safely.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary containing JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        error(f"Invalid JSON in file {file_path}: {e}")
        raise


def save_json_file(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        info(f"JSON file saved: {file_path}")
    except Exception as e:
        error(f"Failed to save JSON file {file_path}: {e}")
        raise


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """Calculate file hash.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)

    Returns:
        Hexadecimal hash string
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def get_device() -> torch.device:
    """Get the best available device for PyTorch.

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        info("Using CPU device")

    return device


def setup_torch_reproducibility(seed: int = 42) -> None:
    """Setup PyTorch for reproducible results.

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def format_number(num: Union[int, float], precision: int = 4) -> str:
    """Format number with specified precision.

    Args:
        num: Number to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    if isinstance(num, float):
        return f"{num:.{precision}f}"
    return str(num)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dictionary with memory statistics
    """
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved() / 1024**2,     # MB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,  # MB
            'max_cached': torch.cuda.max_memory_reserved() / 1024**2       # MB
        }
    return {}


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """Validate configuration has required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required keys

    Returns:
        List of missing keys
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    return missing_keys


def create_experiment_id() -> str:
    """Create unique experiment ID based on timestamp.

    Returns:
        Unique experiment identifier
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"exp_{timestamp}"


def safe_divide(a: Union[int, float], b: Union[int, float], default: float = 0.0) -> float:
    """Safe division that handles division by zero.

    Args:
        a: Numerator
        b: Denominator
        default: Default value if division by zero

    Returns:
        Division result or default value
    """
    try:
        return float(a) / float(b) if b != 0 else default
    except (ValueError, TypeError):
        return default


def flatten_dict(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Flatten nested dictionary with dot notation.

    Args:
        d: Dictionary to flatten
        prefix: Prefix for keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_nested_value(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """Get nested value from dictionary.

    Args:
        d: Dictionary to search
        keys: List of keys for nested access
        default: Default value if key not found

    Returns:
        Value or default
    """
    current = d
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current