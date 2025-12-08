"""
Checkpoint detection utilities.
Simple, centralized logic for finding model checkpoints.
"""

from pathlib import Path
from typing import Optional, Tuple


def find_checkpoint(model_dir: str, best_filename: str = "best_model.pth", 
                    model_filename: str = "model.pth") -> Tuple[bool, Optional[Path], str]:
    """
    Find a checkpoint file in the model directory.
    
    Search order:
    1. Configured best_model filename (e.g. "best_model.pth")
    2. Fallback to literal "best_model.pth" if different was configured
    3. Configured model filename (e.g. "model.pth")
    
    Case-insensitive matching for Windows compatibility.
    
    Args:
        model_dir: Path to the model directory
        best_filename: Preferred checkpoint filename (default: best_model.pth)
        model_filename: Secondary checkpoint filename (default: model.pth)
    
    Returns:
        Tuple of (found: bool, path: Optional[Path], reason: str)
    """
    model_path = Path(model_dir)
    
    if not model_path.is_dir():
        return False, None, "model_dir_missing"
    
    # Build case-insensitive lookup map
    try:
        files = {f.name.lower(): f for f in model_path.iterdir() if f.is_file()}
    except (OSError, PermissionError):
        return False, None, "model_dir_unreadable"
    
    if not files:
        return False, None, "model_dir_empty"
    
    # Normalize filenames
    best_filename = best_filename or "best_model.pth"
    model_filename = model_filename or "model.pth"
    
    # Search candidates in priority order
    candidates = [
        (best_filename, "best_model"),
        ("best_model.pth", "best_model_fallback"),
        (model_filename, "model"),
        ("model.pth", "model_fallback"),
    ]
    
    seen = set()
    for filename, reason in candidates:
        lower_name = filename.lower()
        if lower_name in seen:
            continue
        seen.add(lower_name)
        
        if lower_name in files:
            found_path = files[lower_name]
            return True, found_path, reason
    
    # Also check for any .pth file as last resort
    for name, path in files.items():
        if name.endswith('.pth'):
            return True, path, "any_pth"
    
    return False, None, "no_checkpoint"
