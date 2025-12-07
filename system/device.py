"""Device utilities for Hootsight.

Provides centralized device detection and management for cross-platform compatibility.
Supports CUDA, CPU, and potentially other accelerators (MPS for Apple Silicon).
"""

import torch
from typing import Optional, Any
from contextlib import contextmanager

from system.log import info


# Cached device instance
_device: Optional[torch.device] = None
_device_type: Optional[str] = None
_device_logged: bool = False


def get_device(log: bool = True) -> torch.device:
    """Get the best available device for computation.
    
    Args:
        log: Whether to log the device selection (disable for worker processes)
    
    Returns:
        torch.device: CUDA if available, otherwise CPU
    """
    global _device, _device_logged
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device('cuda')
            if log and not _device_logged:
                info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                _device_logged = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon MPS support
            _device = torch.device('mps')
            if log and not _device_logged:
                info("Using Apple MPS device")
                _device_logged = True
        else:
            _device = torch.device('cpu')
            if log and not _device_logged:
                info("Using CPU device (no GPU acceleration available)")
                _device_logged = True
    return _device


def get_device_type() -> str:
    """Get device type string for torch.amp functions.
    
    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    global _device_type
    if _device_type is None:
        device = get_device()
        _device_type = device.type
    return _device_type


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if Apple Metal Performance Shaders (MPS) is available."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def is_amp_supported() -> bool:
    """Check if automatic mixed precision is supported on current device.
    
    AMP is supported on CUDA and MPS, but not on CPU.
    """
    device_type = get_device_type()
    return device_type in ('cuda', 'mps')


def create_grad_scaler(enabled: bool = True) -> Any:
    """Create a GradScaler for mixed precision training.
    
    Args:
        enabled: Whether mixed precision is enabled in config
        
    Returns:
        GradScaler if AMP is supported and enabled, None otherwise
    """
    if not enabled:
        return None
        
    device_type = get_device_type()
    
    if device_type == 'cuda':
        return torch.amp.GradScaler(device_type)  # type: ignore[attr-defined]
    elif device_type == 'mps':
        # MPS supports autocast but GradScaler behavior may differ
        return torch.amp.GradScaler(device_type)  # type: ignore[attr-defined]
    else:
        # CPU doesn't benefit from GradScaler
        return None


@contextmanager
def autocast_context(enabled: bool = True):
    """Context manager for automatic mixed precision.
    
    Args:
        enabled: Whether AMP is enabled
        
    Yields:
        Autocast context if supported, otherwise no-op context
    """
    if not enabled:
        yield
        return
        
    device_type = get_device_type()
    
    if device_type in ('cuda', 'mps'):
        with torch.amp.autocast(device_type):  # type: ignore[attr-defined]
            yield
    else:
        # CPU autocast is supported in newer PyTorch but with limited benefit
        # For safety, just yield without autocast on CPU
        yield


def get_device_info() -> dict:
    """Get detailed device information.
    
    Returns:
        dict: Device information including type, name, memory, etc.
    """
    device = get_device()
    info_dict = {
        'device_type': device.type,
        'amp_supported': is_amp_supported()
    }
    
    if device.type == 'cuda':
        try:
            cuda_version = torch.version.cuda  # type: ignore[attr-defined]
        except AttributeError:
            cuda_version = None
        info_dict.update({
            'device_name': torch.cuda.get_device_name(0),
            'device_count': torch.cuda.device_count(),
            'cuda_version': cuda_version,
            'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })
    elif device.type == 'mps':
        info_dict.update({
            'device_name': 'Apple Silicon MPS',
        })
    else:
        info_dict.update({
            'device_name': 'CPU',
        })
    
    return info_dict


def setup_torch_reproducibility(seed: int = 42) -> None:
    """Setup PyTorch for reproducible results.
    
    Configures seeds and deterministic algorithms for all available devices.
    
    Args:
        seed: Random seed (default: 42)
    """
    import numpy as np
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if is_cuda_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if is_mps_available():
        # MPS doesn't have separate seed functions yet
        pass


def get_memory_usage() -> dict:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory statistics (in MB)
    """
    if is_cuda_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'cached': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,
            'max_cached': torch.cuda.max_memory_reserved() / 1024**2
        }
    return {}

def reset_device_cache():
    """Reset cached device. Useful for testing or reconfiguration."""
    global _device, _device_type
    _device = None
    _device_type = None
