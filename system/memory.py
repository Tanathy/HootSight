"""Memory management module for Hootsight.

Handles memory calculations, batch size optimization, and threaded data augmentation
to prevent out-of-memory errors during training.
"""
import os
import psutil
import torch
from typing import Dict, Any, Optional, Tuple
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import gc

from system.log import info, warning, error, success
from system.coordinator_settings import SETTINGS


class MemoryManager:
    """Manages memory usage and batch size calculations for training."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_memory = self._get_total_memory()
        self.available_memory = self._get_available_memory()
        self.reserved_memory_ratio = SETTINGS.get('memory', {}).get('reserved_memory_ratio', 0.1)

    def _get_total_memory(self) -> int:
        """Get total system memory in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        else:
            return psutil.virtual_memory().total

    def _get_available_memory(self) -> int:
        """Get available memory in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0]  # Free memory
        else:
            return psutil.virtual_memory().available

    def calculate_optimal_batch_size(
        self,
        model,
        input_shape: Tuple[int, ...],
        target_memory_usage: float = 0.8,
        safety_margin: float = 0.9
    ) -> Dict[str, Any]:
        """Calculate optimal batch size based on available memory.

        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (excluding batch dimension)
            target_memory_usage: Target memory usage ratio (0.0-1.0)
            safety_margin: Safety margin for calculations (0.0-1.0)

        Returns:
            dict: Batch size recommendations and memory analysis
        """
        try:
            # Calculate memory per sample
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            model.eval()

            with torch.no_grad():
                # Forward pass to measure memory usage
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    dummy_input = dummy_input.to(self.device)
                    model = model.to(self.device)
                    _ = model(dummy_input)
                    memory_per_sample = torch.cuda.max_memory_allocated()
                else:
                    # For CPU, estimate based on model parameters
                    memory_per_sample = sum(p.numel() * p.element_size() for p in model.parameters())
                    memory_per_sample += dummy_input.numel() * dummy_input.element_size()

            # Calculate available memory for training
            available_for_training = int(self.available_memory * target_memory_usage * safety_margin)

            # Reserve memory for gradients and optimizer states
            memory_per_sample *= SETTINGS.get('memory', {}).get('memory_per_sample_multiplier', 3)

            # Calculate optimal batch size
            optimal_batch_size = max(SETTINGS.get('memory', {}).get('min_batch_size', 1), available_for_training // memory_per_sample)

            # Apply practical limits
            max_batch = SETTINGS.get('memory', {}).get('max_batch_size', 512)
            optimal_batch_size = min(optimal_batch_size, max_batch)
            optimal_batch_size = max(optimal_batch_size, 1)    # Min batch size

            # Calculate memory usage for recommended batch size
            estimated_memory_usage = (memory_per_sample * optimal_batch_size) / self.total_memory

            return {
                "optimal_batch_size": optimal_batch_size,
                "estimated_memory_per_sample": memory_per_sample,
                "estimated_total_memory_usage": estimated_memory_usage,
                "available_memory": self.available_memory,
                "total_memory": self.total_memory,
                "device": str(self.device),
                "recommendations": self._generate_recommendations(optimal_batch_size, estimated_memory_usage)
            }

        except Exception as e:
            error(f"Failed to calculate optimal batch size: {e}")
            return {
                "error": f"Batch size calculation failed: {str(e)}",
                "fallback_batch_size": SETTINGS.get('system', {}).get('fallback_batch_size', 8),
                "device": str(self.device)
            }

    def _generate_recommendations(self, batch_size: int, memory_usage: float) -> list:
        """Generate recommendations based on calculations."""
        recommendations = []
        memory_thresholds = SETTINGS.get('memory', {}).get('memory_thresholds', {})
        batch_limits = SETTINGS.get('memory', {}).get('batch_size_limits', {})

        high_usage = memory_thresholds.get('high_usage', 0.9)
        moderate_usage = memory_thresholds.get('moderate_usage', 0.7)
        low_usage = memory_thresholds.get('low_usage', 0.3)

        if memory_usage > high_usage:
            recommendations.append("WARNING: High memory usage detected. Consider reducing batch size or using gradient accumulation.")
        elif memory_usage > moderate_usage:
            recommendations.append("INFO: Moderate memory usage. Batch size looks good for current setup.")
        else:
            recommendations.append("SUCCESS: Low memory usage. You could potentially increase batch size for better performance.")

        small_batch = batch_limits.get('small_batch_warning', 4)
        large_batch = batch_limits.get('large_batch_warning', 128)

        if batch_size <= small_batch:
            recommendations.append("REDUCE: Small batch size detected. Consider using gradient accumulation for better training stability.")
        elif batch_size >= large_batch:
            recommendations.append("INCREASE: Large batch size detected. Monitor for training stability and consider learning rate adjustments.")

        return recommendations

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            free, total = torch.cuda.mem_get_info()

            return {
                "device": "cuda",
                "allocated_mb": allocated / 1024 / 1024,
                "reserved_mb": reserved / 1024 / 1024,
                "free_mb": free / 1024 / 1024,
                "total_mb": total / 1024 / 1024,
                "utilization_percent": (allocated / total) * 100
            }
        else:
            memory = psutil.virtual_memory()
            return {
                "device": "cpu",
                "total_mb": memory.total / 1024 / 1024,
                "available_mb": memory.available / 1024 / 1024,
                "used_mb": memory.used / 1024 / 1024,
                "utilization_percent": memory.percent
            }

    def cleanup_memory(self):
        """Clean up memory to free resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class ThreadedAugmentationManager:
    """Manages threaded data augmentation for faster preprocessing."""

    def __init__(self):
        self.max_threads = self._get_max_threads()
        self.executor = None
        self._shutdown_event = threading.Event()
        # Don't start executor automatically to avoid FastAPI issues
        self._auto_start = False

    def _get_max_threads(self) -> int:
        """Get maximum number of threads based on config or auto-detection."""
        config_threads = SETTINGS.get('memory', {}).get('augmentation_threads', 'auto')

        if config_threads == 'auto':
            # Cap by system.max_threads if provided
            sys_max = SETTINGS.get('system', {}).get('max_threads')
            if isinstance(sys_max, int) and sys_max > 0:
                return max(1, min(sys_max, multiprocessing.cpu_count()))
            return max(1, multiprocessing.cpu_count() - 1)
        elif isinstance(config_threads, int):
            return max(1, min(config_threads, multiprocessing.cpu_count()))
        else:
            return max(1, multiprocessing.cpu_count() - 1)

    def start_augmentation_workers(self):
        """Start the thread pool for augmentation tasks."""
        if self.executor is None or self.executor._shutdown:
            try:
                self.executor = ThreadPoolExecutor(max_workers=self.max_threads)
                info(f"Started augmentation thread pool with {self.max_threads} workers")
            except Exception as e:
                warning(f"Failed to start thread pool: {e}")
                self.executor = None

    def stop_augmentation_workers(self):
        """Stop the thread pool."""
        if self.executor and not self.executor._shutdown:
            try:
                self.executor.shutdown(wait=True)
                self.executor = None
                info("Stopped augmentation thread pool")
            except Exception as e:
                warning(f"Error stopping thread pool: {e}")

    def submit_augmentation_task(self, func, *args, **kwargs):
        """Submit an augmentation task to the thread pool."""
        if self.executor is None or self.executor._shutdown:
            self.start_augmentation_workers()

        if self.executor and not self.executor._shutdown:
            try:
                return self.executor.submit(func, *args, **kwargs)
            except Exception as e:
                warning(f"Failed to submit task: {e}")
                # Fallback to synchronous execution
                return func(*args, **kwargs)
        else:
            # Fallback if executor couldn't be started
            warning("Thread pool not available, running task synchronously")
            return func(*args, **kwargs)

    def get_thread_info(self) -> Dict[str, Any]:
        """Get information about current threading setup."""
        executor_active = self.executor is not None and not getattr(self.executor, '_shutdown', True)
        return {
            "max_threads": self.max_threads,
            "cpu_count": multiprocessing.cpu_count(),
            "active_threads": threading.active_count(),
            "thread_pool_active": executor_active
        }


# Global instances
memory_manager = MemoryManager()
augmentation_manager = ThreadedAugmentationManager()


def get_optimal_batch_size(model, input_shape: Tuple[int, ...], **kwargs) -> Dict[str, Any]:
    """Convenience function to get optimal batch size."""
    return memory_manager.calculate_optimal_batch_size(model, input_shape, **kwargs)


def get_memory_status() -> Dict[str, Any]:
    """Convenience function to get memory status."""
    return memory_manager.get_memory_status()


def cleanup_memory():
    """Convenience function to cleanup memory."""
    memory_manager.cleanup_memory()


def get_thread_info() -> Dict[str, Any]:
    """Convenience function to get thread information."""
    return augmentation_manager.get_thread_info()


def initialize_memory_management():
    """Initialize memory management system."""
    info("Initializing memory management system")
    # Only start thread pool if not already started
    if augmentation_manager.executor is None:
        augmentation_manager.start_augmentation_workers()
    success("Memory management system initialized")


def shutdown_memory_management():
    """Shutdown memory management system."""
    info("Shutting down memory management system")
    augmentation_manager.stop_augmentation_workers()
    cleanup_memory()
    success("Memory management system shutdown complete")