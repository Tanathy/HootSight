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
from system.device import get_device, is_cuda_available


class MemoryManager:
    def __init__(self):
        self.device = get_device()
        self.total_memory = self._get_total_memory()
        self.available_memory = self._get_available_memory()
        try:
            mem_cfg = SETTINGS['memory']
        except Exception:
            raise ValueError("Missing required 'memory' section in config/config.json")
        if 'reserved_memory_ratio' not in mem_cfg:
            raise ValueError("memory.reserved_memory_ratio must be defined in config/config.json")
        self.reserved_memory_ratio = mem_cfg['reserved_memory_ratio']

    def _get_total_memory(self) -> int:
        if is_cuda_available():
            return torch.cuda.get_device_properties(0).total_memory
        else:
            return psutil.virtual_memory().total

    def _get_available_memory(self) -> int:
        if is_cuda_available():
            return torch.cuda.mem_get_info()[0]
        else:
            return psutil.virtual_memory().available

    def calculate_optimal_batch_size(
        self,
        model,
        input_shape: Tuple[int, ...],
        target_memory_usage: float = 0.8,
        safety_margin: float = 0.9
    ) -> Dict[str, Any]:
        try:
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            model.eval()

            with torch.no_grad():
                if is_cuda_available():
                    torch.cuda.reset_peak_memory_stats()
                    dummy_input = dummy_input.to(self.device)
                    model = model.to(self.device)
                    _ = model(dummy_input)
                    memory_per_sample = torch.cuda.max_memory_allocated()
                else:
                    memory_per_sample = sum(p.numel() * p.element_size() for p in model.parameters())
                    memory_per_sample += dummy_input.numel() * dummy_input.element_size()

            available_for_training = int(self.available_memory * target_memory_usage * safety_margin)

            try:
                mem_cfg = SETTINGS['memory']
            except Exception:
                raise ValueError("Missing required 'memory' section in config/config.json")
            if 'memory_per_sample_multiplier' not in mem_cfg:
                raise ValueError("memory.memory_per_sample_multiplier must be defined in config/config.json")
            memory_per_sample *= mem_cfg.get('memory_per_sample_multiplier')

            if 'min_batch_size' not in mem_cfg:
                raise ValueError("memory.min_batch_size must be defined in config/config.json")
            optimal_batch_size = max(mem_cfg.get('min_batch_size'), available_for_training // memory_per_sample)

            if 'max_batch_size' not in mem_cfg:
                raise ValueError("memory.max_batch_size must be defined in config/config.json")
            max_batch = mem_cfg.get('max_batch_size')
            optimal_batch_size = min(optimal_batch_size, max_batch)
            optimal_batch_size = max(optimal_batch_size, 1)
            
            # Ensure batch_size is always an integer (PyTorch DataLoader requirement)
            optimal_batch_size = int(optimal_batch_size)

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
            raise

    def _generate_recommendations(self, batch_size: int, memory_usage: float) -> list:
        recommendations = []
        try:
            mem_cfg = SETTINGS['memory']
        except Exception:
            raise ValueError("Missing required 'memory' section in config/config.json")
        if 'memory_thresholds' not in mem_cfg:
            raise ValueError("memory.memory_thresholds must be defined in config/config.json")
        if 'batch_size_limits' not in mem_cfg:
            raise ValueError("memory.batch_size_limits must be defined in config/config.json")
        memory_thresholds = mem_cfg.get('memory_thresholds')
        batch_limits = mem_cfg.get('batch_size_limits')

        high_usage = memory_thresholds.get('high_usage')
        moderate_usage = memory_thresholds.get('moderate_usage')
        low_usage = memory_thresholds.get('low_usage')
        if high_usage is None or moderate_usage is None or low_usage is None:
            raise ValueError("memory.memory_thresholds entries (high_usage/moderate_usage/low_usage) must be defined in config/config.json")

        if memory_usage > high_usage:
            recommendations.append("WARNING: High memory usage detected. Consider reducing batch size or using gradient accumulation.")
        elif memory_usage > moderate_usage:
            recommendations.append("INFO: Moderate memory usage. Batch size looks good for current setup.")
        else:
            recommendations.append("SUCCESS: Low memory usage. You could potentially increase batch size for better performance.")

        small_batch = batch_limits.get('small_batch_warning')
        large_batch = batch_limits.get('large_batch_warning')
        if small_batch is None or large_batch is None:
            raise ValueError("memory.batch_size_limits entries (small_batch_warning/large_batch_warning) must be defined in config/config.json")

        if batch_size <= small_batch:
            recommendations.append("REDUCE: Small batch size detected. Consider using gradient accumulation for better training stability.")
        elif batch_size >= large_batch:
            recommendations.append("INCREASE: Large batch size detected. Monitor for training stability and consider learning rate adjustments.")

        return recommendations

    def get_memory_status(self) -> Dict[str, Any]:
        if is_cuda_available():
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
        if is_cuda_available():
            torch.cuda.empty_cache()
        gc.collect()


class ThreadedAugmentationManager:
    def __init__(self):
        self.max_threads = self._get_max_threads()
        self.executor = None
        self._shutdown_event = threading.Event()
        self._auto_start = False

    def _get_max_threads(self) -> int:
        try:
            config_threads = SETTINGS['memory']['augmentation_threads']
        except Exception:
            raise ValueError("Missing 'memory.augmentation_threads' in config/config.json")
        if config_threads is None:
            raise ValueError("memory.augmentation_threads must be defined in config/config.json")

        if config_threads == 'auto':
            try:
                sys_max = SETTINGS['system']['max_threads']
            except Exception:
                sys_max = None
            if isinstance(sys_max, int) and sys_max > 0:
                return max(1, min(sys_max, multiprocessing.cpu_count()))
            return max(1, multiprocessing.cpu_count() - 1)
        elif isinstance(config_threads, int):
            return max(1, min(config_threads, multiprocessing.cpu_count()))
        else:
            return max(1, multiprocessing.cpu_count() - 1)

    def start_augmentation_workers(self):
        if self.executor is None or self.executor._shutdown:
            try:
                self.executor = ThreadPoolExecutor(max_workers=self.max_threads)
                info(f"Started augmentation thread pool with {self.max_threads} workers")
            except Exception as e:
                warning(f"Failed to start thread pool: {e}")
                self.executor = None

    def stop_augmentation_workers(self):
        if self.executor and not self.executor._shutdown:
            try:
                self.executor.shutdown(wait=True)
                self.executor = None
                info("Stopped augmentation thread pool")
            except Exception as e:
                warning(f"Error stopping thread pool: {e}")

    def submit_augmentation_task(self, func, *args, **kwargs):
        if self.executor is None or self.executor._shutdown:
            self.start_augmentation_workers()

        if self.executor and not self.executor._shutdown:
            try:
                return self.executor.submit(func, *args, **kwargs)
            except Exception as e:
                warning(f"Failed to submit task: {e}")
                return func(*args, **kwargs)
        else:
            warning("Thread pool not available, running task synchronously")
            return func(*args, **kwargs)

    def get_thread_info(self) -> Dict[str, Any]:
        executor_active = self.executor is not None and not getattr(self.executor, '_shutdown', True)
        return {
            "max_threads": self.max_threads,
            "cpu_count": multiprocessing.cpu_count(),
            "active_threads": threading.active_count(),
            "thread_pool_active": executor_active
        }


memory_manager = MemoryManager()
augmentation_manager = ThreadedAugmentationManager()


def get_optimal_batch_size(model, input_shape: Tuple[int, ...], **kwargs) -> Dict[str, Any]:
    return memory_manager.calculate_optimal_batch_size(model, input_shape, **kwargs)


def get_memory_status() -> Dict[str, Any]:
    return memory_manager.get_memory_status()


def cleanup_memory():
    memory_manager.cleanup_memory()


def get_thread_info() -> Dict[str, Any]:
    return augmentation_manager.get_thread_info()


def initialize_memory_management():
    info("Initializing memory management system")
    if augmentation_manager.executor is None:
        augmentation_manager.start_augmentation_workers()
    success("Memory management system initialized")


def shutdown_memory_management():
    info("Shutting down memory management system")
    augmentation_manager.stop_augmentation_workers()
    cleanup_memory()
    success("Memory management system shutdown complete")