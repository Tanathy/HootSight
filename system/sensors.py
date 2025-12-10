"""
HootSight - System Sensors Module
Hardware monitoring for CPU, Memory, and GPU statistics
"""

from __future__ import annotations

import platform
import subprocess
from typing import Any, Dict, List, Optional

import psutil
import torch

from system.device import is_cuda_available


def _get_cpu_name_windows() -> str:
    """Get CPU name from Windows registry."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        winreg.CloseKey(key)
        return cpu_name.strip()
    except Exception:
        return ""


def _get_cpu_name_linux() -> str:
    """Get CPU name from /proc/cpuinfo on Linux."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return ""


def _get_cpu_name_macos() -> str:
    """Get CPU name on macOS using sysctl."""
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            encoding="utf-8",
            timeout=2
        )
        return output.strip()
    except Exception:
        pass
    return ""


def _get_cpu_name() -> str:
    """Get CPU name across platforms."""
    system = platform.system()
    cpu_name = ""
    
    if system == "Windows":
        cpu_name = _get_cpu_name_windows()
    elif system == "Linux":
        cpu_name = _get_cpu_name_linux()
    elif system == "Darwin":
        cpu_name = _get_cpu_name_macos()
    
    if not cpu_name:
        cpu_name = platform.processor()
    
    if cpu_name in ("", "Intel64", "AMD64", "x86_64", "aarch64"):
        cpu_name = ""
    
    return cpu_name if cpu_name else "Unknown CPU"


def _get_nvidia_smi_data() -> Dict[int, Dict[str, Any]]:
    """Query nvidia-smi for detailed GPU information."""
    nvidia_data: Dict[int, Dict[str, Any]] = {}
    
    try:
        output = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,fan.speed,pcie.link.gen.current,pcie.link.width.current,clocks.current.graphics,clocks.current.memory,driver_version',
            '--format=csv,noheader,nounits'
        ], encoding='utf-8', timeout=3, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
        
        for line in output.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 15:
                idx = int(parts[0])
                
                def parse_float(val: str) -> Optional[float]:
                    if val in ('[N/A]', '[Not Supported]', 'N/A', ''):
                        return None
                    try:
                        return float(val)
                    except ValueError:
                        return None
                
                def parse_int(val: str) -> Optional[int]:
                    if val in ('[N/A]', '[Not Supported]', 'N/A', ''):
                        return None
                    try:
                        return int(val)
                    except ValueError:
                        return None
                
                nvidia_data[idx] = {
                    "name": parts[1],
                    "usage": parse_float(parts[2]) or 0,
                    "memory_total_mb": parse_float(parts[3]) or 0,
                    "memory_used_mb": parse_float(parts[4]) or 0,
                    "memory_free_mb": parse_float(parts[5]) or 0,
                    "temperature": parse_float(parts[6]),
                    "power_draw_w": parse_float(parts[7]),
                    "power_limit_w": parse_float(parts[8]),
                    "fan_speed": parse_int(parts[9]),
                    "pcie_gen": parse_int(parts[10]),
                    "pcie_width": parse_int(parts[11]),
                    "clock_graphics_mhz": parse_int(parts[12]),
                    "clock_memory_mhz": parse_int(parts[13]),
                    "driver_version": parts[14] if parts[14] not in ('[N/A]', '') else None
                }
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    
    return nvidia_data


def get_cpu_stats() -> Dict[str, Any]:
    """Get CPU statistics."""
    cpu_freq = psutil.cpu_freq()
    
    cpu_temp = None
    try:
        if hasattr(psutil, 'sensors_temperatures'):
            temps = psutil.sensors_temperatures()
            if temps:
                for name in ['coretemp', 'k10temp', 'cpu_thermal', 'cpu-thermal']:
                    if name in temps and temps[name]:
                        cpu_temp = temps[name][0].current
                        break
                if cpu_temp is None:
                    for entries in temps.values():
                        if entries:
                            cpu_temp = entries[0].current
                            break
    except Exception:
        pass
    
    per_core = psutil.cpu_percent(interval=None, percpu=True)
    
    return {
        "name": _get_cpu_name(),
        "usage": psutil.cpu_percent(interval=None),
        "usage_per_core": per_core,
        "core_count": psutil.cpu_count(logical=False) or 0,
        "thread_count": psutil.cpu_count(logical=True) or 0,
        "frequency_mhz": round(cpu_freq.current, 0) if cpu_freq else 0,
        "frequency_max_mhz": round(cpu_freq.max, 0) if cpu_freq and cpu_freq.max else None,
        "frequency_min_mhz": round(cpu_freq.min, 0) if cpu_freq and cpu_freq.min else None,
        "temperature": cpu_temp
    }


def get_memory_stats() -> Dict[str, Any]:
    """Get system memory statistics."""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        "usage": mem.percent,
        "total_mb": round(mem.total / (1024 * 1024), 0),
        "available_mb": round(mem.available / (1024 * 1024), 0),
        "used_mb": round(mem.used / (1024 * 1024), 0),
        "swap_total_mb": round(swap.total / (1024 * 1024), 0),
        "swap_used_mb": round(swap.used / (1024 * 1024), 0),
        "swap_percent": swap.percent
    }


def get_gpu_stats() -> List[Dict[str, Any]]:
    """Get GPU statistics for all available GPUs."""
    gpus: List[Dict[str, Any]] = []
    
    if not is_cuda_available():
        return gpus
    
    try:
        gpu_count = torch.cuda.device_count()
        nvidia_data = _get_nvidia_smi_data()
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            nv = nvidia_data.get(i, {})
            
            mem_total_mb = nv.get("memory_total_mb") or (props.total_memory / (1024 * 1024))
            mem_used_mb = nv.get("memory_used_mb") or (torch.cuda.memory_allocated(i) / (1024 * 1024))
            mem_free_mb = nv.get("memory_free_mb") or (mem_total_mb - mem_used_mb)
            mem_percent = (mem_used_mb / mem_total_mb) * 100 if mem_total_mb > 0 else 0
            
            gpu_info: Dict[str, Any] = {
                "index": i,
                "name": nv.get("name") or props.name,
                "usage": nv.get("usage", 0),
                "memory_percent": round(mem_percent, 1),
                "memory_total_mb": round(mem_total_mb, 0),
                "memory_used_mb": round(mem_used_mb, 0),
                "memory_free_mb": round(mem_free_mb, 0),
                "temperature": nv.get("temperature"),
                "power_draw_w": nv.get("power_draw_w"),
                "power_limit_w": nv.get("power_limit_w"),
                "fan_speed": nv.get("fan_speed"),
                "clock_graphics_mhz": nv.get("clock_graphics_mhz"),
                "clock_memory_mhz": nv.get("clock_memory_mhz"),
                "pcie_gen": nv.get("pcie_gen"),
                "pcie_width": nv.get("pcie_width"),
                "driver_version": nv.get("driver_version"),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            }
            
            gpus.append(gpu_info)
            
    except Exception:
        pass
    
    return gpus


def get_platform_info() -> Dict[str, Any]:
    """Get platform/OS information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version()
    }


def get_system_stats() -> Dict[str, Any]:
    """Get comprehensive system statistics."""
    cpu = get_cpu_stats()
    memory = get_memory_stats()
    gpus = get_gpu_stats()
    
    return {
        "platform": get_platform_info(),
        
        "cpu": cpu["usage"],
        "cpu_name": cpu["name"],
        "cpu_speed_mhz": cpu["frequency_mhz"],
        "cpu_temp": cpu["temperature"],
        "cpu_cores": cpu["core_count"],
        "cpu_threads": cpu["thread_count"],
        
        "memory": memory["usage"],
        "memory_total_mb": memory["total_mb"],
        "memory_available_mb": memory["available_mb"],
        "memory_used_mb": memory["used_mb"],
        
        "gpus": gpus,
        
        "cpu_details": cpu,
        "memory_details": memory
    }
