"""Entry point for Hootsight system.

Starts FastAPI server in background thread and opens UI in pywebview window.
"""
import sys
import os
import threading
import time
import socket
import uvicorn

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from system.api import create_app
from system.ui import start_ui
from system.log import info, success, error
from system.coordinator_settings import SETTINGS


def is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


def start_api_server():
    """Start FastAPI server in background thread."""
    config = SETTINGS.get("api", {})
    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8000)
    debug = bool(config.get("debug", False))

    # Check if port is already in use
    if is_port_in_use(host, port):
        error(f"Port {port} is already in use on {host}. System will exit.")
        sys.exit(1)

    info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level=("debug" if debug else "error"))


def main():
    """Main entry point."""
    info("Starting Hootsight system")
    
    # Start API server in background
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # Wait for API to initialize (configurable)
    wait_seconds = int(SETTINGS.get("system", {}).get("startup_wait_seconds", 2))
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    success("API server started")
    
    # Start UI
    start_ui()


if __name__ == "__main__":
    main()
