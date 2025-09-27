"""UI module for Hootsight.

Handles pywebview window creation and management.
"""
import os
import webview

from system.log import info, error
from system.coordinator_settings import SETTINGS


def create_window():
    """Create and configure pywebview window."""
    config = SETTINGS.get("ui", {})
    api_config = SETTINGS.get("api", {})
    
    title = config.get("title", "Hootsight")
    width = config.get("width", 1200)
    height = config.get("height", 800)
    resizable = config.get("resizable", True)
    
    # Get API URL to serve UI from FastAPI
    host = api_config.get("host", "127.0.0.1")
    port = api_config.get("port", 8000)
    ui_url = f"http://{host}:{port}"
    
    info(f"Creating UI window: {title} ({width}x{height})")
    info(f"UI URL: {ui_url}")
    
    window = webview.create_window(
        title=title,
        url=ui_url,
        width=width,
        height=height,
        resizable=resizable
    )
    
    return window


def start_ui():
    """Start the pywebview UI."""
    try:
        create_window()
        info("Starting pywebview")
        webview.start()
    except Exception as e:
        error(f"Failed to start UI: {e}")
        raise
