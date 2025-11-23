import os
import webview

from system.log import info, error
from system.coordinator_settings import SETTINGS


def create_window():
    config = SETTINGS['ui']
    api_config = SETTINGS['api']
    
    title = config['title']
    width = config['width']
    height = config['height']
    resizable = config['resizable']
    
    host = api_config['host']
    port = api_config['port']
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
    try:
        create_window()
        info("Starting pywebview")
        webview.start()
    except Exception as e:
        error(f"Failed to start UI: {e}")
        raise
