#!/usr/bin/env python3
"""
Integration Tools Configuration
------------------------------
Configuration settings for the Arcanum integration tools.
This file centralizes configuration to avoid hardcoding paths and settings.
"""

import os
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration directory
DEFAULT_CONFIG_DIR = os.path.expanduser("~/.config/arcanum")
CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "integration_config.json")

# Default cache directories
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/arcanum")
DEFAULT_CACHE_DIRS = {
    "google_3d_tiles": os.path.join(DEFAULT_CACHE_DIR, "google_3d_tiles"),
    "street_view": os.path.join(DEFAULT_CACHE_DIR, "street_view"),
    "earth_engine": os.path.join(DEFAULT_CACHE_DIR, "earth_engine"),
    "unity": os.path.join(DEFAULT_CACHE_DIR, "unity"),
    "comfyui": os.path.join(DEFAULT_CACHE_DIR, "comfyui"),
    "storage": os.path.join(DEFAULT_CACHE_DIR, "storage")
}

# Default API configuration
DEFAULT_API_CONFIG = {
    "google_maps_api_key": "",
    "google_earth_api_key": "",
    "google_3d_tiles_api_key": "",
    "base_urls": {
        "google_3d_tiles": "https://tile.googleapis.com/v1/3dtiles",
        "street_view": "https://maps.googleapis.com/maps/api/streetview",
        "earth_engine": "https://earthengine.googleapis.com/v1"
    }
}

# Default storage configuration
DEFAULT_STORAGE_CONFIG = {
    "default_server": "gs://arcanum-3d",
    "server_urls": {
        "gcs": "gs://arcanum-3d",
        "s3": "s3://arcanum-3d",
        "http": "https://arcanum.fortunestold.co/tiles"
    },
    "credentials_paths": {
        "gcs": "",
        "s3": "",
        "http": ""
    }
}

# Default complete configuration
DEFAULT_CONFIG = {
    "cache_dirs": DEFAULT_CACHE_DIRS,
    "api": DEFAULT_API_CONFIG,
    "storage": DEFAULT_STORAGE_CONFIG,
    "retries": 3,
    "timeout": 30
}

def load_config() -> Dict[str, Any]:
    """Load configuration from file or create default if not exists."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logger.info(f"Integration configuration loaded from {CONFIG_FILE}")
            return config
        except Exception as e:
            logger.error(f"Failed to load integration configuration: {str(e)}")
            return DEFAULT_CONFIG.copy()
    else:
        logger.info(f"No integration configuration file found. Creating default at {CONFIG_FILE}")
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file."""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Integration configuration saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save integration configuration: {str(e)}")
        return False

def get_cache_dir(name: str) -> str:
    """Get cache directory for specific integration."""
    config = load_config()
    
    cache_dir = config.get("cache_dirs", {}).get(name, DEFAULT_CACHE_DIRS.get(name))
    
    if not cache_dir:
        cache_dir = os.path.join(DEFAULT_CACHE_DIR, name)
    
    # Ensure the directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    return cache_dir

def get_api_key(service: str) -> Optional[str]:
    """Get API key for specific service."""
    config = load_config()
    
    # Check for service-specific key
    key = config.get("api", {}).get(f"{service}_api_key")
    
    # If no key found, check environment variable
    if not key:
        env_var = f"{service.upper()}_API_KEY"
        key = os.environ.get(env_var)
        
    return key

def get_base_url(service: str) -> str:
    """Get base URL for specific service."""
    config = load_config()
    
    base_url = config.get("api", {}).get("base_urls", {}).get(service)
    
    if not base_url:
        base_url = DEFAULT_API_CONFIG.get("base_urls", {}).get(service)
    
    return base_url

def get_server_url(server_type: str = "default") -> str:
    """Get server URL for specific server type."""
    config = load_config()
    
    if server_type == "default":
        server_url = config.get("storage", {}).get("default_server")
    else:
        server_url = config.get("storage", {}).get("server_urls", {}).get(server_type)
    
    if not server_url:
        server_url = DEFAULT_STORAGE_CONFIG.get("server_urls", {}).get(server_type)
    
    return server_url

def get_credentials_path(server_type: str) -> Optional[str]:
    """Get credentials path for specific server type."""
    config = load_config()
    
    creds_path = config.get("storage", {}).get("credentials_paths", {}).get(server_type)
    
    return creds_path if creds_path else None

# Initialize configuration on module import
if not os.path.exists(CONFIG_FILE):
    for directory in DEFAULT_CACHE_DIRS.values():
        os.makedirs(directory, exist_ok=True)
    save_config(DEFAULT_CONFIG)