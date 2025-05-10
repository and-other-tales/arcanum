#!/usr/bin/env python3
"""
Earth Engine Authentication Module
---------------------------------
This module handles authentication with Google Earth Engine API.
"""

import os
import sys
import logging
from typing import Optional, Tuple, Dict, Any

# Set up logger
logger = logging.getLogger(__name__)

def get_service_account_key_path() -> str:
    """
    Get the path to the service account key file.
    
    Returns:
        String path to the key file
    """
    # Check environment variable first
    env_key_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if env_key_path and os.path.exists(env_key_path):
        return env_key_path
        
    # Then check default locations
    default_paths = [
        os.path.expanduser('~/arcanum/key.json'),
        './key.json',
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'key.json')
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            return path
            
    # Return the default path even if it doesn't exist
    return default_paths[0]

def initialize_ee(key_file: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Initialize Earth Engine with service account.
    
    Args:
        key_file: Path to service account key file (optional)
        
    Returns:
        Tuple containing:
        - Boolean indicating initialization success
        - Dictionary with information about the initialization
    """
    info = {}
    
    try:
        import ee
        info["ee_version"] = ee.__version__
    except ImportError as e:
        info["error"] = f"Earth Engine API import error: {e}"
        logger.error(f"Failed to import Earth Engine API: {e}")
        return False, info
    
    # Determine key file path
    key_path = key_file or get_service_account_key_path()
    info["key_file"] = key_path
    
    if not os.path.exists(key_path):
        info["error"] = f"Service account key file not found: {key_path}"
        logger.error(f"Service account key file not found: {key_path}")
        return False, info
    
    try:
        # Create credentials
        credentials = ee.ServiceAccountCredentials(
            email=None,  # Will be read from the key file
            key_file=key_path
        )
        
        # Initialize Earth Engine
        ee.Initialize(credentials)
        
        # Test the connection
        info["initialized"] = True
        
        try:
            # Simple test to ensure we're actually connected
            image = ee.Image('USGS/SRTMGL1_003')
            _ = image.getInfo()
            info["connection_test"] = "success"
        except Exception as e:
            info["connection_test"] = "failed"
            info["connection_error"] = str(e)
            logger.warning(f"Earth Engine initialized, but connection test failed: {e}")
        
        logger.info("Google Earth Engine initialized successfully with service account.")
        return True, info
        
    except Exception as e:
        info["error"] = str(e)
        logger.error(f"Error initializing Earth Engine: {e}")
        return False, info

def main() -> int:
    """
    Main function to initialize Earth Engine and return status code.
    
    Returns:
        0 if initialization successful, non-zero otherwise
    """
    success, info = initialize_ee()
    
    if success:
        print("Google Earth Engine initialized successfully with service account.")
        if info.get("connection_test") == "success":
            print("Connection test successful.")
        else:
            print(f"Connection test failed: {info.get('connection_error', 'Unknown error')}")
            return 2
        return 0
    else:
        print(f"Error initializing Earth Engine: {info.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())