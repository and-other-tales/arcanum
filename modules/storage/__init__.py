#!/usr/bin/env python3
"""
Arcanum Storage Module
---------------------
This module handles storage-related functionality for the Arcanum system.
"""

# Import storage functions and classes for direct access
from modules.storage.storage_integration import (
    transform_and_upload_image as upload_image,
    transform_and_upload_directory as upload_dir,
    transform_satellite_images as upload_satellite,
    ArcanumStorageIntegration
)

from modules.storage.storage import ArcanumStorageManager
from modules.storage.uploader import transform_and_upload as upload

__all__ = [
    'upload_image',
    'upload_dir',
    'upload_satellite',
    'ArcanumStorageIntegration',
    'ArcanumStorageManager',
    'upload'
]