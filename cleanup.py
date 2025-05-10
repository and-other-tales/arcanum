#!/usr/bin/env python3
"""
Cleanup Script
------------
This script removes deprecated files from the root directory
that have been moved to the modules structure.
"""

import os
import sys
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Files that have been migrated to modules
deprecated_files = [
    # AI-related files
    "download_flux_models.py",
    "download_flux_upscaler.py",
    
    # Integration files
    "fetch_google_3d_tiles.py",
    "fetch_city_3d_tiles.py",
    "fetch_street_view.py",
    "fetch_street_view_along_roads.py",
    "project_textures.py",
    "verify_coverage.py",
    
    # Storage files
    "storage_manager.py",
    "transform_and_upload.py",
    
    # Utility files
    "authenticate_ee.py",
    "gpu_check.py",
    "gpu_check_env.py",
    
    # Other deprecated files
    "unified_workflow.py",
    "comfyui_automation.py",
    "comfyui_automation_enhanced.py",
    "fixed_generator.py",
    "fixed_graph_from_bbox.py",
    "fixed_graph_from_bbox_large.py",
    "osm_grid_downloader.py",
    "osmnx_config.py"
]

def main():
    """Main function to remove deprecated files."""
    # Get current directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Track deleted files
    deleted_count = 0
    
    # Process each deprecated file
    for filename in deprecated_files:
        file_path = os.path.join(root_dir, filename)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted: {filename}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {filename}: {str(e)}")
        else:
            logger.debug(f"File not found: {filename}")
    
    # Report results
    if deleted_count > 0:
        logger.info(f"Successfully removed {deleted_count} deprecated files")
    else:
        logger.info("No deprecated files found to remove")

if __name__ == "__main__":
    main()