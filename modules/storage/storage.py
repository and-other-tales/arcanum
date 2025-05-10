#!/usr/bin/env python3
"""
Arcanum Storage Manager
----------------------
This module handles efficient storage management for the Arcanum generator, 
implementing Google Cloud Storage integration and optimizing local storage usage.
"""

import os
import sys
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Try to import Google Cloud Storage
try:
    from google.cloud import storage
    from google.api_core.exceptions import NotFound
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("Google Cloud Storage library not available. Install with: pip install google-cloud-storage")

class ArcanumStorageManager:
    """Manages storage for Arcanum generator with Google Cloud Storage integration."""
    
    def __init__(self, 
                 local_root_dir: str = "./arcanum_output",
                 gcs_bucket_name: str = "arcanum-maps",
                 cdn_url: str = "https://arcanum.fortunestold.co",
                 cleanup_originals: bool = True,
                 max_local_cache_size_gb: float = 10.0):
        """Initialize the storage manager.
        
        Args:
            local_root_dir: Local directory for Arcanum output files
            gcs_bucket_name: Google Cloud Storage bucket name
            cdn_url: Base URL for the CDN that serves the GCS content
            cleanup_originals: Whether to delete original files after transformation and upload
            max_local_cache_size_gb: Maximum size of local cache in GB
        """
        self.local_root = os.path.abspath(os.path.expanduser(local_root_dir))
        self.gcs_bucket_name = gcs_bucket_name
        self.cdn_url = cdn_url
        self.cleanup_originals = cleanup_originals
        self.max_local_cache_size_bytes = max_local_cache_size_gb * 1024 * 1024 * 1024
        
        # Create local storage directories
        self._ensure_directories()
        
        # Initialize Google Cloud Storage client if available
        self.gcs_client = None
        self.gcs_bucket = None
        if GCS_AVAILABLE:
            try:
                self.gcs_client = storage.Client()
                try:
                    self.gcs_bucket = self.gcs_client.get_bucket(gcs_bucket_name)
                    logger.info(f"Connected to GCS bucket: {gcs_bucket_name}")
                except NotFound:
                    logger.warning(f"GCS bucket not found: {gcs_bucket_name}")
                    logger.warning("Will attempt to create bucket or continue with local storage only")
                    try:
                        self.gcs_bucket = self.gcs_client.create_bucket(gcs_bucket_name)
                        logger.info(f"Created GCS bucket: {gcs_bucket_name}")
                    except Exception as e:
                        logger.error(f"Failed to create GCS bucket: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to initialize Google Cloud Storage client: {str(e)}")
                logger.warning("Continuing with local storage only")
    
    def _ensure_directories(self):
        """Ensure all required local directories exist."""
        # Main directories
        os.makedirs(self.local_root, exist_ok=True)
        
        # Create specific subdirectories
        subdirs = [
            "raw_data",           # Original unprocessed data
            "processed_data",      # Processed data
            "final",              # Final output data
            "temp",               # Temporary storage
            "cache"               # Cache for frequently accessed files
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.local_root, subdir), exist_ok=True)
    
    def check_local_cache_size(self) -> float:
        """Check the current size of the local cache in bytes.
        
        Returns:
            The size of the cache in bytes
        """
        cache_dir = os.path.join(self.local_root, "cache")
        temp_dir = os.path.join(self.local_root, "temp")
        
        total_size = 0
        
        # Calculate cache size
        for dirpath, _, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        # Add temp size
        for dirpath, _, filenames in os.walk(temp_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        return total_size
    
    def clean_cache_if_needed(self, reserve_bytes: int = 0) -> bool:
        """Clean the cache if it exceeds the maximum size.
        
        Args:
            reserve_bytes: Additional space to reserve beyond current usage
            
        Returns:
            True if cleaning was performed, False otherwise
        """
        current_size = self.check_local_cache_size()
        if current_size + reserve_bytes > self.max_local_cache_size_bytes:
            logger.info(f"Cache size ({current_size/1024/1024:.2f} MB) exceeds limit. Cleaning...")
            
            # Get all files in cache with their modification times
            cache_dir = os.path.join(self.local_root, "cache")
            temp_dir = os.path.join(self.local_root, "temp")
            
            files_with_times = []
            
            # Collect cache files
            for dirpath, _, filenames in os.walk(cache_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    mtime = os.path.getmtime(filepath)
                    size = os.path.getsize(filepath)
                    files_with_times.append((filepath, mtime, size))
            
            # Collect temp files older than 1 day
            one_day_ago = time.time() - (24 * 60 * 60)
            for dirpath, _, filenames in os.walk(temp_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    mtime = os.path.getmtime(filepath)
                    if mtime < one_day_ago:
                        size = os.path.getsize(filepath)
                        files_with_times.append((filepath, mtime, size))
            
            # Sort by modification time (oldest first)
            files_with_times.sort(key=lambda x: x[1])
            
            # Calculate how much space we need to free
            space_to_free = current_size + reserve_bytes - self.max_local_cache_size_bytes
            space_freed = 0
            
            # Delete files until we've freed enough space
            for filepath, _, size in files_with_times:
                try:
                    os.remove(filepath)
                    space_freed += size
                    logger.debug(f"Removed cached file: {filepath} ({size/1024/1024:.2f} MB)")
                    
                    if space_freed >= space_to_free:
                        break
                except Exception as e:
                    logger.warning(f"Failed to remove cached file {filepath}: {str(e)}")
            
            logger.info(f"Cleaned {space_freed/1024/1024:.2f} MB from cache")
            return True
        
        return False
    
    def store_file_locally(self, 
                         source_path: str, 
                         category: str, 
                         filename: Optional[str] = None,
                         subfolder: Optional[str] = None) -> str:
        """Store a file in the local storage system.
        
        Args:
            source_path: Path to the source file
            category: Category of data (raw_data, processed_data, final)
            filename: Optional custom filename (uses source basename if None)
            subfolder: Optional subfolder within the category
            
        Returns:
            Path to the stored file
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        # Determine destination path
        if filename is None:
            filename = os.path.basename(source_path)
        
        category_dir = os.path.join(self.local_root, category)
        if subfolder:
            category_dir = os.path.join(category_dir, subfolder)
        
        os.makedirs(category_dir, exist_ok=True)
        dest_path = os.path.join(category_dir, filename)
        
        # Reserve enough space in cache if needed
        file_size = os.path.getsize(source_path)
        if category in ["cache", "temp"]:
            self.clean_cache_if_needed(file_size)
        
        # Copy the file
        shutil.copy2(source_path, dest_path)
        logger.debug(f"Stored {source_path} to {dest_path}")
        
        return dest_path
    
    def upload_to_gcs(self, 
                     local_path: str, 
                     gcs_path: str,
                     content_type: Optional[str] = None,
                     cache_control: str = "public, max-age=86400") -> Optional[str]:
        """Upload a file to Google Cloud Storage.
        
        Args:
            local_path: Path to the local file
            gcs_path: Path in GCS (without bucket name)
            content_type: Optional content type (auto-detected if None)
            cache_control: Cache control header
            
        Returns:
            GCS URL if successful, None otherwise
        """
        if not self.gcs_bucket:
            logger.warning("GCS bucket not available. Skipping upload.")
            return None
        
        if not os.path.exists(local_path):
            logger.error(f"Local file not found: {local_path}")
            return None
        
        try:
            # Create a blob
            blob = self.gcs_bucket.blob(gcs_path)
            
            # Set blob properties
            if content_type:
                blob.content_type = content_type
            
            # Set cache control
            blob.cache_control = cache_control
            
            # Upload the file
            logger.info(f"Uploading {local_path} to gs://{self.gcs_bucket_name}/{gcs_path}")
            blob.upload_from_filename(local_path)
            
            # Return the public URL
            gcs_url = f"gs://{self.gcs_bucket_name}/{gcs_path}"
            cdn_url = f"{self.cdn_url}/{gcs_path}"
            
            logger.info(f"File uploaded successfully: {gcs_url}")
            logger.info(f"CDN URL: {cdn_url}")
            
            return cdn_url
        
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {str(e)}")
            return None
    
    def process_tile(self,
                   source_path: str,
                   tile_type: str,
                   x: int,
                   y: int,
                   z: int,
                   tileset_name: str = "arcanum",
                   transform_function = None,
                   delete_original: bool = None) -> Dict:
        """Process a tile and upload it to GCS with proper OGC 3D Tiles structure.
        
        Args:
            source_path: Path to the source tile file
            tile_type: Type of tile (b3dm, pnts, etc.)
            x: X coordinate
            y: Y coordinate
            z: Z/level coordinate
            tileset_name: Name of the tileset
            transform_function: Optional function to transform the tile before upload
            delete_original: Whether to delete the original file (overrides global setting)
            
        Returns:
            Dictionary with information about the processed tile
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source tile not found: {source_path}")
        
        # Determine if we should delete the original
        if delete_original is None:
            delete_original = self.cleanup_originals
        
        # Construct the OGC 3D Tiles path
        gcs_path = f"tilesets/{tileset_name}/tiles/{z}/{x}/{y}.{tile_type}"
        
        # Apply transformation if provided
        transformed_path = source_path
        if transform_function:
            try:
                # Create a temporary directory for the transformed file
                temp_dir = os.path.join(self.local_root, "temp")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Generate temp filename
                temp_filename = f"transform_{os.path.basename(source_path)}"
                transformed_path = os.path.join(temp_dir, temp_filename)
                
                # Apply the transformation
                logger.info(f"Transforming tile: {source_path}")
                transform_function(source_path, transformed_path)
                
                if not os.path.exists(transformed_path):
                    logger.error("Transformation failed to produce output file")
                    transformed_path = source_path
            except Exception as e:
                logger.error(f"Error during tile transformation: {str(e)}")
                transformed_path = source_path
        
        # Upload to GCS
        content_type = self._get_content_type(tile_type)
        cdn_url = self.upload_to_gcs(
            transformed_path,
            gcs_path,
            content_type=content_type,
            cache_control="public, max-age=86400"
        )
        
        # Clean up if requested
        if delete_original and transformed_path != source_path:
            try:
                # Delete the original file only if transformed path is different
                logger.info(f"Deleting original file: {source_path}")
                os.remove(source_path)
            except Exception as e:
                logger.warning(f"Failed to delete original file: {str(e)}")
        
        # Clean up transformed file if it's in the temp directory
        if transformed_path != source_path and transformed_path.startswith(os.path.join(self.local_root, "temp")):
            try:
                logger.debug(f"Cleaning up temporary transformed file: {transformed_path}")
                os.remove(transformed_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")
        
        # Return information about the processed tile
        return {
            "original_path": source_path,
            "transformed_path": transformed_path if transformed_path != source_path else None,
            "gcs_path": gcs_path,
            "cdn_url": cdn_url,
            "tile_type": tile_type,
            "coordinates": {
                "x": x,
                "y": y,
                "z": z
            },
            "tileset": tileset_name,
            "success": cdn_url is not None
        }
    
    def process_image_batch(self,
                          source_dir: str,
                          target_tileset: str,
                          transform_function = None,
                          file_pattern: str = "*.jpg",
                          z_level: int = 0,
                          delete_originals: bool = None) -> List[Dict]:
        """Process a batch of images as tiles.
        
        Args:
            source_dir: Directory containing source images
            target_tileset: Target tileset name
            transform_function: Function to transform images
            file_pattern: Pattern to match files
            z_level: Z-level for the tiles
            delete_originals: Whether to delete original files
            
        Returns:
            List of dictionaries with information about processed tiles
        """
        import glob
        from itertools import count
        
        results = []
        
        # Find all matching files
        file_paths = glob.glob(os.path.join(source_dir, file_pattern))
        logger.info(f"Found {len(file_paths)} files matching {file_pattern} in {source_dir}")
        
        # Process each file
        for i, file_path in enumerate(file_paths):
            try:
                # Extract x,y coordinates from filename or use counter
                filename = os.path.basename(file_path)
                x, y = self._extract_coordinates_from_filename(filename, i)
                
                # Determine tile type from file extension
                _, ext = os.path.splitext(file_path)
                tile_type = ext[1:].lower()  # Remove dot from extension
                
                # Process the tile
                result = self.process_tile(
                    source_path=file_path,
                    tile_type=tile_type,
                    x=x,
                    y=y,
                    z=z_level,
                    tileset_name=target_tileset,
                    transform_function=transform_function,
                    delete_original=delete_originals
                )
                
                results.append(result)
                logger.info(f"Processed tile {i+1}/{len(file_paths)}: {result['cdn_url'] if result['success'] else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                results.append({
                    "original_path": file_path,
                    "error": str(e),
                    "success": False
                })
        
        # Create tileset.json if it doesn't exist
        self._ensure_tileset_json(target_tileset, results)
        
        return results
    
    def _extract_coordinates_from_filename(self, filename: str, default_index: int) -> Tuple[int, int]:
        """Extract x,y coordinates from filename or use a default mapping.
        
        Args:
            filename: The filename to parse
            default_index: Default index if coordinates can't be extracted
            
        Returns:
            Tuple of (x, y) coordinates
        """
        import re
        
        # Try to extract coordinates from standard patterns
        # Pattern like "tile_x_y.ext" or "x_y.ext"
        match = re.search(r'(?:tile_)?(\d+)_(\d+)', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # Pattern like "x-y.ext"
        match = re.search(r'(\d+)-(\d+)', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        
        # If no pattern matched, use modulo for a grid-like arrangement
        # This allows handling arbitrary filenames by arranging them in a grid
        grid_size = max(1, int(default_index**0.5) + 1)  # Square grid
        x = default_index % grid_size
        y = default_index // grid_size
        
        return x, y
    
    def _ensure_tileset_json(self, tileset_name: str, tiles: List[Dict]) -> Optional[str]:
        """Ensure the tileset.json file exists, creating it if needed.
        
        Args:
            tileset_name: Name of the tileset
            tiles: List of processed tiles
            
        Returns:
            CDN URL of the tileset.json file
        """
        if not self.gcs_bucket:
            logger.warning("GCS bucket not available. Skipping tileset.json creation.")
            return None
        
        # Check if tileset.json already exists
        try:
            tileset_path = f"tilesets/{tileset_name}/tileset.json"
            blob = self.gcs_bucket.blob(tileset_path)
            
            if blob.exists():
                logger.info(f"Tileset.json already exists for {tileset_name}")
                return f"{self.cdn_url}/{tileset_path}"
        except Exception as e:
            logger.error(f"Error checking for existing tileset.json: {str(e)}")
        
        # Create a basic tileset.json
        if tiles:
            # Get unique z-levels
            z_levels = set(tile['coordinates']['z'] for tile in tiles if 'coordinates' in tile)
            
            # Create a simple tileset.json
            tileset_json = {
                "asset": {
                    "version": "1.0"
                },
                "geometricError": 500,
                "root": {
                    "boundingVolume": {
                        "region": [
                            -3.14159, -1.5708, 3.14159, 1.5708, 0, 10000
                        ]
                    },
                    "geometricError": 500,
                    "refine": "ADD",
                    "children": []
                }
            }
            
            # Add root tiles for each z-level
            for z in z_levels:
                z_tiles = [t for t in tiles if 'coordinates' in t and t['coordinates']['z'] == z]
                
                if z_tiles:
                    # Simple child node for this level
                    child = {
                        "boundingVolume": {
                            "region": [
                                -3.14159, -1.5708, 3.14159, 1.5708, 0, 10000
                            ]
                        },
                        "geometricError": 500 / (z + 1),
                        "refine": "ADD",
                        "content": {
                            "uri": f"tiles/{z}/0/0.{z_tiles[0]['tile_type']}"
                        }
                    }
                    
                    tileset_json["root"]["children"].append(child)
            
            # Write tileset.json to a temporary file
            temp_dir = os.path.join(self.local_root, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"{tileset_name}_tileset.json")
            
            with open(temp_path, 'w') as f:
                json.dump(tileset_json, f, indent=2)
            
            # Upload to GCS
            cdn_url = self.upload_to_gcs(
                temp_path,
                f"tilesets/{tileset_name}/tileset.json",
                content_type="application/json",
                cache_control="public, max-age=3600"
            )
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary tileset.json: {str(e)}")
            
            return cdn_url
        
        return None
    
    def _get_content_type(self, file_extension: str) -> str:
        """Determine the content type based on file extension.
        
        Args:
            file_extension: File extension without the dot
            
        Returns:
            Content type string
        """
        # Content types for different tile formats
        content_types = {
            # 3D Tiles formats
            'b3dm': 'application/octet-stream',
            'pnts': 'application/octet-stream',
            'i3dm': 'application/octet-stream',
            'cmpt': 'application/octet-stream',
            
            # Images
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            
            # JSON
            'json': 'application/json',
            
            # Other formats
            'tif': 'image/tiff',
            'tiff': 'image/tiff',
            'obj': 'text/plain',
            'mtl': 'text/plain',
            'gltf': 'model/gltf+json',
            'glb': 'model/gltf-binary'
        }
        
        extension = file_extension.lower().strip('.')
        return content_types.get(extension, 'application/octet-stream')

def main():
    """Main function for testing the module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Arcanum Storage Manager")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--local-dir", default="./arcanum_output", help="Local storage directory")
    parser.add_argument("--gcs-bucket", default="arcanum-maps", help="GCS bucket name")
    parser.add_argument("--cdn-url", default="https://arcanum.fortunestold.co", help="CDN URL")
    args = parser.parse_args()
    
    if args.test:
        # Simple test
        storage_manager = ArcanumStorageManager(
            local_root_dir=args.local_dir,
            gcs_bucket_name=args.gcs_bucket,
            cdn_url=args.cdn_url
        )
        
        print("Storage manager initialized.")
        print(f"Local root: {storage_manager.local_root}")
        print(f"GCS bucket: {storage_manager.gcs_bucket_name}")
        print(f"GCS available: {GCS_AVAILABLE}")
        print(f"GCS client initialized: {storage_manager.gcs_client is not None}")
        print(f"GCS bucket initialized: {storage_manager.gcs_bucket is not None}")
        
        # Check cache size
        cache_size = storage_manager.check_local_cache_size()
        print(f"Current cache size: {cache_size/1024/1024:.2f} MB")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())