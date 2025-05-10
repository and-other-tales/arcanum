#!/usr/bin/env python3
"""
Google 3D Tiles Integration Module
---------------------------------
This module provides integration with Google Maps Platform's Photorealistic 3D Tiles API,
allowing for streaming and processing of photorealistic 3D tiles within the Arcanum system.
"""

import os
import sys
import logging
import json
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import time
from urllib.parse import urlparse, urlencode, quote

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class Google3DTilesIntegration:
    """Class that provides integration with Google Maps Platform's Photorealistic 3D Tiles API."""
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = "https://tile.googleapis.com",
                 cache_dir: str = None,
                 retries: int = 3,
                 timeout: int = 30):
        """Initialize the Google 3D Tiles integration.
        
        Args:
            api_key: Google Maps API key with 3D Tiles access (from env var if None)
            base_url: Base URL for the Google 3D Tiles API
            cache_dir: Directory to cache downloaded tiles
            retries: Number of retries for failed requests
            timeout: Timeout in seconds for requests
        """
        # Load API key from environment if not provided
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            logger.warning("No Google Maps API key provided. Set GOOGLE_MAPS_API_KEY environment variable.")
        
        self.base_url = base_url
        self.retries = retries
        self.timeout = timeout
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.path.expanduser("~/.cache/arcanum/google_3d_tiles"))
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Google 3D Tiles cache directory: {self.cache_dir}")
        
        # API endpoints
        self.tileset_endpoint = "/v1/3dtiles/features/basic"
        self.metadata_endpoint = "/v1/3dtiles/features/basic/metadata"
        
        logger.info("Google 3D Tiles Integration initialized")
    
    def get_tileset_url(self, region: str = None) -> str:
        """Get the tileset.json URL for a region.
        
        Args:
            region: Optional region name (default global tileset)
            
        Returns:
            URL for tileset.json with API key included
        """
        # Build the tileset URL for the region
        url = f"{self.base_url}{self.tileset_endpoint}/tileset.json"
        
        # Add API key
        params = {"key": self.api_key}
        if region:
            params["region"] = region
            
        # Append parameters
        if params:
            url = f"{url}?{urlencode(params)}"
            
        return url
    
    def get_metadata_url(self) -> str:
        """Get the metadata URL for the 3D Tiles service.
        
        Returns:
            URL for metadata with API key included
        """
        url = f"{self.base_url}{self.metadata_endpoint}"
        params = {"key": self.api_key}
        
        # Append parameters
        if params:
            url = f"{url}?{urlencode(params)}"
            
        return url
    
    def get_tile_url(self, tile_path: str) -> str:
        """Get a full URL for a specific tile.
        
        Args:
            tile_path: Relative path to the tile
            
        Returns:
            Full URL for the tile with API key included
        """
        # Ensure the tile path doesn't start with a slash
        if tile_path.startswith("/"):
            tile_path = tile_path[1:]
        
        # Build the full URL
        url = f"{self.base_url}/{self.tileset_endpoint}/{tile_path}"
        
        # Add API key
        params = {"key": self.api_key}
        
        # Append parameters
        if params:
            url = f"{url}?{urlencode(params)}"
            
        return url
        
    def fetch_tileset_json(self, region: str = None) -> Dict:
        """Fetch the tileset.json from Google 3D Tiles API.
        
        Args:
            region: Optional region name (default global tileset)
            
        Returns:
            Dictionary containing the tileset.json content
        """
        url = self.get_tileset_url(region)
        logger.info(f"Fetching tileset.json from {url}")
        
        # Try to fetch with retries
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Parse JSON
                tileset_json = response.json()
                
                # Cache the tileset.json
                cache_path = self.cache_dir / f"tileset_{region or 'global'}.json"
                with open(cache_path, "w") as f:
                    json.dump(tileset_json, f, indent=2)
                
                logger.info(f"Successfully fetched tileset.json (cached at {cache_path})")
                return tileset_json
                
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed: {str(e)}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
        
        logger.error(f"Failed to fetch tileset.json after {self.retries} attempts.")
        raise Exception(f"Failed to fetch tileset.json after {self.retries} attempts.")
    
    def fetch_metadata(self) -> Dict:
        """Fetch the 3D Tiles metadata from Google Maps API.
        
        Returns:
            Dictionary containing the metadata
        """
        url = self.get_metadata_url()
        logger.info(f"Fetching 3D tiles metadata from {url}")
        
        # Try to fetch with retries
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Parse JSON
                metadata = response.json()
                
                # Cache the metadata
                cache_path = self.cache_dir / "metadata.json"
                with open(cache_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Successfully fetched metadata (cached at {cache_path})")
                return metadata
                
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed: {str(e)}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
        
        logger.error(f"Failed to fetch metadata after {self.retries} attempts.")
        raise Exception(f"Failed to fetch metadata after {self.retries} attempts.")
    
    def fetch_tile(self, tile_path: str, output_path: str = None) -> Optional[str]:
        """Fetch a specific tile from the Google 3D Tiles API.
        
        Args:
            tile_path: Relative path to the tile 
            output_path: Path to save the tile (if None, uses cache directory)
            
        Returns:
            Path to the downloaded tile, or None if failed
        """
        url = self.get_tile_url(tile_path)
        
        # Determine output path if not provided
        if not output_path:
            # Generate a clean filename from the tile path
            filename = Path(tile_path).name
            output_path = self.cache_dir / filename
        else:
            output_path = Path(output_path)
            
        logger.info(f"Fetching tile from {url} to {output_path}")
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to fetch with retries
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Save the tile
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded tile to {output_path}")
                return str(output_path)
                
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed: {str(e)}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
        
        logger.error(f"Failed to fetch tile after {self.retries} attempts.")
        return None
    
    def fetch_tiles_recursive(self, tileset_json: Dict, max_depth: int = 3, 
                             output_dir: str = None, root_path: str = None) -> List[str]:
        """Recursively fetch all tiles referenced in a tileset.json.
        
        Args:
            tileset_json: The tileset JSON object
            max_depth: Maximum recursion depth
            output_dir: Directory to save tiles
            root_path: Root path for relative URLs
            
        Returns:
            List of paths to downloaded tiles
        """
        downloaded_paths = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Helper function to process a tile
        def process_tile(tile, current_depth=0, parent_path=None):
            # Stop if we've reached max depth
            if current_depth > max_depth:
                return
            
            # Check if the tile has content to download
            if "content" in tile and "uri" in tile["content"]:
                uri = tile["content"]["uri"]
                
                # Handle relative URLs
                if not uri.startswith("http"):
                    # If we have a root path, join it with the URI
                    if parent_path:
                        # Remove filename from parent path to get directory
                        parent_dir = os.path.dirname(parent_path)
                        full_path = os.path.join(parent_dir, uri)
                    elif root_path:
                        full_path = os.path.join(root_path, uri)
                    else:
                        full_path = uri
                        
                    # Clean up path
                    full_path = os.path.normpath(full_path)
                else:
                    # For absolute URLs, we need to extract the path
                    parsed_url = urlparse(uri)
                    full_path = parsed_url.path
                    
                    # Remove leading /v1/3dtiles/features/basic if present
                    if full_path.startswith(self.tileset_endpoint):
                        full_path = full_path[len(self.tileset_endpoint):]
                    
                    # Remove leading slash
                    if full_path.startswith("/"):
                        full_path = full_path[1:]
                
                # Download the tile
                if output_dir:
                    tile_output_path = output_dir / full_path
                else:
                    tile_output_path = None
                    
                downloaded_path = self.fetch_tile(full_path, tile_output_path)
                
                if downloaded_path:
                    downloaded_paths.append(downloaded_path)
                    
                    # If the tile is a tileset, load it and process recursively
                    if uri.endswith(".json"):
                        try:
                            with open(downloaded_path, "r") as f:
                                subtileset = json.load(f)
                                
                            # Process the root tile of the subtileset
                            if "root" in subtileset:
                                process_tile(subtileset["root"], current_depth + 1, full_path)
                        except Exception as e:
                            logger.error(f"Error processing subtileset {downloaded_path}: {str(e)}")
            
            # Process child tiles
            if "children" in tile:
                for child in tile["children"]:
                    process_tile(child, current_depth + 1, parent_path)
        
        # Start with the root tile
        if "root" in tileset_json:
            process_tile(tileset_json["root"])
        
        return downloaded_paths
    
    def stream_tiles_to_storage(self, region: str = None, 
                              max_depth: int = 3, 
                              storage_manager = None) -> Dict:
        """Stream 3D tiles from Google Maps to Arcanum storage.
        
        Args:
            region: Optional region name (default global tileset)
            max_depth: Maximum recursion depth for tile fetching
            storage_manager: Optional ArcanumStorageManager instance
            
        Returns:
            Dictionary with streaming results
        """
        try:
            # First fetch the tileset.json
            tileset_json = self.fetch_tileset_json(region)
            
            # Create a temporary directory for tiles
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="arcanum_3d_tiles_")
            
            # Fetch all tiles recursively
            downloaded_paths = self.fetch_tiles_recursive(
                tileset_json, 
                max_depth=max_depth, 
                output_dir=temp_dir
            )
            
            results = {
                "tileset": tileset_json,
                "downloaded_tiles": len(downloaded_paths),
                "temp_dir": temp_dir
            }
            
            # If we have a storage manager, upload to it
            if storage_manager:
                # Import here to avoid circular imports
                from storage_manager import ArcanumStorageManager
                
                if not isinstance(storage_manager, ArcanumStorageManager):
                    logger.warning("Provided storage_manager is not an ArcanumStorageManager instance")
                    return results
                
                # Upload all tiles to storage
                uploaded_files = []
                
                # First upload the tileset.json
                tileset_path = os.path.join(temp_dir, "tileset.json")
                with open(tileset_path, "w") as f:
                    json.dump(tileset_json, f, indent=2)
                
                tileset_gcs_path = f"tilesets/google_3d_tiles_{region or 'global'}/tileset.json"
                cdn_url = storage_manager.upload_to_gcs(
                    tileset_path,
                    tileset_gcs_path,
                    content_type="application/json",
                    cache_control="public, max-age=3600"
                )
                
                if cdn_url:
                    uploaded_files.append({
                        "local_path": tileset_path,
                        "gcs_path": tileset_gcs_path,
                        "cdn_url": cdn_url
                    })
                
                # Upload all downloaded tiles
                for local_path in downloaded_paths:
                    # Determine relative path from temp_dir
                    rel_path = os.path.relpath(local_path, temp_dir)
                    
                    # Create GCS path
                    gcs_path = f"tilesets/google_3d_tiles_{region or 'global'}/{rel_path}"
                    
                    # Determine content type
                    content_type = storage_manager._get_content_type(os.path.splitext(local_path)[1])
                    
                    # Upload to GCS
                    cdn_url = storage_manager.upload_to_gcs(
                        local_path,
                        gcs_path,
                        content_type=content_type,
                        cache_control="public, max-age=86400"
                    )
                    
                    if cdn_url:
                        uploaded_files.append({
                            "local_path": local_path,
                            "gcs_path": gcs_path,
                            "cdn_url": cdn_url
                        })
                
                results["uploaded_files"] = uploaded_files
                results["upload_success"] = len(uploaded_files)
                
                logger.info(f"Uploaded {len(uploaded_files)} files to GCS")
                
                # Clean up temporary directory
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
                results["temp_dir_cleaned"] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Error streaming 3D tiles: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

def main():
    """Main function for testing the module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Google 3D Tiles Integration")
    parser.add_argument("--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    parser.add_argument("--region", help="Region for tileset (default global)")
    parser.add_argument("--fetch", help="Fetch a specific tile path")
    parser.add_argument("--output", help="Output path for fetched tile")
    parser.add_argument("--fetch-tileset", action="store_true", help="Fetch tileset.json")
    parser.add_argument("--fetch-metadata", action="store_true", help="Fetch metadata")
    parser.add_argument("--stream", action="store_true", help="Stream tiles to storage")
    parser.add_argument("--depth", type=int, default=2, help="Max depth for recursive fetching")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize integration
    tiles_integration = Google3DTilesIntegration(api_key=args.api_key)
    
    if args.test:
        logger.info("Running basic test...")
        
        # Test metadata fetching
        try:
            metadata = tiles_integration.fetch_metadata()
            logger.info(f"Metadata fetched successfully: {json.dumps(metadata, indent=2)[:500]}...")
        except Exception as e:
            logger.error(f"Error fetching metadata: {str(e)}")
        
        # Test tileset fetching
        try:
            tileset_json = tiles_integration.fetch_tileset_json(args.region)
            logger.info(f"Tileset.json fetched successfully: {json.dumps(tileset_json, indent=2)[:500]}...")
        except Exception as e:
            logger.error(f"Error fetching tileset.json: {str(e)}")
    
    if args.fetch_metadata:
        try:
            metadata = tiles_integration.fetch_metadata()
            print(json.dumps(metadata, indent=2))
        except Exception as e:
            logger.error(f"Error fetching metadata: {str(e)}")
            return 1
    
    if args.fetch_tileset:
        try:
            tileset_json = tiles_integration.fetch_tileset_json(args.region)
            print(json.dumps(tileset_json, indent=2))
        except Exception as e:
            logger.error(f"Error fetching tileset.json: {str(e)}")
            return 1
    
    if args.fetch:
        output_path = args.output
        try:
            path = tiles_integration.fetch_tile(args.fetch, output_path)
            if path:
                logger.info(f"Successfully fetched tile to {path}")
            else:
                logger.error("Failed to fetch tile")
                return 1
        except Exception as e:
            logger.error(f"Error fetching tile: {str(e)}")
            return 1
    
    if args.stream:
        try:
            # Conditionally import storage manager
            try:
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from storage_manager import ArcanumStorageManager
                storage_manager = ArcanumStorageManager()
            except ImportError:
                logger.warning("ArcanumStorageManager not available, streaming to local only")
                storage_manager = None
            
            results = tiles_integration.stream_tiles_to_storage(
                region=args.region,
                max_depth=args.depth,
                storage_manager=storage_manager
            )
            
            print(json.dumps(results, indent=2))
        except Exception as e:
            logger.error(f"Error streaming tiles: {str(e)}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())