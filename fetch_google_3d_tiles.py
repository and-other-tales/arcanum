#!/usr/bin/env python3
"""
Google 3D Tiles Fetcher
----------------------
This script provides a command-line interface for fetching, processing, and integrating
Google Maps Platform's Photorealistic 3D Tiles with the Arcanum system.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from integration_tools.google_3d_tiles_integration import Google3DTilesIntegration
from storage_manager import ArcanumStorageManager
from integration_tools.storage_integration import ArcanumStorageIntegration

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arcanum_3d_tiles.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Google 3D Tiles Fetcher for Arcanum")
    
    # API Key options
    parser.add_argument("--api-key", help="Google Maps API key (overrides environment variable)")
    
    # Region and location options
    parser.add_argument("--region", default="us",
                        help="Region name for tileset (default: us)")
    parser.add_argument("--lat", type=float, help="Latitude of target location")
    parser.add_argument("--lon", type=float, help="Longitude of target location")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Radius in kilometers around location (default: 1.0)")
    
    # Tile fetching options
    parser.add_argument("--depth", type=int, default=3,
                        help="Maximum recursion depth for tile fetching (default: 3)")
    parser.add_argument("--max-tiles", type=int, default=1000,
                        help="Maximum number of tiles to fetch (default: 1000)")
    parser.add_argument("--format", choices=["glb", "b3dm", "pnts", "all"], 
                        default="all", help="Tile format to fetch (default: all)")
    
    # Output options
    parser.add_argument("--output-dir", default="./arcanum_3d_output",
                        help="Local output directory for downloaded tiles")
    parser.add_argument("--gcs-bucket", default="arcanum-maps",
                        help="GCS bucket name for uploading (default: arcanum-maps)")
    parser.add_argument("--cdn-url", default="https://arcanum.fortunestold.co",
                        help="CDN URL for accessing tiles (default: arcanum.fortunestold.co)")
    parser.add_argument("--tileset-name", default="google_3d",
                        help="Name for the tileset (default: google_3d)")
    
    # Action options
    parser.add_argument("--fetch-metadata", action="store_true", 
                        help="Fetch and display 3D Tiles metadata")
    parser.add_argument("--fetch-tileset", action="store_true",
                        help="Fetch and display tileset.json")
    parser.add_argument("--download", action="store_true",
                        help="Download 3D tiles to local storage")
    parser.add_argument("--upload", action="store_true",
                        help="Upload downloaded tiles to GCS")
    parser.add_argument("--stream", action="store_true",
                        help="Stream tiles directly to GCS (combines download and upload)")
    parser.add_argument("--convert", action="store_true",
                        help="Convert to Arcanum style after downloading")
    
    # Other options
    parser.add_argument("--keep-originals", action="store_true",
                        help="Keep original tiles after transforming")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip existing files when downloading")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    
    return parser.parse_args()

def setup_environment(args):
    """Setup environment with configurations.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with configuration variables
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Create configuration dictionary
    config = {
        "api_key": args.api_key or os.environ.get("GOOGLE_MAPS_API_KEY"),
        "region": args.region,
        "output_dir": os.path.abspath(os.path.expanduser(args.output_dir)),
        "gcs_bucket": args.gcs_bucket,
        "cdn_url": args.cdn_url,
        "tileset_name": args.tileset_name,
        "depth": args.depth,
        "max_tiles": args.max_tiles,
        "keep_originals": args.keep_originals,
    }
    
    # Ensure the API key is available
    if not config["api_key"]:
        logger.error("No Google Maps API key provided. Set GOOGLE_MAPS_API_KEY environment variable or use --api-key.")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(config["output_dir"], exist_ok=True)
    
    return config

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup environment and configuration
    config = setup_environment(args)
    
    # Initialize 3D Tiles integration
    tiles_integration = Google3DTilesIntegration(
        api_key=config["api_key"],
        cache_dir=os.path.join(config["output_dir"], "cache")
    )
    
    # Fetch metadata if requested
    if args.fetch_metadata:
        try:
            metadata = tiles_integration.fetch_metadata()
            print(json.dumps(metadata, indent=2))
            print(f"\nMetadata successfully fetched and displayed.")
        except Exception as e:
            logger.error(f"Error fetching metadata: {str(e)}")
            return 1
    
    # Fetch tileset if requested
    if args.fetch_tileset:
        try:
            tileset_json = tiles_integration.fetch_tileset_json(args.region)
            print(json.dumps(tileset_json, indent=2))
            print(f"\nTileset.json for region '{args.region}' successfully fetched and displayed.")
        except Exception as e:
            logger.error(f"Error fetching tileset.json: {str(e)}")
            return 1
    
    # Download tiles if requested
    if args.download:
        try:
            tileset_json = tiles_integration.fetch_tileset_json(args.region)
            
            download_dir = os.path.join(config["output_dir"], 
                                       f"google_3d_tiles_{args.region}")
            os.makedirs(download_dir, exist_ok=True)
            
            logger.info(f"Downloading tiles to {download_dir} with max depth {args.depth}")
            
            # Fetch tiles recursively
            downloaded_paths = tiles_integration.fetch_tiles_recursive(
                tileset_json, 
                max_depth=args.depth, 
                output_dir=download_dir
            )
            
            logger.info(f"Downloaded {len(downloaded_paths)} tiles to {download_dir}")
            print(f"\nSuccessfully downloaded {len(downloaded_paths)} tiles to {download_dir}")
            
        except Exception as e:
            logger.error(f"Error downloading tiles: {str(e)}")
            return 1
    
    # Stream tiles if requested
    if args.stream:
        try:
            # Initialize storage manager
            storage_manager = ArcanumStorageManager(
                local_root_dir=config["output_dir"],
                gcs_bucket_name=config["gcs_bucket"],
                cdn_url=config["cdn_url"],
                cleanup_originals=not config["keep_originals"]
            )
            
            logger.info(f"Streaming 3D tiles from region '{args.region}' to GCS bucket '{config['gcs_bucket']}'")
            
            results = tiles_integration.stream_tiles_to_storage(
                region=args.region,
                max_depth=args.depth,
                storage_manager=storage_manager
            )
            
            logger.info(f"Streaming complete. Processed {results.get('downloaded_tiles', 0)} tiles.")
            
            if "upload_success" in results:
                logger.info(f"Successfully uploaded {results['upload_success']} files to GCS")
                
                # Show the tileset URL
                tileset_gcs_path = f"tilesets/google_3d_tiles_{args.region or 'global'}/tileset.json"
                tileset_url = f"{config['cdn_url']}/{tileset_gcs_path}"
                
                print(f"\nTileset URL: {tileset_url}")
                print(f"Successfully streamed and uploaded {results['upload_success']} files to GCS")
            
        except Exception as e:
            logger.error(f"Error streaming tiles: {str(e)}")
            return 1
    
    # Upload to GCS if requested
    if args.upload and not args.stream and args.download:
        try:
            # Initialize storage manager
            storage_manager = ArcanumStorageManager(
                local_root_dir=config["output_dir"],
                gcs_bucket_name=config["gcs_bucket"],
                cdn_url=config["cdn_url"],
                cleanup_originals=not config["keep_originals"]
            )
            
            # Source directory for previously downloaded tiles
            source_dir = os.path.join(config["output_dir"], 
                                    f"google_3d_tiles_{args.region}")
            
            if not os.path.exists(source_dir):
                logger.error(f"Source directory not found: {source_dir}. Run with --download first.")
                return 1
            
            # Prepare tileset.json path
            tileset_path = os.path.join(source_dir, "tileset.json")
            
            # Check if tileset.json exists
            if not os.path.exists(tileset_path):
                logger.error(f"Tileset.json not found: {tileset_path}. Run with --download first.")
                return 1
            
            # Upload tileset.json
            tileset_gcs_path = f"tilesets/google_3d_tiles_{args.region}/tileset.json"
            cdn_url = storage_manager.upload_to_gcs(
                tileset_path,
                tileset_gcs_path,
                content_type="application/json",
                cache_control="public, max-age=3600"
            )
            
            uploaded_files = []
            if cdn_url:
                uploaded_files.append({
                    "local_path": tileset_path,
                    "gcs_path": tileset_gcs_path,
                    "cdn_url": cdn_url
                })
            
            # Process all files in the directory recursively
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file == "tileset.json" and root == source_dir:
                        # Already uploaded the root tileset.json
                        continue
                    
                    local_path = os.path.join(root, file)
                    
                    # Get relative path from source directory
                    rel_path = os.path.relpath(local_path, source_dir)
                    
                    # Create GCS path
                    gcs_path = f"tilesets/google_3d_tiles_{args.region}/{rel_path}"
                    
                    # Determine content type
                    ext = os.path.splitext(file)[1]
                    content_type = storage_manager._get_content_type(ext)
                    
                    # Upload to GCS
                    logger.info(f"Uploading {local_path} to {gcs_path}")
                    
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
            
            logger.info(f"Uploaded {len(uploaded_files)} files to GCS")
            
            # Show the tileset URL
            tileset_url = f"{config['cdn_url']}/{tileset_gcs_path}"
            print(f"\nTileset URL: {tileset_url}")
            print(f"Successfully uploaded {len(uploaded_files)} files to GCS")
            
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            return 1
    
    # Convert to Arcanum style if requested
    if args.convert:
        try:
            # Initialize storage integration
            from integration_tools.comfyui_integration import ComfyUIStyleTransformer
            
            logger.info("Initializing ComfyUI style transformer...")
            
            comfyui_path = os.environ.get("COMFYUI_PATH", "./ComfyUI")
            
            # Initialize the style transformer
            style_transformer = ComfyUIStyleTransformer(
                comfyui_path=comfyui_path
            )
            
            # Initialize storage integration
            storage_integration = ArcanumStorageIntegration(
                comfyui_path=comfyui_path,
                local_root_dir=config["output_dir"],
                gcs_bucket_name=config["gcs_bucket"],
                cdn_url=config["cdn_url"],
                cleanup_originals=not config["keep_originals"]
            )
            
            # Source directory for previously downloaded tiles
            source_dir = os.path.join(config["output_dir"], 
                                    f"google_3d_tiles_{args.region}")
            
            if not os.path.exists(source_dir):
                logger.error(f"Source directory not found: {source_dir}. Run with --download first.")
                return 1
            
            # Process image files in the directory
            results = storage_integration.transform_and_upload_directory(
                input_dir=source_dir,
                tileset_name=f"arcanum_{args.region}_3d",
                file_pattern="*.{jpg,jpeg,png}",
                z_level=0,
                prompt="arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate building details, foggy streets, gas lamps, mystical",
                use_image_captioning=True,
                strength=0.75,
                delete_originals=not config["keep_originals"]
            )
            
            # Print summary
            successes = sum(1 for r in results if r.get('success', False))
            logger.info(f"Processed {len(results)} images with {successes} successes")
            
            if successes > 0:
                # Get the tileset.json URL
                tileset_url = f"{config['cdn_url']}/tilesets/arcanum_{args.region}_3d/tileset.json"
                print(f"\nTransformed Tileset URL: {tileset_url}")
                print(f"Successfully transformed {successes} images to Arcanum style")
            
        except Exception as e:
            logger.error(f"Error converting tiles to Arcanum style: {str(e)}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())