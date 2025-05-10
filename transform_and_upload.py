#!/usr/bin/env python3
"""
Arcanum Image Transformation and Upload Tool
-------------------------------------------
This tool transforms images to Arcanum style and uploads them to Google Cloud Storage
with proper OGC 3D Tiles structure, while managing local storage efficiently.
"""

import os
import sys
import logging
import argparse
import glob
from typing import List, Dict, Optional

# Import our storage integration
from integration_tools.storage_integration import ArcanumStorageIntegration

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arcanum_transform_upload.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Arcanum Image Transformation and Upload Tool")
    parser.add_argument("--input", help="Input image or directory", required=True)
    parser.add_argument("--output-dir", default="./arcanum_output", help="Local output directory")
    parser.add_argument("--comfyui-path", help="Path to ComfyUI installation")
    parser.add_argument("--gcs-bucket", default="arcanum-maps", help="GCS bucket name")
    parser.add_argument("--cdn-url", default="https://arcanum.fortunestold.co", help="CDN URL")
    parser.add_argument("--tileset", default="arcanum", help="Tileset name")
    parser.add_argument("--prompt", help="Custom prompt for transformation")
    parser.add_argument("--strength", type=float, default=0.75, help="Transformation strength (0-1)")
    parser.add_argument("--keep-originals", action="store_true", help="Keep original files")
    parser.add_argument("--mode", choices=["satellite", "street_view", "facade", "general"], 
                      default="general", help="Type of imagery to process")
    parser.add_argument("--building-type", choices=["residential", "commercial", "historical", "modern"],
                      help="Building type for facade mode")
    parser.add_argument("--era", choices=["victorian", "georgian", "modern", "postwar"],
                      help="Architectural era for facade mode")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use for inference")
    parser.add_argument("--cache-size", type=float, default=10.0, help="Maximum local cache size in GB")
    parser.add_argument("--file-pattern", default="*.{jpg,jpeg,png}", 
                      help="File pattern when processing directories")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Initialize the storage integration
    integration = ArcanumStorageIntegration(
        comfyui_path=args.comfyui_path,
        local_root_dir=args.output_dir,
        gcs_bucket_name=args.gcs_bucket,
        cdn_url=args.cdn_url,
        cleanup_originals=not args.keep_originals,
        max_local_cache_size_gb=args.cache_size,
        device=args.device
    )
    
    # Process based on mode
    if args.mode == "facade" and args.building_type and args.era:
        # Process facade texture
        logger.info(f"Processing facade texture for {args.building_type} ({args.era})")
        result = integration.transform_facade_textures(
            building_type=args.building_type,
            era=args.era,
            reference_image_path=args.input if os.path.isfile(args.input) else None,
            output_tileset=f"{args.tileset}_facades"
        )
        
        if result.get('success', False):
            logger.info(f"Successfully processed facade: {result['cdn_url']}")
            logger.info(f"Original image: {result['original_path']}")
            if result.get('transformed_path'):
                logger.info(f"Transformed image: {result['transformed_path']}")
            print(f"SUCCESS: Facade uploaded to {result['cdn_url']}")
            return 0
        else:
            logger.error(f"Failed to process facade: {result.get('error', 'Unknown error')}")
            print(f"ERROR: Failed to process facade")
            return 1
    
    elif args.mode == "satellite":
        # Process satellite imagery
        if not os.path.isdir(args.input):
            logger.error("Satellite mode requires a directory input")
            print("ERROR: Satellite mode requires a directory input")
            return 1
        
        logger.info(f"Processing satellite imagery directory: {args.input}")
        results = integration.transform_satellite_images(
            satellite_dir=args.input,
            output_tileset=f"{args.tileset}_satellite"
        )
        
        # Print summary
        successes = sum(1 for r in results if r.get('success', False))
        logger.info(f"Processed {len(results)} satellite images with {successes} successes")
        print(f"SUCCESS: Processed {len(results)} satellite images with {successes} successes")
        
        if successes > 0:
            # Get the tileset.json URL
            tileset_url = f"{args.cdn_url}/tilesets/{args.tileset}_satellite/tileset.json"
            print(f"Tileset URL: {tileset_url}")
        
        return 0 if successes > 0 else 1
    
    elif args.mode == "street_view":
        # Process street view imagery
        if not os.path.isdir(args.input):
            logger.error("Street view mode requires a directory input")
            print("ERROR: Street view mode requires a directory input")
            return 1
        
        logger.info(f"Processing street view imagery directory: {args.input}")
        results = integration.transform_street_view_images(
            street_view_dir=args.input,
            output_tileset=f"{args.tileset}_street_view"
        )
        
        # Print summary
        successes = sum(1 for r in results if r.get('success', False))
        logger.info(f"Processed {len(results)} street view images with {successes} successes")
        print(f"SUCCESS: Processed {len(results)} street view images with {successes} successes")
        
        if successes > 0:
            # Get the tileset.json URL
            tileset_url = f"{args.cdn_url}/tilesets/{args.tileset}_street_view/tileset.json"
            print(f"Tileset URL: {tileset_url}")
        
        return 0 if successes > 0 else 1
    
    else:  # General mode
        # Process general imagery
        if os.path.isdir(args.input):
            logger.info(f"Processing directory: {args.input}")
            results = integration.transform_and_upload_directory(
                input_dir=args.input,
                tileset_name=args.tileset,
                file_pattern=args.file_pattern,
                prompt=args.prompt,
                strength=args.strength
            )
            
            # Print summary
            successes = sum(1 for r in results if r.get('success', False))
            logger.info(f"Processed {len(results)} images with {successes} successes")
            print(f"SUCCESS: Processed {len(results)} images with {successes} successes")
            
            if successes > 0:
                # Get the tileset.json URL
                tileset_url = f"{args.cdn_url}/tilesets/{args.tileset}/tileset.json"
                print(f"Tileset URL: {tileset_url}")
            
            return 0 if successes > 0 else 1
            
        elif os.path.isfile(args.input):
            logger.info(f"Processing file: {args.input}")
            result = integration.transform_and_upload_image(
                image_path=args.input,
                tileset_name=args.tileset,
                prompt=args.prompt,
                strength=args.strength
            )
            
            if result.get('success', False):
                logger.info(f"Successfully processed image: {result['cdn_url']}")
                print(f"SUCCESS: Image uploaded to {result['cdn_url']}")
                return 0
            else:
                logger.error(f"Failed to process image: {result.get('error', 'Unknown error')}")
                print(f"ERROR: Failed to process image")
                return 1
                
        else:
            logger.error(f"Input not found: {args.input}")
            print(f"ERROR: Input not found: {args.input}")
            return 1

if __name__ == "__main__":
    sys.exit(main())