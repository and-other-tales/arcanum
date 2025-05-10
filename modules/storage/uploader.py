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
from modules.storage.storage_integration import ArcanumStorageIntegration

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

def transform_and_upload(input_path: str, output_dir: str, tileset_name: str = "arcanum",
                      prompt: Optional[str] = None, strength: float = 0.75,
                      keep_originals: bool = False, mode: str = "general") -> Dict:
    """
    Transform images to Arcanum style and upload them to storage.
    
    Args:
        input_path: Path to input image or directory
        output_dir: Local output directory
        tileset_name: Name of the tileset to use
        prompt: Custom prompt for transformation
        strength: Transformation strength (0-1)
        keep_originals: Whether to keep original files
        mode: Type of imagery to process (satellite, street_view, facade, general)
        
    Returns:
        Dictionary with transformation and upload results
    """
    # Initialize the storage integration
    integration = ArcanumStorageIntegration(
        local_root_dir=output_dir,
        cleanup_originals=not keep_originals
    )
    
    # Process input
    if os.path.isdir(input_path):
        logger.info(f"Processing directory: {input_path}")
        
        # Determine which method to use based on mode
        if mode == "satellite":
            results = integration.transform_satellite_images(
                satellite_dir=input_path,
                output_tileset=tileset_name
            )
        elif mode == "street_view":
            results = integration.transform_street_view_images(
                street_view_dir=input_path,
                output_tileset=tileset_name
            )
        else:  # general mode
            results = integration.transform_and_upload_directory(
                input_dir=input_path,
                tileset_name=tileset_name,
                prompt=prompt,
                strength=strength,
                delete_originals=not keep_originals
            )
        
        return {
            "success": any(r.get('success', False) for r in results),
            "processed_count": len(results),
            "success_count": sum(1 for r in results if r.get('success', False)),
            "results": results
        }
        
    elif os.path.isfile(input_path):
        logger.info(f"Processing file: {input_path}")
        
        result = integration.transform_and_upload_image(
            image_path=input_path,
            tileset_name=tileset_name,
            prompt=prompt,
            strength=strength,
            delete_original=not keep_originals
        )
        
        return {
            "success": result.get('success', False),
            "result": result
        }
        
    else:
        error_msg = f"Input not found: {input_path}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

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
    
    # Process input
    if os.path.isdir(args.input):
        logger.info(f"Processing directory: {args.input}")
        
        # Determine which method to use based on mode
        if args.mode == "satellite":
            results = integration.transform_satellite_images(
                satellite_dir=args.input,
                output_tileset=args.tileset
            )
        elif args.mode == "street_view":
            results = integration.transform_street_view_images(
                street_view_dir=args.input,
                output_tileset=args.tileset
            )
        elif args.mode == "facade":
            if not args.building_type or not args.era:
                logger.error("Building type and era are required for facade mode")
                return 1
                
            # Find reference images if they exist
            reference_images = glob.glob(os.path.join(args.input, "*.jpg"))
            if not reference_images:
                reference_images = glob.glob(os.path.join(args.input, "*.png"))
                
            reference_path = reference_images[0] if reference_images else None
                
            result = integration.transform_facade_textures(
                building_type=args.building_type,
                era=args.era,
                reference_image_path=reference_path,
                output_tileset=args.tileset
            )
            
            if result.get('success', False):
                print(f"Successfully processed facade: {result.get('cdn_url')}")
            else:
                print(f"Failed to process facade: {result.get('error', 'Unknown error')}")
            
            return 0
        else:  # general mode
            results = integration.transform_and_upload_directory(
                input_dir=args.input,
                tileset_name=args.tileset,
                prompt=args.prompt,
                strength=args.strength
            )
        
        # Print summary
        successes = sum(1 for r in results if r.get('success', False))
        logger.info(f"Processed {len(results)} images with {successes} successes")
        
    elif os.path.isfile(args.input):
        logger.info(f"Processing file: {args.input}")
        
        result = integration.transform_and_upload_image(
            image_path=args.input,
            tileset_name=args.tileset,
            prompt=args.prompt,
            strength=args.strength
        )
        
        if result.get('success', False):
            logger.info(f"Successfully processed image: {result.get('cdn_url')}")
        else:
            logger.error(f"Failed to process image: {result.get('error', 'Unknown error')}")
            return 1
            
    else:
        logger.error(f"Input not found: {args.input}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())