#!/usr/bin/env python3
"""
Arcanum Storage Integration Module
----------------------------------
This module integrates the storage manager with the ComfyUI style transformation,
ensuring efficient storage usage and GCS bucket uploads.
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Union, Callable
import shutil
from pathlib import Path

# Import our storage manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from storage_manager import ArcanumStorageManager

# Import ComfyUI integration
from integration_tools.comfyui_integration import ComfyUIStyleTransformer

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ArcanumStorageIntegration:
    """Class that integrates ComfyUI style transformation with efficient storage management."""
    
    def __init__(self, 
                 comfyui_path: str = None,
                 local_root_dir: str = "./arcanum_output",
                 gcs_bucket_name: str = "arcanum-maps",
                 cdn_url: str = "https://arcanum.fortunestold.co",
                 cleanup_originals: bool = True,
                 max_local_cache_size_gb: float = 10.0,
                 device: str = None):
        """Initialize the storage integration.
        
        Args:
            comfyui_path: Path to ComfyUI installation
            local_root_dir: Local directory for Arcanum output files
            gcs_bucket_name: Google Cloud Storage bucket name
            cdn_url: Base URL for the CDN that serves the GCS content
            cleanup_originals: Whether to delete original files after transformation and upload
            max_local_cache_size_gb: Maximum size of local cache in GB
            device: Device to use for inference (cuda, cpu, etc.)
        """
        # Initialize the storage manager
        self.storage_manager = ArcanumStorageManager(
            local_root_dir=local_root_dir,
            gcs_bucket_name=gcs_bucket_name,
            cdn_url=cdn_url,
            cleanup_originals=cleanup_originals,
            max_local_cache_size_gb=max_local_cache_size_gb
        )
        
        # Initialize the style transformer
        self.style_transformer = ComfyUIStyleTransformer(
            comfyui_path=comfyui_path,
            device=device
        )
        
        logger.info("Arcanum Storage Integration initialized")
    
    def transform_and_upload_image(self,
                                 image_path: str,
                                 tileset_name: str = "arcanum",
                                 x: Optional[int] = None,
                                 y: Optional[int] = None,
                                 z: int = 0,
                                 prompt: str = None,
                                 material_type: str = None,
                                 use_image_captioning: bool = True,
                                 negative_prompt: str = "photorealistic, modern, contemporary, bright colors, clear sky",
                                 strength: float = 0.75,
                                 delete_original: bool = None) -> Dict:
        """Transform an image using ComfyUI and upload it to GCS storage with OGC 3D Tiles structure.
        
        Args:
            image_path: Path to the input image
            tileset_name: Name of the tileset to use
            x: X coordinate for the tile (extracted from filename if None)
            y: Y coordinate for the tile (extracted from filename if None)
            z: Z level for the tile
            prompt: The prompt to guide the image transformation
            material_type: The material type (brick, stone, wood, metal, glass, or general)
            use_image_captioning: Whether to use BLIP-2 for automatic image captioning
            negative_prompt: Negative prompt to guide what to avoid in the image
            strength: Strength of the transformation (0.0 to 1.0)
            delete_original: Whether to delete the original file (overrides global setting)
            
        Returns:
            Dictionary with information about the processed tile
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        # Extract coordinates from filename if not provided
        if x is None or y is None:
            filename = os.path.basename(image_path)
            extracted_x, extracted_y = self.storage_manager._extract_coordinates_from_filename(filename, 0)
            x = x if x is not None else extracted_x
            y = y if y is not None else extracted_y
        
        # Determine the tile type based on file extension
        _, file_extension = os.path.splitext(image_path)
        tile_type = file_extension[1:].lower()  # Remove the dot
        
        # Create a transformation function that uses our style transformer
        def transform_function(src_path, dst_path):
            return self.style_transformer.transform_image(
                image_path=src_path,
                output_path=dst_path,
                prompt=prompt,
                material_type=material_type,
                use_image_captioning=use_image_captioning,
                negative_prompt=negative_prompt,
                strength=strength
            )
        
        # Process the tile using the storage manager
        result = self.storage_manager.process_tile(
            source_path=image_path,
            tile_type=tile_type,
            x=x,
            y=y,
            z=z,
            tileset_name=tileset_name,
            transform_function=transform_function,
            delete_original=delete_original
        )
        
        return result
    
    def transform_and_upload_directory(self,
                                     input_dir: str,
                                     tileset_name: str = "arcanum",
                                     file_pattern: str = "*.jpg",
                                     z_level: int = 0,
                                     prompt: str = None,
                                     use_image_captioning: bool = True,
                                     negative_prompt: str = "photorealistic, modern, contemporary, bright colors, clear sky",
                                     strength: float = 0.75,
                                     delete_originals: bool = None) -> List[Dict]:
        """Transform all images in a directory and upload them to GCS with OGC 3D Tiles structure.
        
        Args:
            input_dir: Directory containing input images
            tileset_name: Name of the tileset
            file_pattern: Pattern to match files
            z_level: Z-level for the tiles
            prompt: The prompt to guide the image transformation
            use_image_captioning: Whether to use BLIP-2 for automatic image captioning
            negative_prompt: Negative prompt to guide what to avoid in the image
            strength: Strength of the transformation (0.0 to 1.0)
            delete_originals: Whether to delete original files
            
        Returns:
            List of dictionaries with information about processed tiles
        """
        if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Create a transformation function that uses our style transformer
        def transform_function(src_path, dst_path):
            # Determine material type based on filename
            material_type = self.style_transformer.get_material_type(src_path)
            
            return self.style_transformer.transform_image(
                image_path=src_path,
                output_path=dst_path,
                prompt=prompt,
                material_type=material_type,
                use_image_captioning=use_image_captioning,
                negative_prompt=negative_prompt,
                strength=strength
            )
        
        # Process the image batch using the storage manager
        results = self.storage_manager.process_image_batch(
            source_dir=input_dir,
            target_tileset=tileset_name,
            transform_function=transform_function,
            file_pattern=file_pattern,
            z_level=z_level,
            delete_originals=delete_originals
        )
        
        return results
    
    def transform_satellite_images(self, satellite_dir: str, output_tileset: str = "arcanum") -> List[Dict]:
        """Transform satellite imagery to match Arcanum style and upload to GCS.
        
        Args:
            satellite_dir: Directory containing satellite images
            output_tileset: Name of the output tileset
            
        Returns:
            List of dictionaries with information about processed tiles
        """
        # Create a specialized prompt for satellite imagery
        satellite_prompt = """arcanum aerial view, gothic victorian fantasy steampunk city,
alternative London, dark atmosphere, fog and mist, intricate cityscape, aerial perspective"""
        
        # Use a lower strength for satellite imagery to preserve geographic features
        return self.transform_and_upload_directory(
            input_dir=satellite_dir,
            tileset_name=output_tileset,
            file_pattern="*.{jpg,jpeg,png,tif,tiff}",
            z_level=0,
            prompt=satellite_prompt,
            use_image_captioning=True,
            strength=0.65,
            delete_originals=True
        )
    
    def transform_street_view_images(self, street_view_dir: str, output_tileset: str = "arcanum_street_view") -> List[Dict]:
        """Transform street view imagery to Arcanum style and upload to GCS.
        
        Args:
            street_view_dir: Directory containing street view images
            output_tileset: Name of the output tileset
            
        Returns:
            List of dictionaries with information about processed tiles
        """
        # Create a specialized prompt for street view imagery
        street_view_prompt = """arcanum street view, gothic victorian fantasy steampunk,
alternative London, dark atmosphere, ornate building details, foggy streets, gas lamps, mystical"""
        
        return self.transform_and_upload_directory(
            input_dir=street_view_dir,
            tileset_name=output_tileset,
            file_pattern="*.{jpg,jpeg,png}",
            z_level=0,
            prompt=street_view_prompt,
            use_image_captioning=True,
            strength=0.8,
            delete_originals=True
        )
    
    def transform_facade_textures(self, 
                                building_type: str,
                                era: str,
                                reference_image_path: str = None,
                                output_tileset: str = "arcanum_facades") -> Dict:
        """Generate a facade texture based on building type and era, with Arcanum styling.
        
        Args:
            building_type: Type of building (residential, commercial, etc.)
            era: Architectural era (victorian, georgian, etc.)
            reference_image_path: Optional path to a reference image
            output_tileset: Name of the output tileset
            
        Returns:
            Dictionary with information about the processed texture
        """
        # If no reference image is provided, use a default one
        if not reference_image_path or not os.path.exists(reference_image_path):
            # Use a default facade image
            default_facades_dir = os.path.join(self.storage_manager.local_root, "raw_data", "default_facades")
            os.makedirs(default_facades_dir, exist_ok=True)
            
            # Check if we have a default image for this building type and era
            default_path = os.path.join(default_facades_dir, f"default_{building_type}_{era}.jpg")
            if not os.path.exists(default_path):
                logger.warning(f"No reference image provided and no default facade found for {building_type} {era}")
                # Return a placeholder result
                return {
                    "error": "No reference image provided and no default facade found",
                    "building_type": building_type,
                    "era": era,
                    "success": False
                }
            
            reference_image_path = default_path
        
        # Use a specialized prompt for facades
        # Custom prompt based on building type and era
        facade_prompt = f"arcanum {era} {building_type} building facade, gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure"
        
        # Set x and y based on building type and era to create a grid of facades
        building_types = ["residential", "commercial", "historical", "modern"]
        eras = ["victorian", "georgian", "modern", "postwar"]
        
        # Get x from building type index
        x = building_types.index(building_type) if building_type in building_types else 0
        
        # Get y from era index
        y = eras.index(era) if era in eras else 0
        
        # Create a specific output path for the facade
        facade_filename = f"facade_{building_type}_{era}.jpg"
        temp_dir = os.path.join(self.storage_manager.local_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, facade_filename)
        
        # Transform the facade using our style transformer
        result = self.style_transformer.transform_image(
            image_path=reference_image_path,
            output_path=temp_path,
            prompt=facade_prompt,
            material_type="general",
            use_image_captioning=False,
            strength=0.8
        )
        
        if result:
            # Upload the transformed facade to GCS
            return self.transform_and_upload_image(
                image_path=temp_path,
                tileset_name=output_tileset,
                x=x,
                y=y,
                z=0,
                delete_original=True
            )
        else:
            return {
                "error": "Failed to transform facade image",
                "building_type": building_type,
                "era": era,
                "success": False
            }
    
def main():
    """Main function for testing the module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Arcanum Storage Integration")
    parser.add_argument("--input", help="Input image or directory", required=True)
    parser.add_argument("--comfyui-path", help="Path to ComfyUI installation")
    parser.add_argument("--output-dir", default="./arcanum_output", help="Output directory")
    parser.add_argument("--gcs-bucket", default="arcanum-maps", help="GCS bucket name")
    parser.add_argument("--cdn-url", default="https://arcanum.fortunestold.co", help="CDN URL")
    parser.add_argument("--tileset", default="arcanum", help="Tileset name")
    parser.add_argument("--prompt", help="Custom prompt for transformation")
    parser.add_argument("--strength", type=float, default=0.75, help="Transformation strength")
    parser.add_argument("--keep-originals", action="store_true", help="Keep original files")
    args = parser.parse_args()
    
    # Initialize the integration
    integration = ArcanumStorageIntegration(
        comfyui_path=args.comfyui_path,
        local_root_dir=args.output_dir,
        gcs_bucket_name=args.gcs_bucket,
        cdn_url=args.cdn_url,
        cleanup_originals=not args.keep_originals
    )
    
    # Process input
    if os.path.isdir(args.input):
        logger.info(f"Processing directory: {args.input}")
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
            logger.info(f"Successfully processed image: {result['cdn_url']}")
        else:
            logger.error(f"Failed to process image: {result.get('error', 'Unknown error')}")
            
    else:
        logger.error(f"Input not found: {args.input}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())