#!/usr/bin/env python3
"""
Arcanum City Generator
--------------------
This is the main entry point for the Arcanum city generator.
It provides a unified interface to the various modules and functionality.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arcanum.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from modules import osm, comfyui
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Arcanum modules: {str(e)}")
    MODULES_AVAILABLE = False

# Project configuration
PROJECT_CONFIG = {
    "project_name": "Arcanum3D",
    "output_directory": "./arcanum_3d_output",
    "coordinate_system": "EPSG:4326",  # WGS84 (latitude/longitude)
    "center_point": (51.5074, -0.1278),  # Approximate center of Arcanum (London)
    "bounds": {
        "north": 51.5084,  # Northern extent (latitude) - ~100m north of center
        "south": 51.5064,  # Southern extent (latitude) - ~100m south of center
        "east": -0.1258,   # Eastern extent (longitude) - ~100m east of center
        "west": -0.1298    # Western extent (longitude) - ~100m west of center
    },
    "cell_size": 100,    # Grid cell size in meters
    "lod_levels": {
        "LOD0": 1000,     # Simple blocks for far viewing (1km+)
        "LOD1": 500,      # Basic buildings with accurate heights (500m-1km)
        "LOD2": 250,      # Buildings with roof structures (250-500m)
        "LOD3": 100,      # Detailed exteriors with architectural features (100-250m)
        "LOD4": 0         # Highly detailed models with facade elements (0-100m)
    }
}

def setup_directory_structure(base_dir: str = None) -> str:
    """Create the necessary directory structure for the project."""
    if base_dir is None:
        base_dir = PROJECT_CONFIG["output_directory"]
    
    # Create main directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create subdirectories for different data types
    subdirs = [
        "raw_data/satellite",
        "raw_data/lidar",
        "raw_data/vector",
        "raw_data/street_view",
        "processed_data/terrain",
        "processed_data/buildings",
        "processed_data/textures",
        "processed_data/vegetation",
        "3d_models/buildings",
        "3d_models/landmarks",
        "3d_models/street_furniture",
        "unity_assets/prefabs",
        "unity_assets/materials",
        "unity_assets/textures",
        "logs"
    ]
    
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
    
    logger.info("Directory structure set up complete")
    return base_dir

def download_osm_data(bounds: Dict[str, float], output_dir: str, grid_size: int = None, coordinate_system: str = "EPSG:4326") -> Dict[str, Any]:
    """
    Download OpenStreetMap data using the appropriate method.
    
    Args:
        bounds: Dictionary with north, south, east, west boundaries
        output_dir: Directory to save output files
        grid_size: Size of grid cells in meters (uses grid-based approach if provided)
        coordinate_system: Coordinate system of the bounds
        
    Returns:
        Dictionary with status information
    """
    # Ensure OSM module is available
    if not MODULES_AVAILABLE:
        return {
            "success": False,
            "error": "Arcanum modules not available"
        }
    
    # Determine which approach to use based on grid_size
    if grid_size is not None and grid_size > 0:
        # Use grid-based approach for larger areas
        logger.info(f"Using grid-based OSM downloader with {grid_size}m cells")
        result = osm.download_osm_grid(
            bounds=bounds,
            output_dir=output_dir,
            cell_size_meters=grid_size
        )
        
        # Merge grid data
        if result.get("success_count", 0) > 0:
            merge_result = osm.merge_grid_data(output_dir)
            if merge_result.get("success", False):
                logger.info("Successfully merged grid data")
                return {
                    "success": True,
                    "message": f"Downloaded and merged OSM data for {result.get('success_count', 0)} grid cells",
                    "buildings_path": merge_result.get("buildings_path"),
                    "roads_path": merge_result.get("roads_path")
                }
            else:
                logger.warning(f"Grid data merge failed: {merge_result.get('error', 'Unknown error')}")
        
        return {
            "success": result.get("success_count", 0) > 0,
            "message": f"Downloaded OSM data for {result.get('success_count', 0)} grid cells"
        }
    else:
        # Use direct approach for smaller areas
        logger.info("Using direct OSM downloader")
        return osm.download_osm_data(
            bounds=bounds,
            output_dir=output_dir,
            coordinate_system=coordinate_system
        )

def transform_images(
    input_dir: str,
    output_dir: str,
    prompt: str = "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical",
    pattern: str = "*.jpg",
    use_controlnet: bool = True
) -> Dict[str, Any]:
    """
    Transform images in input_dir to Arcanum style and save to output_dir.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save transformed images
        prompt: Text prompt for image generation
        pattern: Glob pattern to match input images
        use_controlnet: Whether to use ControlNet for better structure preservation
        
    Returns:
        Dictionary with transformation results
    """
    # Ensure ComfyUI module is available
    if not MODULES_AVAILABLE:
        return {
            "success": False,
            "error": "Arcanum modules not available"
        }
    
    # Find all matching images
    import glob
    from pathlib import Path
    
    image_paths = []
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not image_paths:
        return {
            "success": False,
            "error": f"No images found in {input_dir}"
        }
    
    # Create transformer
    transformer = comfyui.ArcanumStyleTransformer(
        comfyui_path=os.environ.get("COMFYUI_PATH", "./ComfyUI"),
        hf_token=os.environ.get("HUGGINGFACE_TOKEN")
    )
    
    # Transform images
    logger.info(f"Transforming {len(image_paths)} images from {input_dir} to {output_dir}")
    result_paths = transformer.batch_transform_images(
        image_paths=image_paths,
        output_dir=output_dir,
        prompt=prompt,
        use_controlnet=use_controlnet
    )
    
    return {
        "success": len(result_paths) > 0,
        "count": len(result_paths),
        "total": len(image_paths),
        "output_dir": output_dir
    }

def run_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the complete Arcanum workflow.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with workflow results
    """
    try:
        # Set up directories
        base_dir = setup_directory_structure(config.get("output_directory"))
        
        # Step 1: Download OSM data
        logger.info("Step 1: Downloading OpenStreetMap data")
        osm_result = download_osm_data(
            bounds=config["bounds"],
            output_dir=base_dir,
            grid_size=config.get("cell_size"),
            coordinate_system=config.get("coordinate_system", "EPSG:4326")
        )
        
        if not osm_result.get("success", False):
            logger.warning(f"OSM data download failed: {osm_result.get('error', 'Unknown error')}")
        
        # Step 2: Download and transform satellite imagery
        # This is a placeholder for the actual implementation
        logger.info("Step 2: Satellite imagery processing not implemented in this version")
        
        # Step 3: Process buildings and roads
        # This is a placeholder for the actual implementation
        logger.info("Step 3: Building and road processing not implemented in this version")
        
        # Step 4: Apply Arcanum styling
        # This is a placeholder for the actual implementation
        logger.info("Step 4: Arcanum styling not fully implemented in this version")
        
        # Return workflow results
        return {
            "success": True,
            "output_dir": base_dir,
            "completed_at": datetime.now().isoformat(),
            "steps_completed": ["data_collection"],
            "steps_pending": ["satellite_processing", "building_processing", "styling"]
        }
    
    except Exception as e:
        logger.error(f"Workflow error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Arcanum 3D City Generator")
    
    # Main operation modes
    subparsers = parser.add_subparsers(dest="command", help="Operation mode")
    
    # Full workflow mode
    workflow_parser = subparsers.add_parser("generate", help="Run full Arcanum generation workflow")
    workflow_parser.add_argument("--output", help="Output directory", default="./arcanum_3d_output")
    workflow_parser.add_argument("--bounds", help="Area bounds (north,south,east,west)", default="51.5084,51.5064,-0.1258,-0.1298")
    workflow_parser.add_argument("--cell-size", type=int, help="Grid cell size in meters", default=100)
    
    # OSM download mode
    osm_parser = subparsers.add_parser("osm", help="Download OpenStreetMap data")
    osm_parser.add_argument("--output", help="Output directory", default="./arcanum_3d_output")
    osm_parser.add_argument("--bounds", help="Area bounds (north,south,east,west)", default="51.5084,51.5064,-0.1258,-0.1298")
    osm_parser.add_argument("--cell-size", type=int, help="Grid cell size in meters (0 for direct download)", default=100)
    
    # Style transform mode
    transform_parser = subparsers.add_parser("transform", help="Transform images to Arcanum style")
    transform_parser.add_argument("--input", help="Input directory containing images", required=True)
    transform_parser.add_argument("--output", help="Output directory for transformed images", default="./arcanum_output/final")
    transform_parser.add_argument("--prompt", help="Text prompt for image generation", 
                               default="arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details")
    transform_parser.add_argument("--no-controlnet", help="Disable ControlNet for faster processing", action="store_true")
    
    # Parse arguments
    args = parser.parse_args()
    
    # No command specified - show help
    if args.command is None:
        parser.print_help()
        return 0
    
    # Run the selected command
    if args.command == "generate":
        # Parse bounds
        bounds_values = args.bounds.split(",")
        if len(bounds_values) != 4:
            logger.error("Invalid bounds format. Use: north,south,east,west")
            return 1
        
        bounds = {
            "north": float(bounds_values[0]),
            "south": float(bounds_values[1]),
            "east": float(bounds_values[2]),
            "west": float(bounds_values[3])
        }
        
        # Update config
        config = PROJECT_CONFIG.copy()
        config["output_directory"] = args.output
        config["bounds"] = bounds
        config["cell_size"] = args.cell_size
        
        # Run the workflow
        result = run_workflow(config)
        
        if result["success"]:
            logger.info(f"Arcanum generation completed successfully.")
            logger.info(f"Output directory: {result['output_dir']}")
            return 0
        else:
            logger.error(f"Arcanum generation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.command == "osm":
        # Parse bounds
        bounds_values = args.bounds.split(",")
        if len(bounds_values) != 4:
            logger.error("Invalid bounds format. Use: north,south,east,west")
            return 1
        
        bounds = {
            "north": float(bounds_values[0]),
            "south": float(bounds_values[1]),
            "east": float(bounds_values[2]),
            "west": float(bounds_values[3])
        }
        
        # Download OSM data
        result = download_osm_data(
            bounds=bounds,
            output_dir=args.output,
            grid_size=args.cell_size
        )
        
        if result.get("success", False):
            logger.info(f"OSM data download completed successfully.")
            logger.info(result.get("message", ""))
            return 0
        else:
            logger.error(f"OSM data download failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.command == "transform":
        # Transform images
        result = transform_images(
            input_dir=args.input,
            output_dir=args.output,
            prompt=args.prompt,
            use_controlnet=not args.no_controlnet
        )
        
        if result.get("success", False):
            logger.info(f"Transformed {result.get('count', 0)}/{result.get('total', 0)} images.")
            logger.info(f"Output directory: {result.get('output_dir')}")
            return 0
        else:
            logger.error(f"Image transformation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())