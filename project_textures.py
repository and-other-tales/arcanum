#!/usr/bin/env python3
"""
Project Textures
---------------
Standalone script to project Street View imagery onto 3D building meshes.
This provides a command-line interface to integration_tools.texture_projection.
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from integration_tools.texture_projection import TextureProjector, project_street_view_to_building
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Texture Projection module: {str(e)}")
    MODULES_AVAILABLE = False

def process_building(args) -> int:
    """Process a single building."""
    if not MODULES_AVAILABLE:
        logger.error("Texture Projection module not available")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load metadata if provided
    metadata = None
    if args.metadata:
        try:
            with open(args.metadata, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return 1
    
    # Process building
    logger.info(f"Processing building {args.id} with mesh {args.mesh}")
    logger.info(f"Using Street View images from {args.images}")
    
    start_time = time.time()
    result = project_street_view_to_building(
        args.id,
        args.mesh,
        args.images,
        args.output,
        metadata
    )
    elapsed_time = time.time() - start_time
    
    if result.get("success", False):
        logger.info(f"Successfully processed building {args.id} in {elapsed_time:.2f} seconds")
        logger.info(f"Material: {result.get('material_path')}")
        logger.info(f"Texture: {result.get('texture_path')}")
        
        # Save result summary
        summary_path = os.path.join(args.output, f"{args.id}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Result summary saved to {summary_path}")
        return 0
    else:
        logger.error(f"Failed to process building: {result.get('error', 'Unknown error')}")
        return 1

def process_batch(args) -> int:
    """Process a batch of buildings."""
    if not MODULES_AVAILABLE:
        logger.error("Texture Projection module not available")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load buildings data
    try:
        with open(args.buildings, 'r') as f:
            buildings_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading buildings data: {str(e)}")
        return 1
    
    # Create texture projector
    projector = TextureProjector(
        output_dir=args.output,
        parallel_processing=args.parallel,
        max_workers=args.workers
    )
    
    # Process buildings
    logger.info(f"Processing {len(buildings_data)} buildings from {args.buildings}")
    logger.info(f"Using Street View images from {args.images}")
    logger.info(f"Parallel processing: {args.parallel}, Workers: {args.workers or 'auto'}")
    
    start_time = time.time()
    result = projector.process_building_batch(
        buildings_data,
        args.images,
        args.output
    )
    elapsed_time = time.time() - start_time
    
    if result.get("success", False):
        logger.info(f"Successfully processed {result.get('success_count')} of {result.get('total_buildings')} buildings in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to {result.get('results_path')}")
        return 0
    else:
        logger.error(f"Batch processing failed: {result.get('error', 'Unknown error')}")
        return 1

def process_area(args) -> int:
    """Process all buildings in a geographic area."""
    if not MODULES_AVAILABLE:
        logger.error("Texture Projection module not available")
        return 1
    
    # Parse bounds
    try:
        bounds = {
            "north": float(args.bounds.split(',')[0]),
            "south": float(args.bounds.split(',')[1]),
            "east": float(args.bounds.split(',')[2]),
            "west": float(args.bounds.split(',')[3])
        }
    except Exception as e:
        logger.error(f"Invalid bounds format. Use: north,south,east,west")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"Processing area with bounds: {bounds}")
    logger.info(f"Using mesh directory: {args.mesh_dir}")
    logger.info(f"Using Street View directory: {args.images}")
    
    # Collect all building meshes in the area
    buildings = {}
    
    # If a buildings metadata file is provided, use it
    if args.area_metadata and os.path.exists(args.area_metadata):
        try:
            with open(args.area_metadata, 'r') as f:
                area_data = json.load(f)
                
                # Filter buildings by bounds
                for building_id, metadata in area_data.items():
                    if "center" in metadata:
                        lat, lon = metadata["center"]
                        if (bounds["south"] <= lat <= bounds["north"] and
                            bounds["west"] <= lon <= bounds["east"]):
                            buildings[building_id] = metadata
                
                logger.info(f"Found {len(buildings)} buildings in the specified area from metadata")
                
        except Exception as e:
            logger.error(f"Error loading area metadata: {str(e)}")
            return 1
    
    # If no buildings found from metadata, scan mesh directory
    if not buildings and args.mesh_dir and os.path.exists(args.mesh_dir):
        for file in os.listdir(args.mesh_dir):
            if file.lower().endswith(".obj"):
                building_id = file.split('.')[0]
                buildings[building_id] = {
                    "id": building_id,
                    "mesh_path": os.path.join(args.mesh_dir, file)
                }
        
        logger.info(f"Found {len(buildings)} building meshes in the directory")
    
    if not buildings:
        logger.error("No buildings found for the specified area")
        return 1
    
    # Process all buildings
    projector = TextureProjector(
        output_dir=args.output,
        parallel_processing=args.parallel,
        max_workers=args.workers
    )
    
    # Add mesh paths if not already present
    for building_id, data in buildings.items():
        if "mesh_path" not in data:
            mesh_file = f"{building_id}.obj"
            mesh_path = os.path.join(args.mesh_dir, mesh_file)
            if os.path.exists(mesh_path):
                data["mesh_path"] = mesh_path
    
    # Filter out buildings without mesh paths
    buildings = {bid: data for bid, data in buildings.items() if "mesh_path" in data}
    
    if not buildings:
        logger.error("No valid building meshes found")
        return 1
    
    # Process buildings
    start_time = time.time()
    result = projector.process_building_batch(
        buildings,
        args.images,
        args.output
    )
    elapsed_time = time.time() - start_time
    
    if result.get("success", False):
        logger.info(f"Successfully processed {result.get('success_count')} of {result.get('total_buildings')} buildings in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to {result.get('results_path')}")
        return 0
    else:
        logger.error(f"Area processing failed: {result.get('error', 'Unknown error')}")
        return 1

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Project Street View images onto 3D building meshes")
    
    # Define subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single building processing
    building_parser = subparsers.add_parser("building", help="Process a single building")
    building_parser.add_argument("--id", required=True, help="Building ID")
    building_parser.add_argument("--mesh", required=True, help="Path to building mesh")
    building_parser.add_argument("--images", required=True, help="Path to Street View images directory or file")
    building_parser.add_argument("--output", default="./texture_output", help="Output directory")
    building_parser.add_argument("--metadata", help="Path to building metadata JSON file")
    
    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Process a batch of buildings")
    batch_parser.add_argument("--buildings", required=True, help="Path to buildings metadata JSON")
    batch_parser.add_argument("--images", required=True, help="Path to Street View images directory")
    batch_parser.add_argument("--output", default="./texture_output", help="Output directory")
    batch_parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    batch_parser.add_argument("--workers", type=int, help="Number of worker processes (default: auto)")
    
    # Area processing
    area_parser = subparsers.add_parser("area", help="Process all buildings in a geographic area")
    area_parser.add_argument("--bounds", required=True, help="Area bounds as north,south,east,west")
    area_parser.add_argument("--mesh-dir", required=True, help="Directory containing building meshes")
    area_parser.add_argument("--images", required=True, help="Path to Street View images directory")
    area_parser.add_argument("--output", default="./texture_output", help="Output directory")
    area_parser.add_argument("--area-metadata", help="Path to area metadata JSON file")
    area_parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    area_parser.add_argument("--workers", type=int, help="Number of worker processes (default: auto)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check for required modules
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Please check your installation.")
        return 1
    
    # Process based on command
    if args.command == "building":
        return process_building(args)
    elif args.command == "batch":
        return process_batch(args)
    elif args.command == "area":
        return process_area(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        sys.exit(1)