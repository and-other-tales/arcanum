#!/usr/bin/env python3
"""
Fetch Street View Images
-----------------------
Standalone script to download Google Street View images for a specified area.
This provides command-line interface to integration_tools.street_view_integration.
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, List, Any, Optional, Tuple, Union

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
    from integration_tools.street_view_integration import GoogleStreetViewIntegration, fetch_street_view
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Street View integration module: {str(e)}")
    MODULES_AVAILABLE = False

def parse_coordinates(coords_str: str) -> Tuple[float, float]:
    """Parse lat,lng string into a tuple of floats."""
    try:
        lat, lng = map(float, coords_str.split(','))
        return lat, lng
    except ValueError:
        raise ValueError("Coordinates must be in format 'latitude,longitude'")

def fetch_panorama_grid(
    bounds: Dict[str, float],
    output_dir: str,
    grid_size: float = 100.0,
    api_key: str = None,
    fov: int = 90,
    radius: int = 100
) -> Dict[str, Any]:
    """
    Fetch Street View panoramas for a grid of points within bounds.
    
    Args:
        bounds: Dictionary with north, south, east, west bounds
        output_dir: Directory to save images to
        grid_size: Size of grid cells in meters
        api_key: Google Maps API key
        fov: Field of view (20-120)
        radius: Maximum radius in meters to search for imagery
        
    Returns:
        Dictionary with results summary
    """
    if not MODULES_AVAILABLE:
        return {
            "success": False,
            "error": "Street View integration module not available"
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define grid spacing in degrees (approximate)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km * cos(latitude)
    lat_center = (bounds["north"] + bounds["south"]) / 2
    
    # Convert grid_size from meters to degrees
    lat_spacing = grid_size / 111000  # degrees latitude
    lng_spacing = grid_size / (111000 * abs(math.cos(math.radians(lat_center))))  # degrees longitude
    
    # Create grid points
    grid_points = []
    for lat in np.arange(bounds["south"], bounds["north"], lat_spacing):
        for lng in np.arange(bounds["west"], bounds["east"], lng_spacing):
            grid_points.append((lat, lng))
    
    # Initialize Street View integration
    integration = GoogleStreetViewIntegration(api_key=api_key)
    
    # Track results
    results = {
        "total_points": len(grid_points),
        "success_count": 0,
        "failed_count": 0,
        "no_imagery_count": 0,
        "downloaded_images": 0,
        "points_with_imagery": []
    }
    
    # Process each grid point
    for i, (lat, lng) in enumerate(grid_points):
        logger.info(f"Processing grid point {i+1}/{len(grid_points)}: ({lat}, {lng})")
        
        # Create subdirectory for this point
        point_dir = os.path.join(output_dir, f"point_{i:04d}_{lat:.6f}_{lng:.6f}")
        
        # Check if Street View is available at this point
        metadata = integration.check_street_view_availability(lat, lng, radius)
        
        if metadata.get("status") != "OK":
            logger.warning(f"No Street View imagery available at ({lat}, {lng})")
            results["no_imagery_count"] += 1
            continue
        
        # If imagery exists, download it
        panorama_result = integration.fetch_panorama(lat, lng, point_dir, radius=radius)
        
        if panorama_result.get("success", False):
            results["success_count"] += 1
            results["downloaded_images"] += panorama_result.get("downloaded_images", 0)
            results["points_with_imagery"].append({
                "point_index": i,
                "requested_location": (lat, lng),
                "actual_location": panorama_result.get("location"),
                "image_count": panorama_result.get("downloaded_images", 0),
                "point_dir": point_dir
            })
        else:
            logger.error(f"Failed to download panorama at ({lat}, {lng}): {panorama_result.get('error', 'Unknown error')}")
            results["failed_count"] += 1
    
    # Save results summary
    results_file = os.path.join(output_dir, "street_view_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Street View processing complete. Results saved to {results_file}")
    
    return {
        "success": True,
        "points_processed": len(grid_points),
        "points_with_imagery": results["success_count"],
        "images_downloaded": results["downloaded_images"],
        "results_file": results_file
    }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Fetch Google Street View images")
    
    # Define operation modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Single point mode
    point_parser = subparsers.add_parser("point", help="Download Street View images for a single point")
    point_parser.add_argument("--coords", required=True, help="Coordinates as 'latitude,longitude'")
    point_parser.add_argument("--output", default="./street_view_output", help="Output directory")
    point_parser.add_argument("--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)")
    point_parser.add_argument("--heading", type=int, help="Specific heading (0-360)")
    point_parser.add_argument("--fov", type=int, default=90, help="Field of view (20-120)")
    point_parser.add_argument("--pitch", type=int, default=0, help="Camera pitch (-90 to 90)")
    point_parser.add_argument("--radius", type=int, default=100, help="Search radius in meters")
    point_parser.add_argument("--panorama", action="store_true", help="Capture full panorama")
    
    # Area mode
    area_parser = subparsers.add_parser("area", help="Download Street View images for an area")
    area_parser.add_argument("--bounds", required=True, help="Area bounds as 'north,south,east,west'")
    area_parser.add_argument("--output", default="./street_view_output", help="Output directory")
    area_parser.add_argument("--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)")
    area_parser.add_argument("--grid-size", type=float, default=100.0, help="Size of grid cells in meters")
    area_parser.add_argument("--fov", type=int, default=90, help="Field of view (20-120)")
    area_parser.add_argument("--radius", type=int, default=100, help="Search radius in meters")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if required modules are available
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Please check your installation.")
        return 1
    
    # Process based on mode
    if args.mode == "point":
        # Parse coordinates
        try:
            lat, lng = parse_coordinates(args.coords)
        except ValueError as e:
            logger.error(str(e))
            return 1
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Initialize integration
        integration = GoogleStreetViewIntegration(api_key=args.api_key)
        
        if args.panorama:
            logger.info(f"Fetching Street View panorama for location ({lat}, {lng})")
            result = integration.fetch_panorama(
                lat, lng, args.output,
                radius=args.radius
            )
        else:
            logger.info(f"Fetching Street View image for location ({lat}, {lng})")
            result = integration.fetch_street_view(
                lat, lng, args.output,
                heading=args.heading, fov=args.fov, pitch=args.pitch, radius=args.radius
            )
        
        if result.get("success", False):
            print(f"Successfully downloaded {result['downloaded_images']} Street View images")
            print(f"Images saved to {args.output}")
            return 0
        else:
            logger.error(f"Failed to download Street View imagery: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.mode == "area":
        try:
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
            
            # Import required modules for grid calculation
            try:
                import math
                import numpy as np
            except ImportError as e:
                logger.error(f"Required modules for grid calculation not available: {str(e)}")
                return 1
            
            # Process grid
            result = fetch_panorama_grid(
                bounds=bounds,
                output_dir=args.output,
                grid_size=args.grid_size,
                api_key=args.api_key,
                fov=args.fov,
                radius=args.radius
            )
            
            if result.get("success", False):
                print(f"Processed {result['points_processed']} grid points")
                print(f"Found imagery at {result['points_with_imagery']} locations")
                print(f"Downloaded {result['images_downloaded']} Street View images")
                print(f"Results saved to {result['results_file']}")
                return 0
            else:
                logger.error(f"Failed to process area: {result.get('error', 'Unknown error')}")
                return 1
            
        except Exception as e:
            logger.error(f"Error processing area: {str(e)}")
            return 1
    
    else:
        # No mode specified
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())