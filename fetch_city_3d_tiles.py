#!/usr/bin/env python3
"""
Fetch 3D Tiles for Entire Cities
-------------------------------
This script downloads Google Photorealistic 3D Tiles for entire cities,
ensuring complete coverage by dividing large cities into grid cells.
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, List, Any

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
    from integration_tools.google_3d_tiles_integration import fetch_city_tiles
    from integration_tools.spatial_bounds import city_polygon
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Make sure to install shapely and pyproj packages:")
    logger.error("pip install shapely pyproj")
    MODULES_AVAILABLE = False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Fetch Google 3D Tiles for entire cities")
    
    parser.add_argument("--city", required=True, help="City name (e.g., 'London', 'New York')")
    parser.add_argument("--output", default="./3d_tiles_output", help="Output directory")
    parser.add_argument("--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)")
    parser.add_argument("--region", help="Optional region name")
    parser.add_argument("--depth", type=int, default=4, help="Maximum recursion depth (default: 4)")
    parser.add_argument("--list-cities", action="store_true", help="List available predefined cities")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if modules are available
    if not MODULES_AVAILABLE:
        sys.exit(1)
    
    # List cities if requested
    if args.list_cities:
        try:
            from integration_tools.spatial_bounds import city_polygon
            # Extract city names from the predefined list
            # This is a bit hacky but works because we know the function has a dictionary of cities
            cities = []
            for name in city_polygon.__globals__["city_bounds"].keys():
                cities.append(name.title())
            
            print("Available predefined cities:")
            for city in sorted(cities):
                print(f"- {city}")
            return 0
        except Exception as e:
            logger.error(f"Error listing cities: {str(e)}")
            return 1
    
    # Fetch tiles for the specified city
    try:
        logger.info(f"Fetching 3D tiles for city: {args.city}")
        
        result = fetch_city_tiles(
            args.city,
            args.output,
            max_depth=args.depth,
            region=args.region,
            api_key=args.api_key
        )
        
        if result.get("success", False):
            downloaded_tiles = result.get("downloaded_tiles", 0)
            grid_cells = result.get("grid_cells", 0)
            
            logger.info(f"Successfully downloaded {downloaded_tiles} tiles for {args.city}")
            logger.info(f"The city was divided into {grid_cells} grid cells for processing")
            logger.info(f"Tiles saved to {args.output}")
            return 0
        else:
            logger.error(f"Failed to download 3D tiles for {args.city}: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())