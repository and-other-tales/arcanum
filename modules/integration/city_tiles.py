#!/usr/bin/env python3
"""
City 3D Tiles Fetcher
-------------------
This module provides functionality for fetching and processing Google Maps 3D Tiles 
for entire cities, ensuring complete coverage of urban areas.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional
import time

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import Google 3D Tiles integration
try:
    from modules.integration.google_3d_tiles_integration import Google3DTilesIntegration
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Make sure to install shapely and pyproj packages:")
    logger.error("pip install shapely pyproj")
    MODULES_AVAILABLE = False

def fetch_city_3d_tiles(city_name: str, output_dir: str, max_depth: int = 4, 
                     region: str = None, api_key: str = None) -> Dict:
    """
    Fetch 3D tiles for an entire city and save to output directory.
    
    This function ensures complete coverage of a city by efficiently dividing the
    city area into a grid of smaller areas if needed, and fetching tiles for each.
    
    Args:
        city_name: Name of the city (e.g., "London", "New York")
        output_dir: Directory to save downloaded tiles
        max_depth: Maximum recursion depth for tile fetching (4 recommended for cities)
        region: Optional region name (default global tileset)
        api_key: Google Maps API key with 3D Tiles access (from env var if None)
        
    Returns:
        Dictionary with download results
    """
    if not MODULES_AVAILABLE:
        return {
            "success": False,
            "error": "Required modules not available. Install shapely and pyproj."
        }
    
    try:
        # Create Google3DTilesIntegration instance
        integration = Google3DTilesIntegration(api_key=api_key)
        
        # Create output directory
        city_dir = os.path.join(output_dir, f"{city_name.lower().replace(' ', '_')}")
        os.makedirs(city_dir, exist_ok=True)
        
        # Log the request
        logger.info(f"Fetching 3D tiles for city '{city_name}' with max_depth={max_depth}")
        
        # Fetch the city tiles
        result = integration.fetch_city_tiles(city_name, city_dir, max_depth, region)
        
        if result.get("success", False):
            logger.info(f"Successfully fetched 3D tiles for {city_name}")
            logger.info(f"Downloaded {result.get('downloaded_tiles', 0)} tiles")
            
            # Save summary
            summary_path = os.path.join(city_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump({
                    "city": city_name,
                    "tiles_downloaded": result.get("downloaded_tiles", 0),
                    "grid_cells": result.get("grid_cells", 1),
                    "fetch_time": time.time()
                }, f, indent=2)
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching 3D tiles for city '{city_name}': {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="City 3D Tiles Fetcher")
    
    # Required arguments
    parser.add_argument("--city", required=True, help="City name (e.g., 'London', 'New York')")
    
    # Optional arguments
    parser.add_argument("--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)")
    parser.add_argument("--output-dir", default="./arcanum_3d_cities", 
                     help="Output directory for downloaded tiles")
    parser.add_argument("--depth", type=int, default=4, 
                     help="Maximum recursion depth for tile fetching (default: 4)")
    parser.add_argument("--region", help="Region name for tileset (default: global)")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Fetch city 3D tiles
    result = fetch_city_3d_tiles(
        city_name=args.city,
        output_dir=args.output_dir,
        max_depth=args.depth,
        region=args.region,
        api_key=args.api_key
    )
    
    # Print result
    if result.get("success", False):
        print(f"Successfully fetched 3D tiles for {args.city}")
        print(f"Downloaded {result.get('downloaded_tiles', 0)} tiles")
        print(f"Output directory: {os.path.join(args.output_dir, args.city.lower().replace(' ', '_'))}")
    else:
        print(f"Failed to fetch 3D tiles: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())