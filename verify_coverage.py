#!/usr/bin/env python3
"""
Verify Data Coverage
------------------
This script verifies the coverage of 3D Tiles and Street View imagery against
OpenStreetMap road network and city bounds.
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, Any

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
    from integration_tools.coverage_verification import CoverageVerifier
    from integration_tools.spatial_bounds import city_polygon, polygon_to_bounds
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Make sure to install required packages: matplotlib, geopandas, networkx, rtree, shapely, pyproj")
    MODULES_AVAILABLE = False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Verify data coverage for Arcanum project")
    
    parser.add_argument("--mode", choices=["3d", "street-view", "both"], default="both",
                      help="Verification mode")
    
    parser.add_argument("--city", help="City name for predefined bounds (e.g., 'London')")
    parser.add_argument("--bounds", help="Manual bounds specification as 'north,south,east,west'")
    
    parser.add_argument("--osm", help="Path to OSM GeoPackage with roads layer")
    parser.add_argument("--tiles-dir", help="Directory containing 3D tiles")
    parser.add_argument("--street-view-dir", help="Directory containing Street View imagery")
    
    parser.add_argument("--output", default="./coverage_verification",
                      help="Output directory for verification reports and visualizations")
    
    parser.add_argument("--list-cities", action="store_true",
                      help="List available predefined cities")
    
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
    
    # Determine bounds
    bounds = None
    
    # From city name
    if args.city:
        city_poly = city_polygon(args.city)
        if city_poly:
            bounds = polygon_to_bounds(city_poly)
            logger.info(f"Using bounds for city: {args.city}")
        else:
            logger.error(f"City '{args.city}' not found in predefined cities")
            return 1
    
    # From manual bounds
    elif args.bounds:
        try:
            bounds_values = args.bounds.split(",")
            if len(bounds_values) == 4:
                bounds = {
                    "north": float(bounds_values[0]),
                    "south": float(bounds_values[1]),
                    "east": float(bounds_values[2]),
                    "west": float(bounds_values[3])
                }
                logger.info(f"Using manual bounds: {bounds}")
            else:
                logger.error("Invalid bounds format. Use: north,south,east,west")
                return 1
        except ValueError:
            logger.error("Invalid bounds format. All values must be numbers.")
            return 1
    
    # Check required parameters
    if args.mode in ["3d", "both"] and not args.tiles_dir:
        logger.error("--tiles-dir required for 3D tiles verification")
        return 1
    
    if args.mode in ["street-view", "both"] and not args.street_view_dir:
        logger.error("--street-view-dir required for Street View verification")
        return 1
    
    if args.mode in ["street-view", "both"] and not args.osm:
        logger.warning("--osm not provided. Road coverage verification will be limited.")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create verifier
    verifier = CoverageVerifier(
        bounds=bounds,
        road_network_path=args.osm,
        output_dir=args.output
    )
    
    # Run verification
    combined_results = {
        "mode": args.mode,
        "city": args.city,
        "bounds": bounds,
        "osm_path": args.osm,
        "tiles_dir": args.tiles_dir,
        "street_view_dir": args.street_view_dir
    }
    
    # Verify 3D tiles if requested
    if args.mode in ["3d", "both"]:
        try:
            result = verifier.verify_3d_tiles_coverage(args.tiles_dir)
            combined_results["3d_tiles_results"] = result
            
            if result.get("success", False):
                coverage = result.get("bounds_coverage", {})
                
                logger.info("3D Tiles Verification Results:")
                logger.info(f"  Total tiles: {result.get('total_tiles', 0)}")
                
                if coverage:
                    logger.info(f"  Area coverage: {coverage.get('coverage_percentage', 0):.1f}%")
                    logger.info(f"  Complete coverage: {coverage.get('is_complete_coverage', False)}")
                
                logger.info(f"  Report: {result.get('report_path', 'N/A')}")
                
                if "visualization_path" in result:
                    logger.info(f"  Visualization: {result['visualization_path']}")
            else:
                logger.error(f"3D Tiles verification failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error during 3D Tiles verification: {str(e)}")
            combined_results["3d_tiles_error"] = str(e)
    
    # Verify Street View if requested
    if args.mode in ["street-view", "both"]:
        try:
            result = verifier.verify_street_view_coverage(args.street_view_dir)
            combined_results["street_view_results"] = result
            
            if result.get("success", False):
                road_coverage = result.get("road_coverage", {})
                
                logger.info("Street View Verification Results:")
                logger.info(f"  Total points: {result.get('total_points', 0)}")
                logger.info(f"  Points with imagery: {result.get('points_with_imagery', 0)}")
                logger.info(f"  Coverage percentage: {result.get('coverage_percentage', 0):.1f}%")
                
                if road_coverage and "error" not in road_coverage:
                    logger.info(f"  Road coverage by length: {road_coverage.get('length_coverage_percentage', 0):.1f}%")
                    logger.info(f"  Road coverage by count: {road_coverage.get('road_count_coverage_percentage', 0):.1f}%")
                    logger.info(f"  Complete road coverage: {road_coverage.get('is_complete_coverage', False)}")
                
                logger.info(f"  Report: {result.get('report_path', 'N/A')}")
                
                if "visualization_path" in result:
                    logger.info(f"  Visualization: {result['visualization_path']}")
            else:
                logger.error(f"Street View verification failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error during Street View verification: {str(e)}")
            combined_results["street_view_error"] = str(e)
    
    # Save combined results
    combined_report_path = os.path.join(args.output, "coverage_verification_summary.json")
    with open(combined_report_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"Combined verification report saved to: {combined_report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())