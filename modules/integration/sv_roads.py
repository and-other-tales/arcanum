#!/usr/bin/env python3
"""
Fetch Street View Along Roads
----------------------------
This script downloads Google Street View images along road networks extracted from OSM data,
ensuring comprehensive coverage of all streets.
"""

import os
import sys
import logging
import argparse
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
    from integration_tools.road_network import RoadNetwork, sample_osm_roads
    from integration_tools.street_view_integration import GoogleStreetViewIntegration, fetch_street_view
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Make sure to install required packages: geopandas, networkx, rtree, shapely")
    MODULES_AVAILABLE = False

def process_point(
    lat: float, 
    lng: float, 
    point_index: int, 
    output_dir: str, 
    api_key: str = None,
    panorama: bool = True,
    max_search_radius: int = 1000
) -> Dict[str, Any]:
    """Process a single point by fetching Street View imagery.
    
    Args:
        lat: Latitude of the point
        lng: Longitude of the point
        point_index: Index of the point for directory naming
        output_dir: Base output directory
        api_key: Google Maps API key
        panorama: Whether to fetch full panorama
        max_search_radius: Maximum radius to search for nearby imagery
        
    Returns:
        Dictionary with results for this point
    """
    try:
        # Create integration
        integration = GoogleStreetViewIntegration(api_key=api_key)
        
        # Create output directory for this point
        point_dir = os.path.join(output_dir, f"point_{point_index:06d}")
        os.makedirs(point_dir, exist_ok=True)
        
        # Save point location
        with open(os.path.join(point_dir, "location.json"), "w") as f:
            json.dump({
                "original": {"lat": lat, "lng": lng},
                "index": point_index
            }, f, indent=2)
        
        # Fetch Street View imagery
        if panorama:
            logger.info(f"Fetching panorama for point {point_index} at ({lat}, {lng})")
            result = integration.fetch_panorama(
                lat, lng, point_dir,
                max_search_radius=max_search_radius,
                find_nearest=True
            )
        else:
            logger.info(f"Fetching Street View for point {point_index} at ({lat}, {lng})")
            result = integration.fetch_street_view(
                lat, lng, point_dir,
                max_search_radius=max_search_radius,
                find_nearest=True
            )
        
        # Add point index to result
        result["point_index"] = point_index
        
        # Save result summary
        with open(os.path.join(point_dir, "result.json"), "w") as f:
            # Create a serializable result by removing non-serializable items
            serializable_result = {k: v for k, v in result.items() 
                                 if k not in ["paths"]}
            json.dump(serializable_result, f, indent=2)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing point {point_index} at ({lat}, {lng}): {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "point_index": point_index,
            "location": (lat, lng)
        }

def fetch_street_view_along_roads(
    osm_path: str,
    output_dir: str,
    sampling_interval: float = 50.0,
    max_points: Optional[int] = None,
    api_key: str = None,
    panorama: bool = True,
    max_search_radius: int = 1000,
    max_workers: int = 4
) -> Dict[str, Any]:
    """Fetch Street View images along road network.
    
    Args:
        osm_path: Path to OSM GeoPackage file with roads layer
        output_dir: Directory to save imagery
        sampling_interval: Distance between sample points in meters
        max_points: Maximum number of points to process (None for all)
        api_key: Google Maps API key
        panorama: Whether to fetch full panorama
        max_search_radius: Maximum radius to search for nearby imagery
        max_workers: Maximum number of worker threads
        
    Returns:
        Dictionary with results
    """
    try:
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # First sample points along road network
        logger.info(f"Sampling points along road network at {sampling_interval}m intervals")
        road_sample_dir = os.path.join(output_dir, "road_network")
        
        sample_result = sample_osm_roads(osm_path, road_sample_dir, sampling_interval)
        
        if not sample_result.get("success", False):
            return {
                "success": False,
                "error": f"Failed to sample road network: {sample_result.get('error', 'Unknown error')}"
            }
        
        # Load the sampled points
        with open(sample_result["points_path"], "r") as f:
            points_data = json.load(f)
        
        # Extract points from GeoJSON
        sample_points = []
        for feature in points_data["features"]:
            coords = feature["geometry"]["coordinates"]
            index = feature["properties"]["index"]
            # Convert from [lon, lat] to (lat, lon)
            sample_points.append((coords[1], coords[0], index))
        
        logger.info(f"Loaded {len(sample_points)} sample points")
        
        # Limit number of points if specified
        if max_points is not None and max_points < len(sample_points):
            logger.info(f"Limiting to {max_points} points")
            sample_points = sample_points[:max_points]
        
        # Create directory for Street View imagery
        street_view_dir = os.path.join(output_dir, "street_view")
        os.makedirs(street_view_dir, exist_ok=True)
        
        # Process points in parallel
        logger.info(f"Processing {len(sample_points)} points with {max_workers} workers")
        
        results = {
            "success_count": 0,
            "failed_count": 0,
            "no_imagery_count": 0,
            "total_images": 0,
            "points_with_imagery": []
        }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_point = {
                executor.submit(
                    process_point, 
                    lat, lng, idx, street_view_dir, 
                    api_key, panorama, max_search_radius
                ): (lat, lng, idx) 
                for lat, lng, idx in sample_points
            }
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_point), 1):
                point = future_to_point[future]
                try:
                    result = future.result()
                    
                    if result.get("success", False):
                        results["success_count"] += 1
                        results["total_images"] += result.get("downloaded_images", 0)
                        results["points_with_imagery"].append({
                            "point_index": result.get("point_index"),
                            "location": result.get("location"),
                            "original_location": result.get("original_location"),
                            "image_count": result.get("downloaded_images", 0),
                            "search_distance": result.get("search_distance", 0)
                        })
                    else:
                        if "No Street View imagery" in result.get("error", ""):
                            results["no_imagery_count"] += 1
                        else:
                            results["failed_count"] += 1
                    
                    # Log progress periodically
                    if i % 10 == 0 or i == len(sample_points):
                        elapsed = time.time() - start_time
                        logger.info(f"Processed {i}/{len(sample_points)} points ({i/len(sample_points)*100:.1f}%) in {elapsed:.1f}s")
                        logger.info(f"Success: {results['success_count']}, No imagery: {results['no_imagery_count']}, Failed: {results['failed_count']}")
                    
                except Exception as e:
                    logger.error(f"Error processing point {point}: {str(e)}")
                    results["failed_count"] += 1
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        # Add summary information
        results["total_points"] = len(sample_points)
        results["elapsed_seconds"] = elapsed
        results["points_per_second"] = len(sample_points) / elapsed if elapsed > 0 else 0
        
        # Save summary
        summary_path = os.path.join(output_dir, "street_view_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Completed processing {len(sample_points)} points in {elapsed:.1f}s")
        logger.info(f"Success: {results['success_count']}, No imagery: {results['no_imagery_count']}, Failed: {results['failed_count']}")
        logger.info(f"Downloaded {results['total_images']} Street View images")
        logger.info(f"Summary saved to {summary_path}")
        
        return {
            "success": True,
            "points_processed": len(sample_points),
            "points_with_imagery": results["success_count"],
            "images_downloaded": results["total_images"],
            "summary_path": summary_path
        }
        
    except Exception as e:
        logger.error(f"Error fetching Street View along roads: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Fetch Street View images along road networks")
    
    parser.add_argument("--osm", required=True, help="Path to OSM GeoPackage file with roads layer")
    parser.add_argument("--output", default="./street_view_roads_output", help="Output directory")
    parser.add_argument("--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)")
    parser.add_argument("--interval", type=float, default=50.0, help="Sampling interval in meters")
    parser.add_argument("--max-points", type=int, help="Maximum number of points to process")
    parser.add_argument("--max-radius", type=int, default=1000, help="Maximum search radius in meters")
    parser.add_argument("--no-panorama", action="store_true", help="Don't fetch full panoramas")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if modules are available
    if not MODULES_AVAILABLE:
        sys.exit(1)
    
    # Process request
    try:
        logger.info(f"Fetching Street View along roads from {args.osm}")
        
        result = fetch_street_view_along_roads(
            args.osm,
            args.output,
            sampling_interval=args.interval,
            max_points=args.max_points,
            api_key=args.api_key,
            panorama=not args.no_panorama,
            max_search_radius=args.max_radius,
            max_workers=args.workers
        )
        
        if result.get("success", False):
            logger.info(f"Successfully processed {result['points_processed']} points")
            logger.info(f"Found imagery at {result['points_with_imagery']} locations")
            logger.info(f"Downloaded {result['images_downloaded']} Street View images")
            logger.info(f"Results saved to {result['summary_path']}")
            return 0
        else:
            logger.error(f"Failed to process request: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())