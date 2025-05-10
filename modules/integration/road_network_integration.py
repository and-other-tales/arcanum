#!/usr/bin/env python3
"""
Road Network Integration Module
------------------------------
Integrates road network utilities for systematic Street View collection and processing.
"""

import os
import sys
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import required modules
try:
    from integration_tools.road_network import RoadNetwork, sample_osm_roads
    from integration_tools.street_view_integration import GoogleStreetViewIntegration
    from integration_tools.coverage_verification import verify_street_view_coverage
    ROAD_NETWORK_AVAILABLE = True
except ImportError:
    ROAD_NETWORK_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RoadNetworkIntegration:
    """
    Integrates road network utilities for systematic Street View collection and processing.
    """
    
    def __init__(self, output_dir: str, api_key: Optional[str] = None):
        """
        Initialize the road network integration.
        
        Args:
            output_dir: Base directory for outputs
            api_key: Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)
        """
        self.output_dir = Path(output_dir)
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        
        # Create output directories
        self.road_network_dir = self.output_dir / "road_network"
        self.street_view_dir = self.output_dir / "street_view"
        self.coverage_dir = self.output_dir / "coverage"
        
        os.makedirs(self.road_network_dir, exist_ok=True)
        os.makedirs(self.street_view_dir, exist_ok=True)
        os.makedirs(self.coverage_dir, exist_ok=True)
        
        # Initialize road network components if available
        self.road_network = None
        
        if ROAD_NETWORK_AVAILABLE:
            self.street_view_integration = GoogleStreetViewIntegration(api_key=self.api_key)
        
        logger.info(f"Road Network Integration initialized with output directory: {output_dir}")
    
    def load_road_network(self, osm_path: str, layer: str = "roads") -> bool:
        """
        Load a road network from an OSM GeoPackage file.
        
        Args:
            osm_path: Path to OSM GeoPackage file
            layer: Layer name in the GeoPackage containing roads
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not ROAD_NETWORK_AVAILABLE:
            logger.error("Road network utilities not available. Install required packages.")
            return False
        
        try:
            self.road_network = RoadNetwork(osm_path, layer)
            
            # Save road network info
            info_path = self.road_network_dir / "road_network_info.json"
            with open(info_path, "w") as f:
                json.dump({
                    "osm_path": osm_path,
                    "layer": layer,
                    "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "node_count": self.road_network.graph.number_of_nodes() if self.road_network.graph else 0,
                    "edge_count": self.road_network.graph.number_of_edges() if self.road_network.graph else 0
                }, f, indent=2)
            
            logger.info(f"Loaded road network from {osm_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load road network: {str(e)}")
            return False
    
    def sample_points_along_roads(self, interval: float = 50.0, max_points: Optional[int] = None) -> Dict[str, Any]:
        """
        Sample points along all roads at a specified interval.

        Args:
            interval: Distance between sample points in meters
            max_points: Maximum number of points to return (None for all)

        Returns:
            Dictionary with sampling results
        """
        if not ROAD_NETWORK_AVAILABLE:
            return {"success": False, "error": "Road network utilities not available"}

        if not self.road_network:
            return {"success": False, "error": "No road network loaded. Call load_road_network() first."}

        try:
            # Sample points
            sample_points = self.road_network.sample_points_along_roads(interval)

            if not sample_points:
                return {"success": False, "error": "Failed to sample points along roads"}

            # Limit number of points if specified
            if max_points is not None and max_points < len(sample_points):
                logger.info(f"Limiting to {max_points} points")
                sample_points = sample_points[:max_points]

            # Save sample points to file
            points_path = self.road_network_dir / "street_view_sample_points.json"

            # Create GeoJSON feature collection
            feature_collection = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]  # GeoJSON uses [lon, lat]
                        },
                        "properties": {
                            "index": i
                        }
                    }
                    for i, (lat, lon) in enumerate(sample_points)
                ]
            }

            # Save to file
            with open(points_path, 'w') as f:
                json.dump(feature_collection, f, indent=2)

            logger.info(f"Sampled {len(sample_points)} points along roads")

            # Extract network data for visualization
            # This is used by the web interface to visualize the road network
            network_data = {
                "nodes": {},
                "edges": []
            }

            # Add nodes
            node_id = 0
            node_mapping = {}  # Map (lat, lon) tuples to node IDs

            for node in self.road_network.graph.nodes:
                if node not in node_mapping:
                    node_mapping[node] = f"n{node_id}"
                    network_data["nodes"][f"n{node_id}"] = {
                        "x": node[0],
                        "y": node[1],
                        "id": f"n{node_id}"
                    }
                    node_id += 1

            # Add edges
            for i, (start_node, end_node, attrs) in enumerate(self.road_network.graph.edges(data=True)):
                start_node_id = node_mapping[start_node]
                end_node_id = node_mapping[end_node]

                network_data["edges"].append({
                    "id": f"e{i}",
                    "start_node_id": start_node_id,
                    "end_node_id": end_node_id,
                    "length": attrs.get("length", 0),
                    "type": attrs.get("road_type", "unknown")
                })

            # Save network data for visualization
            network_path = self.road_network_dir / "road_network_visualization.json"
            with open(network_path, 'w') as f:
                json.dump(network_data, f, indent=2)

            return {
                "success": True,
                "num_points": len(sample_points),
                "points": sample_points,
                "points_path": str(points_path),
                "sampling_interval": interval,
                "network_data_path": str(network_path)
            }

        except Exception as e:
            logger.error(f"Error sampling points along roads: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def fetch_street_view_along_roads(self,
                                     sampling_interval: float = 50.0,
                                     max_points: Optional[int] = None,
                                     panorama: bool = True,
                                     max_search_radius: int = 1000,
                                     max_workers: int = 4,
                                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Fetch Street View images along road network.

        Args:
            sampling_interval: Distance between sample points in meters
            max_points: Maximum number of points to process (None for all)
            panorama: Whether to fetch full panorama
            max_search_radius: Maximum radius to search for nearby imagery
            max_workers: Maximum number of worker threads
            progress_callback: Optional callback function for progress updates (current, total)

        Returns:
            Dictionary with results
        """
        if not ROAD_NETWORK_AVAILABLE:
            return {"success": False, "error": "Road network utilities not available"}

        if not self.road_network:
            return {"success": False, "error": "No road network loaded. Call load_road_network() first."}

        try:
            # Create output directory for Street View imagery
            street_view_output_dir = self.street_view_dir / f"road_network_{int(sampling_interval)}m"
            os.makedirs(street_view_output_dir, exist_ok=True)

            # Sample points along roads first
            sample_result = self.sample_points_along_roads(sampling_interval)

            if not sample_result.get("success", False):
                return {"success": False, "error": f"Failed to sample road network: {sample_result.get('error', 'Unknown error')}"}

            # Load the sampled points
            with open(sample_result["points_path"], "r") as f:
                points_data = json.load(f)

            # Extract points from GeoJSON
            sample_points = []
            for feature in points_data["features"]:
                coords = feature["geometry"]["coordinates"]
                index = feature["properties"]["index"]
                # Convert from [lon, lat] to (lat, lon, index)
                sample_points.append((coords[1], coords[0], index))

            logger.info(f"Loaded {len(sample_points)} sample points")

            # Limit number of points if specified
            if max_points is not None and max_points < len(sample_points):
                logger.info(f"Limiting to {max_points} points")
                sample_points = sample_points[:max_points]

            # Use the Street View along roads functionality from integration_tools
            from integration_tools.street_view_integration import GoogleStreetViewIntegration

            # Process each point
            start_time = time.time()

            results = {
                "success_count": 0,
                "failed_count": 0,
                "no_imagery_count": 0,
                "total_images": 0,
                "points_with_imagery": []
            }

            # Create Street View integration if not already initialized
            street_view = GoogleStreetViewIntegration(api_key=self.api_key)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_point = {
                    executor.submit(
                        street_view.process_point,
                        lat, lng, idx, str(street_view_output_dir),
                        panorama, max_search_radius
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

                        # Call progress callback if provided
                        if progress_callback:
                            try:
                                progress_callback(i, len(sample_points))
                            except Exception as e:
                                logger.warning(f"Error in progress callback: {str(e)}")

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
            summary_path = street_view_output_dir / "street_view_summary.json"
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Completed processing {len(sample_points)} points in {elapsed:.1f}s")
            logger.info(f"Success: {results['success_count']}, No imagery: {results['no_imagery_count']}, Failed: {results['failed_count']}")
            logger.info(f"Downloaded {results['total_images']} Street View images")

            return {
                "success": True,
                "points_processed": len(sample_points),
                "points_with_imagery": results["success_count"],
                "images_downloaded": results["total_images"],
                "summary_path": str(summary_path),
                "output_dir": str(street_view_output_dir)
            }

        except Exception as e:
            logger.error(f"Error fetching Street View along roads: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def verify_coverage(self, buildings_file: str) -> Dict[str, Any]:
        """
        Verify Street View coverage for buildings.
        
        Args:
            buildings_file: Path to JSON file with building data
            
        Returns:
            Dictionary with verification results
        """
        if not ROAD_NETWORK_AVAILABLE:
            return {"success": False, "error": "Road network utilities not available"}
            
        try:
            # Check if verify_street_view_coverage is available
            if 'verify_street_view_coverage' not in globals():
                return {"success": False, "error": "Coverage verification utility not available"}
            
            # Load buildings data
            with open(buildings_file, "r") as f:
                buildings_data = json.load(f)
            
            # Verify coverage
            coverage_output_dir = self.coverage_dir / "building_coverage"
            os.makedirs(coverage_output_dir, exist_ok=True)
            
            coverage_result = verify_street_view_coverage(
                buildings_data,
                self.street_view_dir,
                str(coverage_output_dir),
                road_network=self.road_network
            )
            
            return coverage_result
            
        except Exception as e:
            logger.error(f"Error verifying coverage: {str(e)}")
            return {"success": False, "error": str(e)}


# Convenience functions for direct usage
def load_and_sample_road_network(osm_path: str, output_dir: str, interval: float = 50.0) -> Dict[str, Any]:
    """
    Load a road network and sample points along roads.
    
    Args:
        osm_path: Path to OSM GeoPackage file
        output_dir: Base directory for outputs
        interval: Distance between sample points in meters
        
    Returns:
        Dictionary with sampling results
    """
    integration = RoadNetworkIntegration(output_dir)
    
    if not integration.load_road_network(osm_path):
        return {"success": False, "error": "Failed to load road network"}
    
    return integration.sample_points_along_roads(interval)

def fetch_street_view_for_area(osm_path: str, output_dir: str, api_key: Optional[str] = None,
                               interval: float = 50.0, max_points: Optional[int] = None) -> Dict[str, Any]:
    """
    Fetch Street View images for an area defined by an OSM file.
    
    Args:
        osm_path: Path to OSM GeoPackage file
        output_dir: Base directory for outputs
        api_key: Google Maps API key
        interval: Distance between sample points in meters
        max_points: Maximum number of points to process
        
    Returns:
        Dictionary with Street View collection results
    """
    integration = RoadNetworkIntegration(output_dir, api_key)
    
    if not integration.load_road_network(osm_path):
        return {"success": False, "error": "Failed to load road network"}
    
    return integration.fetch_street_view_along_roads(
        sampling_interval=interval,
        max_points=max_points
    )


# Run a demo if this module is run directly
if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Road Network Integration")
    parser.add_argument("--osm", required=True, help="Path to OSM GeoPackage file")
    parser.add_argument("--output", default="./road_network_output", help="Output directory")
    parser.add_argument("--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)")
    parser.add_argument("--interval", type=float, default=50.0, help="Sampling interval in meters")
    parser.add_argument("--max-points", type=int, help="Maximum number of points to process")
    parser.add_argument("--panorama", dest="panorama", action="store_true", help="Fetch panoramas (default)")
    parser.add_argument("--no-panorama", dest="panorama", action="store_false", help="Don't fetch panoramas")
    parser.add_argument("--max-radius", type=int, default=1000, help="Maximum search radius in meters")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--sample-only", action="store_true", help="Only sample points, don't fetch Street View")
    parser.set_defaults(panorama=True)
    
    args = parser.parse_args()
    
    # Create integration
    integration = RoadNetworkIntegration(args.output, args.api_key)
    
    # Load road network
    if not integration.load_road_network(args.osm):
        sys.exit(1)
    
    # Sample points along roads if requested
    if args.sample_only:
        result = integration.sample_points_along_roads(args.interval)
        
        if result.get("success", False):
            print(f"Successfully sampled {result['num_points']} points along roads")
            print(f"Sample points saved to {result['points_path']}")
            sys.exit(0)
        else:
            print(f"Failed to sample points: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    # Fetch Street View along roads
    result = integration.fetch_street_view_along_roads(
        sampling_interval=args.interval,
        max_points=args.max_points,
        panorama=args.panorama,
        max_search_radius=args.max_radius,
        max_workers=args.workers
    )
    
    if result.get("success", False):
        print(f"Successfully processed {result['points_processed']} points")
        print(f"Found imagery at {result['points_with_imagery']} locations")
        print(f"Downloaded {result['images_downloaded']} Street View images")
        print(f"Results saved to {result['summary_path']}")
        sys.exit(0)
    else:
        print(f"Failed to fetch Street View: {result.get('error', 'Unknown error')}")
        sys.exit(1)