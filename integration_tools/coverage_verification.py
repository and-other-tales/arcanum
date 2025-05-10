#!/usr/bin/env python3
"""
Coverage Verification Tools
-------------------------
This module provides utilities for verifying the coverage of 3D Tiles and Street View imagery
against OpenStreetMap road network and area bounds.
"""

import os
import sys
import logging
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, box, mapping, shape
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from pathlib import Path

# Import our modules if available
try:
    from integration_tools.spatial_bounds import bounds_to_polygon, polygon_to_bounds
    from integration_tools.road_network import RoadNetwork
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class CoverageVerifier:
    """Class for verifying coverage of geographic data."""

    def __init__(self, 
                 bounds: Dict[str, float] = None,
                 road_network_path: str = None,
                 output_dir: str = None):
        """Initialize coverage verifier.
        
        Args:
            bounds: Dictionary with north, south, east, west bounds
            road_network_path: Path to OSM GeoPackage with roads
            output_dir: Directory to save verification outputs
        """
        self.bounds = bounds
        self.bounds_poly = bounds_to_polygon(bounds) if bounds and MODULES_AVAILABLE else None
        self.road_network = RoadNetwork(road_network_path) if road_network_path else None
        self.output_dir = output_dir or os.getcwd()
        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
    
    def set_bounds(self, bounds: Dict[str, float]):
        """Set the bounds for verification.
        
        Args:
            bounds: Dictionary with north, south, east, west bounds
        """
        self.bounds = bounds
        self.bounds_poly = bounds_to_polygon(bounds) if MODULES_AVAILABLE else None
    
    def load_road_network(self, road_network_path: str) -> bool:
        """Load a road network for verification.
        
        Args:
            road_network_path: Path to OSM GeoPackage with roads
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(road_network_path):
            logger.error(f"Road network file not found: {road_network_path}")
            return False
        
        self.road_network = RoadNetwork(road_network_path)
        return self.road_network is not None
    
    def verify_3d_tiles_coverage(self, 
                                tiles_dir: str, 
                                verify_bounds: bool = True,
                                generate_report: bool = True) -> Dict[str, Any]:
        """Verify coverage of 3D tiles against area bounds.
        
        Args:
            tiles_dir: Directory containing 3D tiles
            verify_bounds: Whether to verify coverage against bounds
            generate_report: Whether to generate a coverage report
            
        Returns:
            Dictionary with verification results
        """
        try:
            logger.info(f"Verifying 3D tiles coverage in {tiles_dir}")
            
            # Check if directory exists
            if not os.path.isdir(tiles_dir):
                return {
                    "success": False,
                    "error": f"Tiles directory not found: {tiles_dir}"
                }
            
            # Get tileset.json
            tileset_path = os.path.join(tiles_dir, "tileset.json")
            if not os.path.exists(tileset_path):
                return {
                    "success": False,
                    "error": f"Tileset.json not found in {tiles_dir}"
                }
            
            # Load tileset
            with open(tileset_path, "r") as f:
                tileset = json.load(f)
            
            # Extract tileset bounds if available
            tileset_bounds = None
            if "root" in tileset and "boundingVolume" in tileset["root"]:
                bounding_volume = tileset["root"]["boundingVolume"]
                if "region" in bounding_volume:
                    region = bounding_volume["region"]
                    if len(region) >= 4:
                        # Convert from radians to degrees
                        tileset_bounds = {
                            "west": math.degrees(region[0]),
                            "south": math.degrees(region[1]),
                            "east": math.degrees(region[2]),
                            "north": math.degrees(region[3])
                        }
                        logger.info(f"Found tileset bounds: {tileset_bounds}")
            
            # Count tiles by type
            tile_counts = {
                "json": 0,
                "b3dm": 0,
                "pnts": 0,
                "cmpt": 0,
                "i3dm": 0,
                "glb": 0,
                "gltf": 0,
                "other": 0
            }
            
            # Walk through directory and count files
            for root, dirs, files in os.walk(tiles_dir):
                for file in files:
                    ext = file.split(".")[-1].lower()
                    if ext in tile_counts:
                        tile_counts[ext] += 1
                    else:
                        tile_counts["other"] += 1
            
            results = {
                "success": True,
                "tileset_path": tileset_path,
                "tileset_bounds": tileset_bounds,
                "tiles_by_type": tile_counts,
                "total_tiles": sum(tile_counts.values())
            }
            
            # Verify coverage against bounds if requested
            if verify_bounds and self.bounds and tileset_bounds:
                coverage_analysis = self._analyze_bounds_coverage(self.bounds, tileset_bounds)
                results["bounds_coverage"] = coverage_analysis
            
            # Generate report if requested
            if generate_report:
                report_path = os.path.join(self.output_dir, "3d_tiles_coverage_report.json")
                with open(report_path, "w") as f:
                    json.dump(results, f, indent=2)
                
                # Generate visualization if bounds are available
                if tileset_bounds and self.bounds:
                    vis_path = os.path.join(self.output_dir, "3d_tiles_coverage.png")
                    self._visualize_bounds_coverage(self.bounds, tileset_bounds, vis_path)
                    results["visualization_path"] = vis_path
                
                results["report_path"] = report_path
                logger.info(f"Generated 3D tiles coverage report: {report_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error verifying 3D tiles coverage: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def verify_street_view_coverage(self, 
                                  street_view_dir: str,
                                  summary_path: str = None,
                                  verify_roads: bool = True,
                                  generate_report: bool = True) -> Dict[str, Any]:
        """Verify coverage of Street View imagery against road network.
        
        Args:
            street_view_dir: Directory containing Street View imagery
            summary_path: Path to Street View summary JSON
            verify_roads: Whether to verify coverage against road network
            generate_report: Whether to generate a coverage report
            
        Returns:
            Dictionary with verification results
        """
        try:
            logger.info(f"Verifying Street View coverage in {street_view_dir}")
            
            # Check if directory exists
            if not os.path.isdir(street_view_dir):
                return {
                    "success": False,
                    "error": f"Street View directory not found: {street_view_dir}"
                }
            
            # Locate summary file if not provided
            if not summary_path:
                # Try common paths for summary file
                possible_paths = [
                    os.path.join(os.path.dirname(street_view_dir), "street_view_summary.json"),
                    os.path.join(street_view_dir, "street_view_summary.json"),
                    os.path.join(street_view_dir, "summary.json")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        summary_path = path
                        break
            
            # Load summary if available
            summary_data = None
            if summary_path and os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    summary_data = json.load(f)
                logger.info(f"Loaded Street View summary from {summary_path}")
            
            # Collect info about all point directories
            point_dirs = []
            
            for item in os.listdir(street_view_dir):
                item_path = os.path.join(street_view_dir, item)
                if os.path.isdir(item_path) and item.startswith("point_"):
                    point_dirs.append(item_path)
            
            logger.info(f"Found {len(point_dirs)} point directories")
            
            # Load metadata for each point
            points_with_imagery = []
            points_without_imagery = []
            
            for point_dir in point_dirs:
                # Check for metadata.json
                metadata_path = os.path.join(point_dir, "metadata.json")
                result_path = os.path.join(point_dir, "result.json")
                location_path = os.path.join(point_dir, "location.json")
                
                location_data = None
                if os.path.exists(location_path):
                    with open(location_path, "r") as f:
                        location_data = json.load(f)
                
                # Check if imagery was found
                has_imagery = False
                
                # Check result.json
                if os.path.exists(result_path):
                    with open(result_path, "r") as f:
                        result_data = json.load(f)
                    
                    if result_data.get("success", False):
                        has_imagery = True
                        
                        # Extract coordinates
                        if "location" in result_data:
                            lat, lng = result_data["location"]
                            
                            point_info = {
                                "dir": os.path.basename(point_dir),
                                "location": {"lat": lat, "lng": lng},
                                "image_count": result_data.get("downloaded_images", 0)
                            }
                            
                            # Add search distance if available
                            if "search_distance" in result_data:
                                point_info["search_distance"] = result_data["search_distance"]
                            
                            # Add original location if available
                            if "original_location" in result_data:
                                point_info["original_location"] = {
                                    "lat": result_data["original_location"][0],
                                    "lng": result_data["original_location"][1]
                                }
                            elif location_data and "original" in location_data:
                                point_info["original_location"] = location_data["original"]
                            
                            points_with_imagery.append(point_info)
                
                # If no imagery, add to without list
                if not has_imagery:
                    # Try to get original location from location.json
                    original_location = None
                    if location_data and "original" in location_data:
                        original_location = location_data["original"]
                    
                    points_without_imagery.append({
                        "dir": os.path.basename(point_dir),
                        "original_location": original_location
                    })
            
            # Prepare results
            results = {
                "success": True,
                "total_points": len(point_dirs),
                "points_with_imagery": len(points_with_imagery),
                "points_without_imagery": len(points_without_imagery),
                "coverage_percentage": (len(points_with_imagery) / len(point_dirs) * 100) if point_dirs else 0
            }
            
            # Add summary data if available
            if summary_data:
                results["summary_data"] = {
                    "total_points": summary_data.get("total_points", 0),
                    "success_count": summary_data.get("success_count", 0),
                    "no_imagery_count": summary_data.get("no_imagery_count", 0),
                    "failed_count": summary_data.get("failed_count", 0),
                    "total_images": summary_data.get("total_images", 0)
                }
            
            # Verify road coverage if requested
            if verify_roads and self.road_network and self.road_network.roads_gdf is not None:
                coverage_analysis = self._analyze_road_coverage(points_with_imagery)
                results["road_coverage"] = coverage_analysis
            
            # Generate report if requested
            if generate_report:
                report_path = os.path.join(self.output_dir, "street_view_coverage_report.json")
                
                # Save detailed point info
                detailed_report = {
                    **results,
                    "points_with_imagery_details": points_with_imagery,
                    "points_without_imagery_details": points_without_imagery
                }
                
                with open(report_path, "w") as f:
                    json.dump(detailed_report, f, indent=2)
                
                # Generate visualization of road coverage
                if verify_roads and self.road_network and self.road_network.roads_gdf is not None:
                    vis_path = os.path.join(self.output_dir, "street_view_coverage.png")
                    self._visualize_road_coverage(points_with_imagery, vis_path)
                    results["visualization_path"] = vis_path
                
                results["report_path"] = report_path
                logger.info(f"Generated Street View coverage report: {report_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error verifying Street View coverage: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_bounds_coverage(self, target_bounds: Dict[str, float], 
                               actual_bounds: Dict[str, float]) -> Dict[str, Any]:
        """Analyze how well the actual bounds cover the target bounds.
        
        Args:
            target_bounds: The intended coverage bounds
            actual_bounds: The actual covered bounds
            
        Returns:
            Dictionary with coverage analysis
        """
        # Convert to polygons
        target_poly = bounds_to_polygon(target_bounds)
        actual_poly = bounds_to_polygon(actual_bounds)
        
        # Calculate areas
        target_area = target_poly.area
        actual_area = actual_poly.area
        
        # Calculate intersection
        intersection = target_poly.intersection(actual_poly)
        intersection_area = intersection.area
        
        # Calculate coverage metrics
        coverage_percentage = (intersection_area / target_area) * 100 if target_area > 0 else 0
        excess_percentage = ((actual_area - intersection_area) / actual_area) * 100 if actual_area > 0 else 0
        
        return {
            "target_bounds": target_bounds,
            "actual_bounds": actual_bounds,
            "coverage_percentage": coverage_percentage,
            "excess_percentage": excess_percentage,
            "is_complete_coverage": coverage_percentage >= 99.9
        }
    
    def _analyze_road_coverage(self, points_with_imagery: List[Dict]) -> Dict[str, Any]:
        """Analyze how well Street View points cover the road network.
        
        Args:
            points_with_imagery: List of points with Street View imagery
            
        Returns:
            Dictionary with coverage analysis
        """
        if not self.road_network or self.road_network.roads_gdf is None:
            return {
                "error": "Road network not available"
            }
        
        try:
            # Extract coordinates from points
            image_points = []
            for point in points_with_imagery:
                if "location" in point:
                    lat = point["location"]["lat"]
                    lng = point["location"]["lng"]
                    image_points.append(Point(lng, lat))
            
            # Convert to GeoDataFrame
            points_gdf = gpd.GeoDataFrame(geometry=image_points)
            
            # Create buffers around points (50m radius)
            buffered_points = points_gdf.copy()
            buffered_points.geometry = points_gdf.geometry.buffer(0.0005)  # ~50m in decimal degrees
            
            # Merge all buffers
            if len(buffered_points) > 0:
                coverage_area = buffered_points.unary_union
            else:
                coverage_area = Polygon()
            
            # Get total road length
            total_road_length = 0
            for _, road in self.road_network.roads_gdf.iterrows():
                total_road_length += road.geometry.length
            
            # Calculate covered road length
            covered_road_length = 0
            covered_roads = 0
            
            for _, road in self.road_network.roads_gdf.iterrows():
                # Check if road intersects with any point buffer
                if road.geometry.intersects(coverage_area):
                    intersection = road.geometry.intersection(coverage_area)
                    intersection_length = intersection.length
                    covered_road_length += intersection_length
                    
                    # If more than 80% of the road is covered, count as fully covered
                    if intersection_length / road.geometry.length >= 0.8:
                        covered_roads += 1
            
            # Calculate coverage metrics
            length_coverage_percentage = (covered_road_length / total_road_length * 100) if total_road_length > 0 else 0
            road_count_coverage = (covered_roads / len(self.road_network.roads_gdf) * 100) if len(self.road_network.roads_gdf) > 0 else 0
            
            return {
                "total_road_count": len(self.road_network.roads_gdf),
                "covered_road_count": covered_roads,
                "road_count_coverage_percentage": road_count_coverage,
                "total_road_length": total_road_length,
                "covered_road_length": covered_road_length,
                "length_coverage_percentage": length_coverage_percentage,
                "is_complete_coverage": length_coverage_percentage >= 90.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing road coverage: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _visualize_bounds_coverage(self, target_bounds: Dict[str, float], 
                                 actual_bounds: Dict[str, float],
                                 output_path: str):
        """Visualize bounds coverage.
        
        Args:
            target_bounds: The intended coverage bounds
            actual_bounds: The actual covered bounds
            output_path: Path to save visualization
        """
        try:
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Convert to polygons
            target_poly = bounds_to_polygon(target_bounds)
            actual_poly = bounds_to_polygon(actual_bounds)
            
            # Get intersection
            intersection = target_poly.intersection(actual_poly)
            
            # Plot target bounds
            x, y = target_poly.exterior.xy
            plt.plot(x, y, 'b-', linewidth=2, label='Target Area')
            
            # Plot actual bounds
            x, y = actual_poly.exterior.xy
            plt.plot(x, y, 'r-', linewidth=2, label='Actual Coverage')
            
            # Shade intersection
            if not intersection.is_empty:
                if intersection.geom_type == 'Polygon':
                    x, y = intersection.exterior.xy
                    plt.fill(x, y, alpha=0.3, color='green', label='Covered Area')
                elif intersection.geom_type == 'MultiPolygon':
                    for poly in intersection.geoms:
                        x, y = poly.exterior.xy
                        plt.fill(x, y, alpha=0.3, color='green')
                    plt.plot([], [], alpha=0.3, color='green', label='Covered Area')  # For legend
            
            # Add title and legend
            coverage_percentage = (intersection.area / target_poly.area) * 100 if target_poly.area > 0 else 0
            plt.title(f'3D Tiles Coverage Analysis\nCoverage: {coverage_percentage:.1f}%')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved bounds coverage visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing bounds coverage: {str(e)}")
    
    def _visualize_road_coverage(self, points_with_imagery: List[Dict], output_path: str):
        """Visualize road coverage with Street View points.
        
        Args:
            points_with_imagery: List of points with Street View imagery
            output_path: Path to save visualization
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot road network
            if self.road_network and self.road_network.roads_gdf is not None:
                self.road_network.roads_gdf.plot(ax=plt.gca(), color='gray', linewidth=0.5, alpha=0.7)
            
            # Extract coordinates from points
            lats = []
            lngs = []
            distances = []
            
            for point in points_with_imagery:
                if "location" in point:
                    lats.append(point["location"]["lat"])
                    lngs.append(point["location"]["lng"])
                    distances.append(point.get("search_distance", 0))
            
            # Create colormap based on search distance
            if distances:
                max_dist = max(distances) if max(distances) > 0 else 100
                norm = plt.Normalize(0, max_dist)
                cmap = cm.get_cmap('viridis_r')
                
                # Plot points with color based on search distance
                scatter = plt.scatter(lngs, lats, c=distances, cmap=cmap, norm=norm, s=30, alpha=0.7)
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Search Distance (m)')
            else:
                # Simple plot if no distance info
                plt.scatter(lngs, lats, color='blue', s=30, alpha=0.7)
            
            # Add title
            plt.title(f'Street View Coverage\n{len(points_with_imagery)} points with imagery')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved road coverage visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing road coverage: {str(e)}")

def main():
    """Main function for testing the module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Coverage Verification Tools")
    parser.add_argument("--mode", choices=["3d", "street-view", "both"], required=True, help="Verification mode")
    parser.add_argument("--bounds", help="Bounds as 'north,south,east,west'")
    parser.add_argument("--city", help="City name for predefined bounds")
    parser.add_argument("--osm", help="Path to OSM GeoPackage with roads")
    parser.add_argument("--tiles-dir", help="Directory containing 3D tiles")
    parser.add_argument("--street-view-dir", help="Directory containing Street View imagery")
    parser.add_argument("--output", default="./coverage_verification", help="Output directory")
    parser.add_argument("--summary", help="Path to Street View summary JSON")
    args = parser.parse_args()
    
    # Check for required modules
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Install shapely and pyproj packages.")
        return 1
    
    # Parse bounds if provided
    bounds = None
    if args.bounds:
        bounds_values = args.bounds.split(",")
        if len(bounds_values) == 4:
            bounds = {
                "north": float(bounds_values[0]),
                "south": float(bounds_values[1]),
                "east": float(bounds_values[2]),
                "west": float(bounds_values[3])
            }
    
    # Create verifier
    verifier = CoverageVerifier(bounds=bounds, road_network_path=args.osm, output_dir=args.output)
    
    # Verify 3D tiles if requested
    if args.mode in ["3d", "both"]:
        if not args.tiles_dir:
            logger.error("--tiles-dir required for 3D tiles verification")
            return 1
        
        result = verifier.verify_3d_tiles_coverage(args.tiles_dir)
        
        if result.get("success", False):
            coverage = result.get("bounds_coverage", {})
            if coverage:
                logger.info(f"3D Tiles coverage: {coverage.get('coverage_percentage', 0):.1f}%")
                logger.info(f"Complete coverage: {coverage.get('is_complete_coverage', False)}")
            
            logger.info(f"Total tiles: {result.get('total_tiles', 0)}")
            logger.info(f"Report saved to: {result.get('report_path', 'N/A')}")
            
            if "visualization_path" in result:
                logger.info(f"Visualization saved to: {result['visualization_path']}")
        else:
            logger.error(f"3D Tiles verification failed: {result.get('error', 'Unknown error')}")
    
    # Verify Street View if requested
    if args.mode in ["street-view", "both"]:
        if not args.street_view_dir:
            logger.error("--street-view-dir required for Street View verification")
            return 1
        
        result = verifier.verify_street_view_coverage(args.street_view_dir, args.summary)
        
        if result.get("success", False):
            coverage = result.get("road_coverage", {})
            if coverage and "error" not in coverage:
                logger.info(f"Street View coverage by length: {coverage.get('length_coverage_percentage', 0):.1f}%")
                logger.info(f"Street View coverage by road count: {coverage.get('road_count_coverage_percentage', 0):.1f}%")
                logger.info(f"Complete coverage: {coverage.get('is_complete_coverage', False)}")
            
            logger.info(f"Total points: {result.get('total_points', 0)}")
            logger.info(f"Points with imagery: {result.get('points_with_imagery', 0)}")
            logger.info(f"Coverage percentage: {result.get('coverage_percentage', 0):.1f}%")
            logger.info(f"Report saved to: {result.get('report_path', 'N/A')}")
            
            if "visualization_path" in result:
                logger.info(f"Visualization saved to: {result['visualization_path']}")
        else:
            logger.error(f"Street View verification failed: {result.get('error', 'Unknown error')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())