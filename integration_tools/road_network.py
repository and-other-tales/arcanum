#!/usr/bin/env python3
"""
Road Network Utilities
---------------------
This module provides utilities for working with road networks from OpenStreetMap data,
particularly for systematically traversing roads for Street View collection.
"""

import os
import sys
import logging
import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
import networkx as nx
from rtree import index

# Setup logging
logger = logging.getLogger(__name__)

class RoadNetwork:
    """Class for managing road networks extracted from OSM data."""

    def __init__(self, gpkg_path: str = None, layer: str = "roads"):
        """Initialize a road network from a GeoPackage file.
        
        Args:
            gpkg_path: Path to the GeoPackage file containing road data
            layer: Layer name in the GeoPackage containing roads
        """
        self.roads_gdf = None
        self.graph = None
        self.spatial_index = None
        
        if gpkg_path:
            self.load_from_gpkg(gpkg_path, layer)
    
    def load_from_gpkg(self, gpkg_path: str, layer: str = "roads") -> bool:
        """Load road data from a GeoPackage file.
        
        Args:
            gpkg_path: Path to the GeoPackage file
            layer: Layer name in the GeoPackage
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading road network from {gpkg_path}, layer {layer}")
            self.roads_gdf = gpd.read_file(gpkg_path, layer=layer)
            
            # Filter to only include highways/roads
            if 'highway' in self.roads_gdf.columns:
                self.roads_gdf = self.roads_gdf[self.roads_gdf['highway'].notna()]
                logger.info(f"Filtered to {len(self.roads_gdf)} road features")
            
            # Create spatial index
            self._create_spatial_index()
            
            # Create network graph
            self._create_network_graph()
            
            return True
        except Exception as e:
            logger.error(f"Error loading road network: {str(e)}")
            return False
    
    def _create_spatial_index(self):
        """Create spatial index for efficient spatial queries."""
        try:
            # Create R-tree index
            self.spatial_index = index.Index()
            
            # Index each road segment
            for idx, row in self.roads_gdf.iterrows():
                # Get bounds of the geometry
                bounds = row.geometry.bounds
                self.spatial_index.insert(idx, bounds)
                
            logger.info(f"Created spatial index with {len(self.roads_gdf)} entries")
        except Exception as e:
            logger.error(f"Error creating spatial index: {str(e)}")
            self.spatial_index = None
    
    def _create_network_graph(self):
        """Create a NetworkX graph from the road network for routing."""
        try:
            # Create a new undirected graph
            self.graph = nx.Graph()
            
            # Add edges for each road segment
            for idx, row in self.roads_gdf.iterrows():
                geom = row.geometry
                
                # Handle MultiLineString geometries
                if geom.geom_type == 'MultiLineString':
                    lines = [line for line in geom.geoms]
                else:
                    lines = [geom]
                
                # Process each line
                for line in lines:
                    # Skip invalid geometries
                    if line.is_empty:
                        continue
                    
                    # Extract coordinates
                    coords = list(line.coords)
                    
                    # Create edges between consecutive points
                    for i in range(len(coords) - 1):
                        start_node = coords[i]
                        end_node = coords[i + 1]
                        
                        # Calculate length between points (approximate in meters)
                        length = self._calculate_distance(start_node, end_node)
                        
                        # Add edge with relevant attributes
                        self.graph.add_edge(
                            start_node, end_node,
                            length=length,
                            road_id=idx,
                            road_type=row.get('highway', 'unknown')
                        )
            
            logger.info(f"Created network graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Error creating network graph: {str(e)}")
            self.graph = None
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate the distance between two coordinates in meters.
        
        Args:
            coord1: (longitude, latitude) of first point
            coord2: (longitude, latitude) of second point
            
        Returns:
            Distance in meters
        """
        # Extract coordinates
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Radius of the Earth in meters
        earth_radius = 6371000
        
        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = earth_radius * c
        
        return distance
    
    def sample_points_along_roads(self, interval: float = 50.0) -> List[Tuple[float, float]]:
        """Sample points along all roads at a specified interval.
        
        Args:
            interval: Distance between sample points in meters
            
        Returns:
            List of (latitude, longitude) tuples representing sample points
        """
        if self.roads_gdf is None:
            logger.error("No road data loaded. Load road data first.")
            return []
        
        sample_points = []
        
        try:
            logger.info(f"Sampling points along roads at {interval}m intervals")
            
            for idx, row in self.roads_gdf.iterrows():
                geom = row.geometry
                
                # Handle MultiLineString geometries
                if geom.geom_type == 'MultiLineString':
                    lines = [line for line in geom.geoms]
                else:
                    lines = [geom]
                
                # Process each line
                for line in lines:
                    # Skip invalid geometries
                    if line.is_empty:
                        continue
                    
                    # Get line length
                    line_length = line.length
                    
                    # Convert from degrees to approximate meters (very rough estimate)
                    # More precise calculation would use a proper projection
                    meters_per_degree = 111000  # ~111km per degree at the equator
                    line_length_meters = line_length * meters_per_degree
                    
                    # Calculate number of segments
                    num_segments = max(1, int(line_length_meters / interval))
                    
                    # Sample evenly along the line
                    for i in range(num_segments + 1):
                        # Calculate segment fraction
                        segment_fraction = i / num_segments if num_segments > 0 else 0
                        
                        # Get point at fraction
                        point = line.interpolate(segment_fraction, normalized=True)
                        
                        # Get coordinates and add to sample points
                        # Convert to (lat, lon) for Street View API
                        sample_points.append((point.y, point.x))
            
            logger.info(f"Generated {len(sample_points)} sample points")
            
            # Deduplicate points (within a small tolerance)
            deduplicated_points = self._deduplicate_points(sample_points)
            logger.info(f"After deduplication: {len(deduplicated_points)} sample points")
            
            return deduplicated_points
        
        except Exception as e:
            logger.error(f"Error sampling points along roads: {str(e)}")
            return []
    
    def _deduplicate_points(self, points: List[Tuple[float, float]], tolerance: float = 1e-5) -> List[Tuple[float, float]]:
        """Deduplicate very close points.
        
        Args:
            points: List of (latitude, longitude) tuples
            tolerance: Tolerance for considering points as duplicates
            
        Returns:
            Deduplicated list of points
        """
        if not points:
            return []
        
        # Convert to numpy array for vectorized operations
        points_array = np.array(points)
        
        # Track points to keep
        keep = np.ones(len(points_array), dtype=bool)
        
        for i, point in enumerate(points_array):
            # Skip if already marked for removal
            if not keep[i]:
                continue
            
            # Calculate distances to all other points
            distances = np.sqrt(np.sum((points_array - point)**2, axis=1))
            
            # Mark points within tolerance (except the point itself) for removal
            close_points = (distances < tolerance) & (distances > 0)
            keep[close_points] = False
        
        # Return kept points
        return [tuple(p) for p in points_array[keep]]
    
    def find_roads_in_bounds(self, bounds: Dict[str, float]) -> gpd.GeoDataFrame:
        """Find all roads within specified bounds.
        
        Args:
            bounds: Dictionary with north, south, east, west bounds
            
        Returns:
            GeoDataFrame containing roads within bounds
        """
        if self.roads_gdf is None:
            logger.error("No road data loaded. Load road data first.")
            return gpd.GeoDataFrame()
        
        try:
            # Create bounds tuple for spatial query
            bounds_tuple = (bounds['west'], bounds['south'], bounds['east'], bounds['north'])
            
            # Query spatial index for candidate roads
            candidate_indices = list(self.spatial_index.intersection(bounds_tuple))
            
            # Get roads within bounds
            roads_in_bounds = self.roads_gdf.iloc[candidate_indices]
            
            logger.info(f"Found {len(roads_in_bounds)} roads within bounds")
            
            return roads_in_bounds
            
        except Exception as e:
            logger.error(f"Error finding roads in bounds: {str(e)}")
            return gpd.GeoDataFrame()

def sample_osm_roads(osm_path: str, output_dir: str, interval: float = 50.0) -> Dict[str, Any]:
    """Sample points along all roads in an OSM file at a specified interval.
    
    Args:
        osm_path: Path to the OSM GeoPackage file
        output_dir: Directory to save output files
        interval: Distance between sample points in meters
        
    Returns:
        Dictionary with results
    """
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Load road network
        road_network = RoadNetwork(osm_path, layer="roads")
        
        # Sample points along roads
        sample_points = road_network.sample_points_along_roads(interval)
        
        # Save sample points to file
        points_path = os.path.join(output_dir, "street_view_sample_points.json")
        with open(points_path, 'w') as f:
            json.dump({
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
            }, f, indent=2)
        
        logger.info(f"Saved {len(sample_points)} sample points to {points_path}")
        
        return {
            "success": True,
            "num_points": len(sample_points),
            "points_path": points_path
        }
        
    except Exception as e:
        logger.error(f"Error sampling OSM roads: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def main():
    """Main function for testing the module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Road Network Utilities")
    parser.add_argument("--osm", required=True, help="Path to OSM GeoPackage file")
    parser.add_argument("--output", default="./road_network_output", help="Output directory")
    parser.add_argument("--interval", type=float, default=50.0, help="Sampling interval in meters")
    args = parser.parse_args()
    
    result = sample_osm_roads(args.osm, args.output, args.interval)
    
    if result.get("success", False):
        print(f"Successfully sampled {result['num_points']} points along roads")
        print(f"Sample points saved to {result['points_path']}")
        return 0
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())