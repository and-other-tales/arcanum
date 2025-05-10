#!/usr/bin/env python3
"""
Spatial Bounds Utilities
-----------------------
This module provides utilities for working with spatial bounds and ensuring complete 
coverage for geo-related operations like 3D tiles downloads and Street View imagery.
"""

import os
import sys
import logging
import json
import math
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
from shapely.geometry import Polygon, Point, LineString, box, mapping, shape
try:
    # For older shapely versions
    from shapely.ops import cascaded_union, transform
except ImportError:
    # For newer shapely versions
    from shapely.ops import transform
    from functools import reduce
    from shapely.ops import unary_union

    def cascaded_union(polygons):
        """Replacement for cascaded_union using unary_union."""
        return unary_union(polygons)
import pyproj
from functools import partial

# Setup logging
logger = logging.getLogger(__name__)

# Constants
EARTH_RADIUS_METERS = 6371000

def bounds_to_polygon(bounds: Dict[str, float]) -> Polygon:
    """Convert a bounds dictionary to a shapely Polygon.
    
    Args:
        bounds: Dictionary with north, south, east, west bounds
        
    Returns:
        Shapely Polygon representing the bounds
    """
    return box(bounds['west'], bounds['south'], bounds['east'], bounds['north'])

def polygon_to_bounds(polygon: Polygon) -> Dict[str, float]:
    """Convert a shapely Polygon to a bounds dictionary.
    
    Args:
        polygon: Shapely Polygon
        
    Returns:
        Dictionary with north, south, east, west bounds
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    return {
        'west': min_x,
        'south': min_y,
        'east': max_x,
        'north': max_y
    }

def city_polygon(name: str) -> Optional[Polygon]:
    """Get a polygon for a named city from a predefined list or by querying an external service.
    
    Args:
        name: City name (e.g., "London", "New York")
        
    Returns:
        Shapely Polygon representing the city boundaries or None if not found
    """
    # Common city boundaries (approximate)
    city_bounds = {
        'london': {
            'north': 51.691874,
            'south': 51.286503,
            'east': 0.335352,
            'west': -0.510375
        },
        'new york': {
            'north': 40.917577,
            'south': 40.477399,
            'east': -73.700272,
            'west': -74.259090
        },
        'san francisco': {
            'north': 37.929824,
            'south': 37.639830,
            'east': -122.281780,
            'west': -122.612086
        },
        'berlin': {
            'north': 52.675499,
            'south': 52.338261,
            'east': 13.760529,
            'west': 13.088359
        },
        'paris': {
            'north': 48.901280,
            'south': 48.815573,
            'east': 2.469920,
            'west': 2.224199
        },
        'tokyo': {
            'north': 35.817813,
            'south': 35.530369,
            'east': 139.910202,
            'west': 139.597927
        }
    }
    
    # Normalize city name
    name_lower = name.lower()
    
    # Check if we have predefined bounds
    if name_lower in city_bounds:
        return bounds_to_polygon(city_bounds[name_lower])
    
    # TODO: Add support for querying external services like OpenStreetMap Nominatim API
    logger.warning(f"City boundary for '{name}' not found in predefined list")
    return None

def lat_lon_to_meters(lat: float, lon: float) -> Tuple[float, float]:
    """Convert latitude/longitude to Web Mercator coordinates in meters.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        Tuple of (x, y) in meters
    """
    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Get x in meters
    x = EARTH_RADIUS_METERS * lon_rad
    
    # Get y in meters using Mercator projection
    y = EARTH_RADIUS_METERS * math.log(math.tan(math.pi/4 + lat_rad/2))
    
    return (x, y)

def meters_to_lat_lon(x: float, y: float) -> Tuple[float, float]:
    """Convert Web Mercator coordinates in meters to latitude/longitude.
    
    Args:
        x: X coordinate in meters
        y: Y coordinate in meters
        
    Returns:
        Tuple of (lat, lon) in degrees
    """
    # Convert to longitude
    lon = math.degrees(x / EARTH_RADIUS_METERS)
    
    # Convert to latitude
    lat = math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_METERS)) - math.pi/2)
    
    return (lat, lon)

def create_grid_from_bounds(bounds: Dict[str, float], cell_size_meters: float) -> List[Dict[str, float]]:
    """Create a grid of smaller bounds within the provided bounds.
    
    Args:
        bounds: Dictionary with north, south, east, west bounds
        cell_size_meters: Approximate size of each grid cell in meters
        
    Returns:
        List of bounds dictionaries representing the grid cells
    """
    # Convert bounds to polygon
    bounds_poly = bounds_to_polygon(bounds)
    
    # Determine bounds dimensions in meters
    south_west = lat_lon_to_meters(bounds['south'], bounds['west'])
    north_east = lat_lon_to_meters(bounds['north'], bounds['east'])
    
    # Calculate dimensions
    width_meters = north_east[0] - south_west[0]
    height_meters = north_east[1] - south_west[1]
    
    # Calculate number of grid cells
    cols = max(1, int(width_meters / cell_size_meters))
    rows = max(1, int(height_meters / cell_size_meters))
    
    # Adjust cell size to fit the bounds exactly
    cell_width_meters = width_meters / cols
    cell_height_meters = height_meters / rows
    
    # Create grid cells
    grid_cells = []
    
    for row in range(rows):
        for col in range(cols):
            # Calculate cell coordinates in meters
            cell_sw_x = south_west[0] + col * cell_width_meters
            cell_sw_y = south_west[1] + row * cell_height_meters
            cell_ne_x = cell_sw_x + cell_width_meters
            cell_ne_y = cell_sw_y + cell_height_meters
            
            # Convert back to lat/lon
            cell_sw_lat, cell_sw_lon = meters_to_lat_lon(cell_sw_x, cell_sw_y)
            cell_ne_lat, cell_ne_lon = meters_to_lat_lon(cell_ne_x, cell_ne_y)
            
            # Create bounds for the cell
            cell_bounds = {
                'south': cell_sw_lat,
                'west': cell_sw_lon,
                'north': cell_ne_lat,
                'east': cell_ne_lon
            }
            
            grid_cells.append(cell_bounds)
    
    return grid_cells

def bounds_contains_tile(bounds: Dict[str, float], tile_bounds: Dict[str, float]) -> bool:
    """Check if bounds contains a tile.
    
    Args:
        bounds: Dictionary with north, south, east, west bounds
        tile_bounds: Dictionary with north, south, east, west bounds of the tile
        
    Returns:
        True if the bounds contains the tile
    """
    bounds_poly = bounds_to_polygon(bounds)
    tile_poly = bounds_to_polygon(tile_bounds)
    
    return bounds_poly.contains(tile_poly) or bounds_poly.intersects(tile_poly)

def parse_3d_tile_bounds(tile_data: Dict) -> Optional[Dict[str, float]]:
    """Parse the geographic bounds of a 3D tile from its data.
    
    Args:
        tile_data: Dictionary containing 3D tile data
        
    Returns:
        Dictionary with north, south, east, west bounds or None if not found
    """
    # Check if the tile has a region
    if "boundingVolume" in tile_data:
        bounding_volume = tile_data["boundingVolume"]
        
        # Check for a region (geographic) bounding volume
        if "region" in bounding_volume:
            region = bounding_volume["region"]
            if len(region) >= 4:
                # Convert from radians to degrees
                west = math.degrees(region[0])
                south = math.degrees(region[1])
                east = math.degrees(region[2])
                north = math.degrees(region[3])
                
                return {
                    "west": west,
                    "south": south,
                    "east": east,
                    "north": north
                }
        
        # Check for a box bounding volume (requires transformation)
        elif "box" in bounding_volume:
            # This is more complex and requires specific tile metadata
            # Simplified approximation
            logger.warning("Box bounding volumes not fully supported yet for spatial filtering")
            return None
    
    # If we don't find geographic bounds, log a warning
    logger.warning("Could not determine geographic bounds for tile")
    return None

def transform_polygon_to_3857(polygon: Polygon) -> Polygon:
    """Transform a polygon from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857).
    
    Args:
        polygon: Shapely Polygon in WGS84
        
    Returns:
        Shapely Polygon in Web Mercator projection
    """
    project = partial(
        pyproj.transform,
        pyproj.Proj('EPSG:4326'),  # source coordinate system (WGS84)
        pyproj.Proj('EPSG:3857')   # destination coordinate system (Web Mercator)
    )
    
    return transform(project, polygon)

def transform_polygon_to_4326(polygon: Polygon) -> Polygon:
    """Transform a polygon from Web Mercator (EPSG:3857) to WGS84 (EPSG:4326).
    
    Args:
        polygon: Shapely Polygon in Web Mercator
        
    Returns:
        Shapely Polygon in WGS84 projection
    """
    project = partial(
        pyproj.transform,
        pyproj.Proj('EPSG:3857'),  # source coordinate system (Web Mercator)
        pyproj.Proj('EPSG:4326')   # destination coordinate system (WGS84)
    )
    
    return transform(project, polygon)

def buffer_polygon_meters(polygon: Polygon, buffer_meters: float) -> Polygon:
    """Buffer a WGS84 polygon by a specified number of meters.
    
    Args:
        polygon: Shapely Polygon in WGS84
        buffer_meters: Buffer distance in meters
        
    Returns:
        Buffered shapely Polygon in WGS84
    """
    # Transform to Web Mercator where distances are in meters
    polygon_3857 = transform_polygon_to_3857(polygon)
    
    # Buffer in meters
    buffered_3857 = polygon_3857.buffer(buffer_meters)
    
    # Transform back to WGS84
    return transform_polygon_to_4326(buffered_3857)

def calculate_bounds_area_km2(bounds: Dict[str, float]) -> float:
    """Calculate the approximate area of bounds in square kilometers.
    
    Args:
        bounds: Dictionary with north, south, east, west bounds
        
    Returns:
        Area in square kilometers
    """
    # Create polygon from bounds
    bounds_poly = bounds_to_polygon(bounds)
    
    # Transform to Web Mercator for accurate area calculation
    bounds_3857 = transform_polygon_to_3857(bounds_poly)
    
    # Calculate area in square meters and convert to square kilometers
    area_m2 = bounds_3857.area
    area_km2 = area_m2 / 1_000_000
    
    return area_km2

def remove_water_bodies(bounds_poly: Polygon) -> Polygon:
    """Remove major water bodies from a polygon (simplified).
    
    Args:
        bounds_poly: Shapely Polygon representing land area
        
    Returns:
        Shapely Polygon with water bodies removed
    """
    # This is a placeholder for a more sophisticated implementation
    # A full implementation would use data from OpenStreetMap or other sources
    # to identify and remove water bodies like oceans, lakes, and rivers
    
    logger.warning("Water body removal not fully implemented. Using original polygon.")
    return bounds_poly

def main():
    """Main function for testing the module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Spatial Bounds Utilities")
    parser.add_argument("--city", help="City name to get bounds for")
    parser.add_argument("--grid", type=float, help="Create grid with specified cell size in meters")
    parser.add_argument("--area", action="store_true", help="Calculate area of bounds in square kilometers")
    parser.add_argument("--buffer", type=float, help="Buffer distance in meters")
    args = parser.parse_args()
    
    if args.city:
        city_poly = city_polygon(args.city)
        if city_poly:
            bounds = polygon_to_bounds(city_poly)
            print(f"Bounds for {args.city}: {json.dumps(bounds, indent=2)}")
            
            if args.area:
                area_km2 = calculate_bounds_area_km2(bounds)
                print(f"Area: {area_km2:.2f} km²")
            
            if args.buffer:
                buffered_poly = buffer_polygon_meters(city_poly, args.buffer)
                buffered_bounds = polygon_to_bounds(buffered_poly)
                print(f"Buffered bounds (+{args.buffer}m): {json.dumps(buffered_bounds, indent=2)}")
            
            if args.grid:
                grid_cells = create_grid_from_bounds(bounds, args.grid)
                print(f"Created grid with {len(grid_cells)} cells of approximately {args.grid}m size")
                print(f"First cell: {json.dumps(grid_cells[0], indent=2)}")
                print(f"Last cell: {json.dumps(grid_cells[-1], indent=2)}")
        else:
            print(f"City '{args.city}' not found")
    
if __name__ == "__main__":
    main()