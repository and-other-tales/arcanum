#!/usr/bin/env python3
"""
OSM Bounding Box Downloader
--------------------------
This module provides functions for downloading OSM data for a specified
bounding box area, handling API compatibility issues.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple

# Set up logger with consistent format
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".arcanum", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "osm.log")

# Add file handler to logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

def download_osm_data(bounds: Dict[str, float], output_dir: str, coordinate_system: str = "EPSG:4326") -> Dict[str, Any]:
    """
    Download OpenStreetMap data for the specified area.
    
    Args:
        bounds: Dictionary with north, south, east, west boundaries
        output_dir: Directory to save output files
        coordinate_system: Coordinate system of input bounds (default: WGS84)
        
    Returns:
        Dictionary with status and paths to generated files
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        import pyproj
        
        from .config import configure_osmnx, get_osmnx_version
        
        # Configure OSMnx with increased limits
        configure_osmnx()
        
        # Convert coordinates to lat/lon for OSM if needed
        if coordinate_system != "EPSG:4326":
            try:
                transformer = pyproj.Transformer.from_crs(
                    coordinate_system,
                    "EPSG:4326",  # WGS84
                    always_xy=True
                )

                north, east = transformer.transform(bounds["east"], bounds["north"])
                south, west = transformer.transform(bounds["west"], bounds["south"])
            except Exception as e:
                logger.error(f"Coordinate transformation error: {str(e)}")
                raise
        else:
            # Already in lat/lon
            north = bounds["north"]
            south = bounds["south"]
            east = bounds["east"]
            west = bounds["west"]
                
        logger.info(f"Query bounds (EPSG:4326): N:{north}, S:{south}, E:{east}, W:{west}")

        # Create vector directory if it doesn't exist
        vector_dir = os.path.join(output_dir, "vector")
        if not os.path.exists(vector_dir):
            os.makedirs(vector_dir)
        
        result = {
            "success": False,
            "bounds": bounds,
            "buildings_path": None,
            "roads_path": None,
            "error": None
        }

        # Get buildings and roads
        try:
            buildings, roads_gdf = download_buildings_and_roads(north, south, east, west)
            
            # Save buildings to GeoPackage format
            if not buildings.empty:
                osm_output = os.path.join(vector_dir, "osm_arcanum.gpkg")
                buildings.to_file(osm_output, layer='buildings', driver='GPKG')
                logger.info(f"Saved buildings to {osm_output}")
                result["buildings_path"] = osm_output
            
            # Save road network separately
            if not roads_gdf.empty:
                roads_output = os.path.join(vector_dir, "osm_roads.gpkg")
                roads_gdf.to_file(roads_output, layer='roads', driver='GPKG')
                logger.info(f"Saved roads to {roads_output}")
                result["roads_path"] = roads_output
            
            result["success"] = True
            result["message"] = "OSM data downloaded successfully"
        except Exception as osm_error:
            logger.warning(f"OSM download error: {str(osm_error)}. Creating placeholders...")
            result["error"] = str(osm_error)

            # Create placeholder files for testing
            buildings_placeholder = os.path.join(vector_dir, "osm_buildings_placeholder.txt")
            roads_placeholder = os.path.join(vector_dir, "osm_roads_placeholder.txt")

            with open(buildings_placeholder, 'w') as f:
                f.write(f"Placeholder for OSM buildings data for bounds: {bounds}")

            with open(roads_placeholder, 'w') as f:
                f.write(f"Placeholder for OSM roads data for bounds: {bounds}")

            result["message"] = f"Created OSM data placeholders in {vector_dir} (real download failed)"
            result["buildings_path"] = buildings_placeholder
            result["roads_path"] = roads_placeholder
            
        return result
        
    except Exception as e:
        logger.error(f"Error in download_osm_data: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to download OSM data: {str(e)}"
        }

def download_buildings_and_roads(north, south, east, west):
    """
    Download buildings and road network for the specified bounding box.
    
    This function handles API differences between OSMnx versions.
    
    Args:
        north, south, east, west: Boundary coordinates in WGS84 (lat/lon)
        
    Returns:
        tuple: (buildings GeoDataFrame, roads GeoDataFrame)
    """
    import osmnx as ox
    import geopandas as gpd
    
    from .config import get_osmnx_version
    
    # Get OSMnx version to determine the correct API to use
    osmnx_version = get_osmnx_version()
    logger.info(f"Using OSMnx version: {osmnx_version}")
    
    # Download road network
    if osmnx_version and osmnx_version.startswith("1."): 
        # Legacy OSMnx 1.x approach with positional arguments
        logger.info("Using legacy OSMnx 1.x API with positional arguments")
        G = ox.graph_from_bbox(north, south, east, west, network_type='all')
    else:
        # Modern OSMnx 2.x approach with single bbox parameter
        logger.info("Using modern OSMnx 2.x API with bbox dictionary")
        G = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type='all'
        )

    logger.info("Successfully downloaded road network graph")
    
    # Download buildings
    try:
        # Handle API differences between OSMnx versions
        try:
            # Modern OSMnx 2.x approach
            buildings = ox.features_from_bbox(
                bbox=(north, south, east, west),
                tags={'building': True}
            )
        except TypeError:
            # Legacy OSMnx 1.x approach
            buildings = ox.features_from_bbox(
                north, south, east, west,
                tags={'building': True}
            )
        
        logger.info(f"Downloaded {len(buildings)} buildings")
    except Exception as e:
        logger.warning(f"Error downloading buildings: {str(e)}")
        buildings = gpd.GeoDataFrame()
    
    # Convert road network to GeoDataFrame
    try:
        roads_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
        logger.info(f"Converted road network to GeoDataFrame with {len(roads_gdf)} edges")
    except Exception as e:
        logger.warning(f"Error converting roads to GeoDataFrame: {str(e)}")
        roads_gdf = gpd.GeoDataFrame()
    
    return buildings, roads_gdf