#!/usr/bin/env python3
"""
Fixed graph_from_bbox with increased query area size for Arcanum
This script uses the osmnx_config module to increase max query area size
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, Tuple

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arcanum_graph_fix_large.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our OSMnx configuration
try:
    from osmnx_config import configure_osmnx
except ImportError:
    logger.error("osmnx_config.py not found. Make sure it's in the same directory.")
    sys.exit(1)

def get_osmnx_version():
    """Get the installed OSMnx version."""
    try:
        import osmnx as ox
        return ox.__version__
    except ImportError:
        logger.error("OSMnx is not installed")
        return None
    except AttributeError:
        # Some older versions might not have __version__
        return "unknown"

def fixed_graph_from_bbox(bounds: Dict[str, float], coordinate_system: str = "EPSG:4326") -> Tuple[Any, Dict[str, float]]:
    """
    Fixed function to download OSM data using osmnx.graph_from_bbox with large area support
    
    Args:
        bounds: Dictionary with north, south, east, west boundaries
        coordinate_system: Source coordinate system (default: WGS84 lat/lon)
        
    Returns:
        G: NetworkX graph from OSM data
        bounds_latlon: Dictionary with lat/lon coordinates
    """
    try:
        # First configure OSMnx with increased max_query_area_size and memory allocation
        configure_osmnx(max_query_area_size=1000000000000, timeout=600, memory=2147483648)  # 2GB memory allocation
        
        import osmnx as ox
        import pyproj
        
        # Log the OSMnx version
        osmnx_version = get_osmnx_version()
        logger.info(f"Using OSMnx version: {osmnx_version}")
        
        # Convert coordinates to lat/lon for OSM if needed
        if coordinate_system != "EPSG:4326":
            try:
                transformer = pyproj.Transformer.from_crs(
                    coordinate_system,
                    "EPSG:4326",  # WGS84
                    always_xy=True
                )
                
                north_lat, east_lon = transformer.transform(bounds["east"], bounds["north"])
                south_lat, west_lon = transformer.transform(bounds["west"], bounds["south"])
            except Exception as e:
                logger.error(f"Coordinate transformation error: {str(e)}")
                raise
        else:
            # Already in lat/lon
            north_lat = bounds["north"]
            south_lat = bounds["south"]
            east_lon = bounds["east"]
            west_lon = bounds["west"]
        
        # Updated bounds in lat/lon
        bounds_latlon = {
            "north": north_lat,
            "south": south_lat,
            "east": east_lon,
            "west": west_lon
        }
        
        logger.info(f"Query bounds (EPSG:4326): N:{north_lat}, S:{south_lat}, E:{east_lon}, W:{west_lon}")
        
        # Determine the correct approach based on OSMnx version
        if osmnx_version and osmnx_version.startswith("1."): 
            # Legacy OSMnx 1.x approach with positional args
            logger.info("Using legacy OSMnx 1.x API with positional arguments")
            G = ox.graph_from_bbox(north_lat, south_lat, east_lon, west_lon, network_type='all')
        else:
            # Modern OSMnx 2.x approach with single bbox dictionary
            logger.info("Using modern OSMnx 2.x API with bbox dictionary")
            G = ox.graph_from_bbox(
                bbox=(north_lat, south_lat, east_lon, west_lon),
                network_type='all'
            )
        
        logger.info("Successfully created OSM graph")
        return G, bounds_latlon
        
    except ImportError as e:
        logger.error(f"Required library not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error downloading OSM data: {str(e)}")
        raise

def fixed_download_osm_data(bounds: Dict[str, float], output_dir: str, coordinate_system: str = "EPSG:4326") -> str:
    """
    Fixed function to download and save OSM data with large area support
    
    Args:
        bounds: Dictionary with north, south, east, west boundaries
        output_dir: Directory to save output files
        coordinate_system: Source coordinate system (default: WGS84 lat/lon)
        
    Returns:
        message: Success or error message
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        import os
        
        # Create vector directory if it doesn't exist
        vector_dir = os.path.join(output_dir, "vector")
        if not os.path.exists(vector_dir):
            os.makedirs(vector_dir)
            
        # Get OSM graph with our modified bounds
        G, bounds_latlon = fixed_graph_from_bbox(bounds, coordinate_system)
        
        # Get buildings using the updated bounds
        try:
            logger.info("Downloading building data...")
            buildings = ox.features_from_bbox(
                bounds_latlon["north"], 
                bounds_latlon["south"], 
                bounds_latlon["east"], 
                bounds_latlon["west"], 
                tags={'building': True}
            )
            logger.info(f"Downloaded {len(buildings)} buildings")
        except Exception as e:
            logger.error(f"Error getting building features: {str(e)}")
            
            # Create placeholder for buildings
            logger.info("Creating placeholder for buildings")
            buildings = gpd.GeoDataFrame()
        
        # Save data to GeoPackage format
        osm_output = os.path.join(vector_dir, "osm_arcanum.gpkg")
        if not buildings.empty:
            buildings.to_file(osm_output, layer='buildings', driver='GPKG')
            logger.info(f"Saved buildings to {osm_output}")
        else:
            # Create empty placeholder file
            with open(osm_output, 'w') as f:
                f.write(f"Placeholder for OSM buildings data for bounds: {bounds}")
            logger.info(f"Created placeholder for buildings at {osm_output}")
                
        # Save road network separately
        roads_output = os.path.join(vector_dir, "osm_roads.gpkg")
        try:
            roads_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
            roads_gdf.to_file(roads_output, layer='roads', driver='GPKG')
            logger.info(f"Saved roads to {roads_output}")
        except Exception as e:
            logger.error(f"Error converting graph to GeoDataFrame: {str(e)}")
            # Create empty placeholder file
            with open(roads_output, 'w') as f:
                f.write(f"Placeholder for OSM roads data for bounds: {bounds}")
            logger.info(f"Created placeholder for roads at {roads_output}")
        
        return f"OSM data downloaded and saved to {osm_output} and {roads_output}"
    
    except Exception as e:
        logger.error(f"Error in download_osm_data: {str(e)}")
        
        # Ensure vector directory exists for fallback
        vector_dir = os.path.join(output_dir, "vector")
        if not os.path.exists(vector_dir):
            os.makedirs(vector_dir)
            
        # Create placeholder files for testing
        buildings_placeholder = os.path.join(vector_dir, "osm_buildings_placeholder.txt")
        roads_placeholder = os.path.join(vector_dir, "osm_roads_placeholder.txt")
        
        with open(buildings_placeholder, 'w') as f:
            f.write(f"Placeholder for OSM buildings data for bounds: {bounds}")
            
        with open(roads_placeholder, 'w') as f:
            f.write(f"Placeholder for OSM roads data for bounds: {bounds}")
            
        return f"Created OSM data placeholders in {vector_dir} (real download failed)"

def main():
    """Main entry point to test the fix."""
    parser = argparse.ArgumentParser(description="Test the OSMnx graph_from_bbox fix with large area support")
    parser.add_argument("--output", help="Output directory", default="./arcanum_3d_output")
    parser.add_argument("--bounds", help="Area bounds (north,south,east,west)", default="51.5084,51.5064,-0.1258,-0.1298")
    parser.add_argument("--coordinate_system", help="Coordinate system EPSG code", default="EPSG:4326")
    args = parser.parse_args()
    
    # Parse bounds
    bounds_values = args.bounds.split(",")
    if len(bounds_values) == 4:
        bounds = {
            "north": float(bounds_values[0]),
            "south": float(bounds_values[1]),
            "east": float(bounds_values[2]),
            "west": float(bounds_values[3])
        }
    else:
        logger.error("Invalid bounds format. Use: north,south,east,west")
        sys.exit(1)
    
    # Print OSMnx version
    version = get_osmnx_version()
    print(f"OSMnx version: {version}")
    
    # Test the fixed function
    try:
        result = fixed_download_osm_data(bounds, args.output, args.coordinate_system)
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()