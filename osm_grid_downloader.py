#!/usr/bin/env python3
"""
OSM Grid Downloader for Arcanum
------------------------------
This script divides the requested area into smaller grid cells
and downloads OSM data for each cell individually to avoid Overpass API limitations.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import time
from typing import Dict, List, Any, Tuple
import math

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arcanum_osm_grid.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our OSMnx configuration utilities if available
try:
    from osmnx_config import configure_osmnx
    OSMNX_CONFIG_AVAILABLE = True
except ImportError:
    OSMNX_CONFIG_AVAILABLE = False
    logger.warning("osmnx_config.py not found. Using default OSMnx settings.")

def configure_osmnx_optimally():
    """Configure OSMnx with optimal settings for grid-based downloads."""
    if OSMNX_CONFIG_AVAILABLE:
        logger.info("Configuring OSMnx with optimal settings for grid-based downloads")
        configure_osmnx(
            max_query_area_size=2500000,  # 2.5 sq km (500m x 500m plus buffer)
            timeout=180,                  # 3 minutes per cell
            memory=1073741824            # 1GB memory allocation
        )
        return True
    return False

def lat_lon_to_meters(lat1, lon1, lat2, lon2):
    """
    Calculate distance in meters between two lat/lon points.
    Uses the Haversine formula.
    """
    # Earth radius in meters
    R = 6371000
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def get_grid_cells(bounds, cell_size_meters=500):
    """
    Divide a bounding box into grid cells of specified size in meters.
    
    Args:
        bounds: Dictionary with north, south, east, west in lat/lon (EPSG:4326)
        cell_size_meters: Size of each grid cell in meters (default 500m)
        
    Returns:
        List of grid cell bounds dictionaries
    """
    north, south, east, west = bounds["north"], bounds["south"], bounds["east"], bounds["west"]
    
    # Calculate approximate distance in meters
    width_meters = lat_lon_to_meters(north, west, north, east)  # East-West distance
    height_meters = lat_lon_to_meters(north, west, south, west)  # North-South distance
    
    logger.info(f"Area size: {width_meters:.1f}m x {height_meters:.1f}m")
    
    # Calculate degrees per meter (approximate)
    deg_lat_per_meter = (north - south) / height_meters if height_meters > 0 else 0
    deg_lon_per_meter = (east - west) / width_meters if width_meters > 0 else 0
    
    # Calculate grid dimensions
    cell_size_lat = deg_lat_per_meter * cell_size_meters
    cell_size_lon = deg_lon_per_meter * cell_size_meters
    
    # Ensure we have at least one grid cell
    num_lat_cells = max(1, int(height_meters / cell_size_meters) + 1)
    num_lon_cells = max(1, int(width_meters / cell_size_meters) + 1)
    
    # Calculate actual cell sizes to cover the entire area
    cell_size_lat = (north - south) / num_lat_cells
    cell_size_lon = (east - west) / num_lon_cells
    
    logger.info(f"Dividing area into {num_lat_cells} x {num_lon_cells} grid (total: {num_lat_cells * num_lon_cells} cells)")
    
    # Generate grid cells
    grid_cells = []
    for i in range(num_lat_cells):
        for j in range(num_lon_cells):
            cell_north = north - i * cell_size_lat
            cell_south = cell_north - cell_size_lat
            cell_west = west + j * cell_size_lon
            cell_east = cell_west + cell_size_lon
            
            cell_bounds = {
                "north": cell_north,
                "south": cell_south,
                "east": cell_east,
                "west": cell_west,
                "grid_position": (i, j)
            }
            grid_cells.append(cell_bounds)
    
    return grid_cells

def download_osm_for_cell(cell_bounds, output_dir, tags=None):
    """
    Download OSM data for a single grid cell.
    
    Args:
        cell_bounds: Dictionary with north, south, east, west coordinates
        output_dir: Base directory to save the data
        tags: Dictionary of OSM tags to download (default: roads and buildings)
        
    Returns:
        Dictionary with success status and file paths
    """
    try:
        import osmnx as ox
        import geopandas as gpd
        
        # Get grid position for naming
        i, j = cell_bounds.get("grid_position", (0, 0))
        cell_name = f"cell_{i}_{j}"
        logger.info(f"Processing {cell_name} - N:{cell_bounds['north']:.6f}, S:{cell_bounds['south']:.6f}, E:{cell_bounds['east']:.6f}, W:{cell_bounds['west']:.6f}")
        
        # Create output directory for this cell
        cell_dir = os.path.join(output_dir, f"grid/{cell_name}")
        os.makedirs(cell_dir, exist_ok=True)
        
        result = {
            "cell_name": cell_name,
            "bounds": cell_bounds,
            "success": False,
            "buildings_path": None,
            "roads_path": None
        }
        
        # Default tags if none provided
        if tags is None:
            tags = {
                "buildings": {"building": True},
                "roads": "all"  # Special case for road networks
            }
        
        # Download building data if requested
        if "buildings" in tags:
            try:
                logger.info(f"Downloading buildings for {cell_name}")
                # Handle API differences between OSMnx versions
                try:
                    # Modern OSMnx 2.x approach
                    buildings = ox.features_from_bbox(
                        bbox=(cell_bounds["north"], cell_bounds["south"], cell_bounds["east"], cell_bounds["west"]),
                        tags=tags["buildings"]
                    )
                except TypeError:
                    # Legacy OSMnx 1.x approach
                    buildings = ox.features_from_bbox(
                        cell_bounds["north"],
                        cell_bounds["south"],
                        cell_bounds["east"],
                        cell_bounds["west"],
                        tags=tags["buildings"]
                    )
                
                if not buildings.empty:
                    buildings_path = os.path.join(cell_dir, f"buildings.gpkg")
                    buildings.to_file(buildings_path, layer='buildings', driver='GPKG')
                    result["buildings_path"] = buildings_path
                    logger.info(f"Downloaded {len(buildings)} buildings for {cell_name}")
                else:
                    logger.info(f"No buildings found in {cell_name}")
            except Exception as e:
                logger.warning(f"Failed to download buildings for {cell_name}: {str(e)}")
        
        # Download road network if requested
        if "roads" in tags:
            try:
                logger.info(f"Downloading road network for {cell_name}")
                
                # Load road network
                road_types = tags["roads"]
                network_type = 'all' if road_types == "all" else road_types

                # Handle API differences between OSMnx versions
                try:
                    # Modern OSMnx 2.x approach
                    G = ox.graph_from_bbox(
                        bbox=(cell_bounds["north"], cell_bounds["south"], cell_bounds["east"], cell_bounds["west"]),
                        network_type=network_type
                    )
                except TypeError:
                    # Legacy OSMnx 1.x approach
                    G = ox.graph_from_bbox(
                        cell_bounds["north"],
                        cell_bounds["south"],
                        cell_bounds["east"],
                        cell_bounds["west"],
                        network_type=network_type
                    )
                
                # Convert to GeoDataFrame and save
                if G is not None and len(G.edges) > 0:
                    roads_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
                    roads_path = os.path.join(cell_dir, f"roads.gpkg")
                    roads_gdf.to_file(roads_path, layer='roads', driver='GPKG')
                    result["roads_path"] = roads_path
                    logger.info(f"Downloaded {len(roads_gdf)} road segments for {cell_name}")
                else:
                    logger.info(f"No roads found in {cell_name}")
            except Exception as e:
                logger.warning(f"Failed to download roads for {cell_name}: {str(e)}")
        
        # Mark as success if we got either buildings or roads
        result["success"] = result["buildings_path"] is not None or result["roads_path"] is not None
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        # Save cell metadata
        metadata_path = os.path.join(cell_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    except ImportError as e:
        logger.error(f"Required library not available: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error downloading OSM data for cell: {str(e)}")
        return {"success": False, "error": str(e)}

def download_osm_grid(bounds, output_dir, cell_size_meters=500, tags=None, max_concurrent=None):
    """
    Download OSM data for an area by dividing it into a grid.
    
    Args:
        bounds: Dictionary with north, south, east, west coordinates
        output_dir: Directory to save the output files
        cell_size_meters: Size of each grid cell in meters
        tags: Dictionary of OSM tags to download
        max_concurrent: Maximum number of concurrent cells to process (None for sequential)
        
    Returns:
        Dictionary with overall results
    """
    start_time = time.time()
    
    # Configure OSMnx for optimal grid downloads
    if OSMNX_CONFIG_AVAILABLE:
        configure_osmnx_optimally()
    
    # Make sure output directory exists
    vector_dir = os.path.join(output_dir, "vector")
    os.makedirs(vector_dir, exist_ok=True)
    
    # Divide area into grid cells
    grid_cells = get_grid_cells(bounds, cell_size_meters)
    
    # Download data for each cell
    results = {
        "bounds": bounds,
        "cell_size_meters": cell_size_meters,
        "cell_count": len(grid_cells),
        "success_count": 0,
        "fail_count": 0,
        "cell_results": []
    }
    
    # Process sequentially for now (can be extended for parallel processing)
    for i, cell in enumerate(grid_cells):
        logger.info(f"Processing cell {i+1}/{len(grid_cells)}")
        
        # Add delay between cells to avoid hitting rate limits
        if i > 0:
            time.sleep(2)  # Wait 2 seconds between cells
        
        # Download data for this cell
        cell_result = download_osm_for_cell(cell, vector_dir, tags)
        
        # Track results
        results["cell_results"].append(cell_result)
        if cell_result.get("success", False):
            results["success_count"] += 1
        else:
            results["fail_count"] += 1
    
    # Calculate stats
    results["duration_seconds"] = time.time() - start_time
    results["completed_at"] = datetime.now().isoformat()
    
    # Save overall metadata
    metadata_path = os.path.join(vector_dir, "grid_download_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Grid download completed in {results['duration_seconds']:.1f} seconds")
    logger.info(f"Successful cells: {results['success_count']}/{results['cell_count']}")
    
    return results

def merge_grid_data(output_dir):
    """
    Merge all grid cell data into combined files.
    
    Args:
        output_dir: Base directory containing the grid data
        
    Returns:
        Dictionary with paths to merged files
    """
    try:
        import geopandas as gpd
        from pathlib import Path
        
        vector_dir = os.path.join(output_dir, "vector")
        grid_dir = os.path.join(vector_dir, "grid")
        
        # Paths for merged files
        merged_buildings_path = os.path.join(vector_dir, "osm_all_buildings.gpkg")
        merged_roads_path = os.path.join(vector_dir, "osm_all_roads.gpkg")
        
        # Find all building and road files
        building_files = list(Path(grid_dir).glob("**/buildings.gpkg"))
        road_files = list(Path(grid_dir).glob("**/roads.gpkg"))
        
        logger.info(f"Found {len(building_files)} building files and {len(road_files)} road files to merge")
        
        # Merge buildings
        if building_files:
            buildings_gdfs = []
            for file in building_files:
                try:
                    gdf = gpd.read_file(file)
                    if not gdf.empty:
                        buildings_gdfs.append(gdf)
                except Exception as e:
                    logger.warning(f"Failed to read {file}: {str(e)}")
            
            if buildings_gdfs:
                merged_buildings = gpd.pd.concat(buildings_gdfs, ignore_index=True)
                merged_buildings.to_file(merged_buildings_path, layer='buildings', driver='GPKG')
                logger.info(f"Merged {len(merged_buildings)} buildings into {merged_buildings_path}")
            else:
                logger.warning("No valid building data to merge")
        
        # Merge roads
        if road_files:
            roads_gdfs = []
            for file in road_files:
                try:
                    gdf = gpd.read_file(file)
                    if not gdf.empty:
                        roads_gdfs.append(gdf)
                except Exception as e:
                    logger.warning(f"Failed to read {file}: {str(e)}")
            
            if roads_gdfs:
                merged_roads = gpd.pd.concat(roads_gdfs, ignore_index=True)
                merged_roads.to_file(merged_roads_path, layer='roads', driver='GPKG')
                logger.info(f"Merged {len(merged_roads)} road segments into {merged_roads_path}")
            else:
                logger.warning("No valid road data to merge")
        
        return {
            "success": True,
            "buildings_path": merged_buildings_path if building_files else None,
            "roads_path": merged_roads_path if road_files else None
        }
    
    except ImportError as e:
        logger.error(f"Required library not available for merging: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error merging grid data: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Download OSM data using a grid approach")
    parser.add_argument("--bounds", type=str, default="51.5124,51.5024,-0.1228,-0.1328",
                      help="Area bounds as north,south,east,west in WGS84 coordinates")
    parser.add_argument("--output", type=str, default="./arcanum_3d_output",
                      help="Output directory")
    parser.add_argument("--cell-size", type=int, default=500,
                      help="Grid cell size in meters (default: 500)")
    parser.add_argument("--merge", action="store_true",
                      help="Merge grid data after download")
    args = parser.parse_args()
    
    # Parse bounds
    bounds_values = args.bounds.split(",")
    if len(bounds_values) != 4:
        logger.error("Invalid bounds format. Use: north,south,east,west")
        sys.exit(1)
    
    bounds = {
        "north": float(bounds_values[0]),
        "south": float(bounds_values[1]),
        "east": float(bounds_values[2]),
        "west": float(bounds_values[3])
    }
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Download data using grid approach
    results = download_osm_grid(bounds, args.output, args.cell_size)
    
    # Merge the data if requested
    if args.merge and results["success_count"] > 0:
        merge_results = merge_grid_data(args.output)
        if merge_results["success"]:
            logger.info("Successfully merged grid data")
            print(f"Downloaded and merged OSM data to:\n"
                  f"  Buildings: {merge_results.get('buildings_path', 'None')}\n"
                  f"  Roads: {merge_results.get('roads_path', 'None')}")
        else:
            logger.error(f"Failed to merge grid data: {merge_results.get('error', 'Unknown error')}")
    else:
        print(f"Downloaded OSM data for {results['success_count']} grid cells to {os.path.join(args.output, 'vector/grid')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())