#!/usr/bin/env python3
"""
Geofabrik OSM Data Downloader
----------------------------
This module provides functions for downloading pre-built OSM data from Geofabrik,
which offers regularly updated extracts for regions worldwide.
"""

import os
import sys
import logging
import requests
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import geopandas as gpd

# Set up logger
logger = logging.getLogger(__name__)

# Base URL for Geofabrik downloads
GEOFABRIK_BASE_URL = "https://download.geofabrik.de"

# Region mapping for common areas
REGION_URLS = {
    "london": "europe/united-kingdom/england/greater-london-latest.osm.pbf",
    "england": "europe/united-kingdom/england-latest.osm.pbf",
    "uk": "europe/united-kingdom-latest.osm.pbf",
    "germany": "europe/germany-latest.osm.pbf",
    "france": "europe/france-latest.osm.pbf",
    "usa": "north-america/us-latest.osm.pbf",
    "california": "north-america/us/california-latest.osm.pbf",
    "new-york": "north-america/us/new-york-latest.osm.pbf",
}

def get_download_url(region: str) -> Optional[str]:
    """
    Get the Geofabrik download URL for a specific region.
    
    Args:
        region: Region name (case-insensitive)
        
    Returns:
        Full URL to download the OSM PBF file, or None if region not found
    """
    region_lower = region.lower()
    
    if region_lower in REGION_URLS:
        return f"{GEOFABRIK_BASE_URL}/{REGION_URLS[region_lower]}"
    
    logger.warning(f"Region '{region}' not found in predefined regions. "
                  f"Available regions: {', '.join(REGION_URLS.keys())}")
    return None

def download_osm_data(region: str, output_dir: str) -> Dict[str, Any]:
    """
    Download OSM data for a region from Geofabrik.
    
    Args:
        region: Region name (case-insensitive)
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with status and paths to generated files
    """
    try:
        # Get download URL for the region
        download_url = get_download_url(region)
        if not download_url:
            return {
                "success": False,
                "error": f"Region '{region}' not found. Available regions: {', '.join(REGION_URLS.keys())}"
            }
        
        # Create vector directory if it doesn't exist
        vector_dir = os.path.join(output_dir, "vector")
        os.makedirs(vector_dir, exist_ok=True)
        
        # Get the filename from the URL
        filename = os.path.basename(download_url)
        output_pbf = os.path.join(vector_dir, filename)
        
        logger.info(f"Downloading OSM data for {region} from {download_url}")
        
        # Download the file if it doesn't exist
        if not os.path.exists(output_pbf):
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                logger.info(f"File size: {total_size / (1024*1024):.1f} MB")
                
                with open(output_pbf, 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Log progress every 10%
                            if total_size > 0 and downloaded % (total_size // 10) < 8192:
                                percent = (downloaded / total_size) * 100
                                logger.info(f"Downloaded {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)")
            
            logger.info(f"Download complete: {output_pbf}")
        else:
            logger.info(f"Using existing download: {output_pbf}")
        
        # Extract data using osmium
        buildings_path, roads_path = extract_buildings_and_roads(output_pbf, vector_dir)
        
        result = {
            "success": True,
            "region": region,
            "download_url": download_url,
            "pbf_path": output_pbf,
            "buildings_path": buildings_path,
            "roads_path": roads_path
        }
        
        return result
        
    except requests.RequestException as e:
        logger.error(f"Error downloading data: {str(e)}")
        return {
            "success": False,
            "error": f"Download failed: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error in download_osm_data: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def extract_buildings_and_roads(pbf_path: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract buildings and roads from an OSM PBF file.
    
    Args:
        pbf_path: Path to the OSM PBF file
        output_dir: Directory to save the extracted data
        
    Returns:
        Tuple of (buildings_path, roads_path)
    """
    buildings_path = os.path.join(output_dir, "osm_arcanum.gpkg")
    roads_path = os.path.join(output_dir, "osm_roads.gpkg")
    
    try:
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract buildings and roads with osmium tags filter
            logger.info(f"Extracting buildings from {pbf_path}")
            
            # Check if osmium is available
            try:
                # Extract buildings using osmium
                buildings_json = os.path.join(temp_dir, "buildings.osm")
                subprocess.run([
                    "osmium", "tags-filter", pbf_path,
                    "building=*", "-o", buildings_json
                ], check=True)
                
                # Convert buildings to GeoPackage
                buildings_gdf = gpd.read_file(buildings_json)
                if not buildings_gdf.empty:
                    logger.info(f"Extracted {len(buildings_gdf)} buildings")
                    buildings_gdf.to_file(buildings_path, layer='buildings', driver='GPKG')
                    logger.info(f"Saved buildings to {buildings_path}")
                else:
                    logger.warning("No buildings found")
                    buildings_path = None
                
                # Extract roads
                logger.info(f"Extracting roads from {pbf_path}")
                roads_json = os.path.join(temp_dir, "roads.osm")
                subprocess.run([
                    "osmium", "tags-filter", pbf_path,
                    "highway=*", "-o", roads_json
                ], check=True)
                
                # Convert roads to GeoPackage
                roads_gdf = gpd.read_file(roads_json)
                if not roads_gdf.empty:
                    logger.info(f"Extracted {len(roads_gdf)} road segments")
                    roads_gdf.to_file(roads_path, layer='roads', driver='GPKG')
                    logger.info(f"Saved roads to {roads_path}")
                else:
                    logger.warning("No roads found")
                    roads_path = None
                
            except FileNotFoundError:
                # Osmium not available, try with osmnx as fallback
                logger.warning("osmium not found, falling back to osmnx for extraction")
                import osmnx as ox
                
                # Use osmnx to extract buildings
                try:
                    logger.info("Extracting buildings using osmnx")
                    buildings = ox.features.features_from_pbf(pbf_path, tags={'building': True})
                    if not buildings.empty:
                        buildings.to_file(buildings_path, layer='buildings', driver='GPKG')
                        logger.info(f"Saved {len(buildings)} buildings to {buildings_path}")
                    else:
                        logger.warning("No buildings found")
                        buildings_path = None
                except Exception as e:
                    logger.error(f"Error extracting buildings with osmnx: {str(e)}")
                    buildings_path = None
                
                # Use osmnx to extract roads
                try:
                    logger.info("Extracting roads using osmnx")
                    G = ox.graph_from_pbf(pbf_path, network_type='all')
                    roads_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
                    if not roads_gdf.empty:
                        roads_gdf.to_file(roads_path, layer='roads', driver='GPKG')
                        logger.info(f"Saved {len(roads_gdf)} road segments to {roads_path}")
                    else:
                        logger.warning("No roads found")
                        roads_path = None
                except Exception as e:
                    logger.error(f"Error extracting roads with osmnx: {str(e)}")
                    roads_path = None
    
    except Exception as e:
        logger.error(f"Error extracting data from PBF: {str(e)}")
        return None, None
    
    return buildings_path, roads_path

def download_for_bbox(bounds: Dict[str, float], output_dir: str, region: str = "london") -> Dict[str, Any]:
    """
    Alternative to the original bbox downloader that uses pre-built Geofabrik data.
    
    Args:
        bounds: Dictionary with north, south, east, west coordinates (ignored, but kept for API compatibility)
        output_dir: Directory to save output files
        region: Region name to download (default: london)
        
    Returns:
        Dictionary with status and paths to generated files
    """
    logger.info(f"Using Geofabrik pre-built data for region: {region}")
    logger.info(f"Note: Requested bounds (N:{bounds['north']}, S:{bounds['south']}, E:{bounds['east']}, W:{bounds['west']}) are ignored")
    
    return download_osm_data(region, output_dir)