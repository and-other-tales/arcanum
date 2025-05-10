#!/usr/bin/env python3
"""
Arcanum OSM Module
-----------------
This module provides OpenStreetMap data retrieval and processing functionality for Arcanum.
It includes methods for downloading map data, buildings, and road networks efficiently.

The module supports multiple approaches for downloading OSM data:
1. Geofabrik pre-built extracts (recommended)
2. Grid-based downloader for custom areas
3. Bounding box downloader for simple areas
"""

from .config import configure_osmnx, get_osmnx_version
from .grid_downloader import download_osm_grid, merge_grid_data, get_grid_cells
from .bbox_downloader import download_osm_data, download_buildings_and_roads
from .geofabrik_downloader import download_osm_data as download_geofabrik_data, download_for_bbox as geofabrik_bbox_download

__all__ = [
    'configure_osmnx',
    'get_osmnx_version',
    'download_osm_grid',
    'merge_grid_data',
    'get_grid_cells',
    'download_osm_data',
    'download_buildings_and_roads',
    'download_geofabrik_data',
    'geofabrik_bbox_download'
]