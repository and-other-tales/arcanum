#!/usr/bin/env python3
"""
OSMnx Configuration for Arcanum
------------------------------
This module provides configuration helpers for OSMnx to optimize
large-area downloads and manage settings.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

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

def configure_osmnx(max_query_area_size=None, timeout=None, memory=None, overpass_endpoint=None):
    """
    Configure OSMnx settings for Arcanum generator

    Args:
        max_query_area_size: Maximum area for any query in square meters (default: 100000000000)
        timeout: Timeout for HTTP requests and Overpass queries in seconds (default: 300)
        memory: Memory allocation for Overpass query in bytes (default: None - server chooses)
        overpass_endpoint: URL for the Overpass API (default: https://overpass-api.de/api)

    Returns:
        dict: Current OSMnx settings
    """
    try:
        import osmnx as ox

        # Set defaults if not provided
        if max_query_area_size is None:
            max_query_area_size = 100000000000  # 100 billion sq meters

        if timeout is None:
            timeout = 300  # 5 minutes

        if overpass_endpoint is None:
            overpass_endpoint = "https://overpass-api.de/api/interpreter"

        # Log original settings
        logger.info(f"Original OSMnx max_query_area_size: {ox.settings.max_query_area_size}")
        logger.info(f"Original OSMnx requests_timeout: {ox.settings.requests_timeout}")
        logger.info(f"Original OSMnx overpass_memory: {ox.settings.overpass_memory}")
        logger.info(f"Original OSMnx overpass_url: {ox.settings.overpass_url}")

        # Update settings
        ox.settings.max_query_area_size = max_query_area_size
        logger.info(f"Updated max_query_area_size to: {max_query_area_size}")

        ox.settings.requests_timeout = timeout
        logger.info(f"Updated requests_timeout to: {timeout}")

        if memory is not None:
            ox.settings.overpass_memory = memory
            logger.info(f"Updated overpass_memory to: {memory}")

        # Set the Overpass API endpoint
        ox.settings.overpass_url = overpass_endpoint
        logger.info(f"Using Overpass API endpoint: {ox.settings.overpass_url}")

        # Return current settings
        current_settings = {
            "max_query_area_size": ox.settings.max_query_area_size,
            "requests_timeout": ox.settings.requests_timeout,
            "overpass_memory": ox.settings.overpass_memory,
            "overpass_url": ox.settings.overpass_url,
            "log_level": ox.settings.log_level
        }
        
        return current_settings
    
    except ImportError:
        logger.error("OSMnx is not installed. Please install it with: pip install osmnx")
        return {"error": "OSMnx not installed"}
    except Exception as e:
        logger.error(f"Error configuring OSMnx: {str(e)}")
        return {"error": str(e)}

def configure_for_grid_downloads(cell_size_meters=500):
    """
    Configure OSMnx for efficient grid-based downloads

    Args:
        cell_size_meters: Size of grid cells in meters

    Returns:
        dict: Updated settings
    """
    # For grid-based downloads, use a reasonable max area size
    # that's large enough for the cell but not too large to cause issues
    # Using 25 sq km (25,000,000 sq meters) as a reasonable limit
    max_area = 25000000  # 25 sq km

    # Standard timeout for each cell
    timeout = 180  # 3 minutes per cell should be enough

    # 1GB memory allocation
    memory = 1073741824

    # Make sure we're using the correct Overpass API endpoint
    overpass_endpoint = "https://overpass-api.de/api/interpreter"

    return configure_osmnx(
        max_query_area_size=max_area,
        timeout=timeout,
        memory=memory,
        overpass_endpoint=overpass_endpoint
    )