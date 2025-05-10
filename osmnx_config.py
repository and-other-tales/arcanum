#!/usr/bin/env python3
"""
OSMnx Configuration Script for Arcanum
-------------------------------------
This script configures OSMnx settings before running the main generator,
particularly increasing the max_query_area_size to accommodate larger areas.
"""

import os
import sys
import logging
import argparse

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arcanum_osmnx_config.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def configure_osmnx(max_query_area_size=None, timeout=None, memory=None):
    """
    Configure OSMnx settings for Arcanum generator
    
    Args:
        max_query_area_size: Maximum area for any query in square meters (default: 100000000000)
        timeout: Timeout for HTTP requests and Overpass queries in seconds (default: 300)
        memory: Memory allocation for Overpass query in bytes (default: None - server chooses)
        
    Returns:
        dict: Current OSMnx settings
    """
    try:
        import osmnx as ox
        
        # Log original settings
        logger.info(f"Original OSMnx max_query_area_size: {ox.settings.max_query_area_size}")
        logger.info(f"Original OSMnx requests_timeout: {ox.settings.requests_timeout}")
        logger.info(f"Original OSMnx overpass_memory: {ox.settings.overpass_memory}")
        
        # Update settings
        if max_query_area_size is not None:
            ox.settings.max_query_area_size = max_query_area_size
            logger.info(f"Updated max_query_area_size to: {max_query_area_size}")
            
        if timeout is not None:
            ox.settings.requests_timeout = timeout
            logger.info(f"Updated requests_timeout to: {timeout}")
            
        if memory is not None:
            ox.settings.overpass_memory = memory
            logger.info(f"Updated overpass_memory to: {memory}")
            
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

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Configure OSMnx settings for Arcanum")
    parser.add_argument("--max-area", type=float, default=100000000000,
                      help="Maximum query area size in square meters (default: 100000000000)")
    parser.add_argument("--timeout", type=int, default=300,
                      help="Timeout for Overpass queries in seconds (default: 300)")
    parser.add_argument("--memory", type=int, default=None,
                      help="Memory allocation for Overpass query in bytes (default: None)")
    args = parser.parse_args()
    
    # Configure OSMnx with provided settings
    settings = configure_osmnx(
        max_query_area_size=args.max_area,
        timeout=args.timeout,
        memory=args.memory
    )
    
    # Print configured settings
    print("\nConfigured OSMnx Settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
    
    # Return a success message that can be used by other scripts
    return "OSMnx configuration successful"

if __name__ == "__main__":
    main()