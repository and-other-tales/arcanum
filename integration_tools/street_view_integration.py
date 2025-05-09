#!/usr/bin/env python3
"""
Google Street View Integration Module
------------------------------------
This module provides integration with Google Maps Platform's Street View API,
allowing for fetching and processing of Street View imagery within the Arcanum system.

DEPRECATED: This module has been moved to modules/integration/street_view_integration.py
Please update your imports to use the new module path.
"""

import warnings
warnings.warn(
    "The integration_tools.street_view_integration module is deprecated. "
    "Please use modules.integration.street_view_integration instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import sys
import logging
import json
import requests
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from urllib.parse import urlencode

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import configuration
try:
    from integration_tools.config import get_cache_dir, get_api_key, get_base_url
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add module level functions for direct import
def fetch_street_view(lat: float, lng: float, output_dir: str, api_key: str = None,
                     heading: Optional[int] = None, fov: int = 90,
                     pitch: int = 0, radius: int = 100, cache_dir: str = None) -> Dict:
    """Fetch Street View imagery for a given location and save to output directory.

    This is a module-level function that creates a GoogleStreetViewIntegration instance
    and delegates to its fetch_street_view method.

    Args:
        lat: Latitude of the location
        lng: Longitude of the location
        output_dir: Directory to save downloaded imagery
        api_key: Google Maps API key with Street View access (from env var if None)
        heading: Optional specific heading in degrees (0-360, default captures panorama)
        fov: Field of view (20-120, default 90)
        pitch: Camera pitch (-90 to 90, default 0)
        radius: Maximum radius in meters to look for Street View imagery
        cache_dir: Directory to cache downloaded images

    Returns:
        Dictionary with download results including success status
    """
    # Create class instance and configure it with API key
    integration = GoogleStreetViewIntegration(
        api_key=api_key,
        cache_dir=cache_dir
    )

    # Log the request
    logger.info(f"Fetching Street View imagery for location ({lat}, {lng})")

    # Delegate to the class method
    return integration.fetch_street_view(lat, lng, output_dir, heading, fov, pitch, radius)

def fetch_street_view_along_roads(
    points: List[Tuple[float, float, int]],
    output_dir: str,
    api_key: str = None,
    panorama: bool = True,
    max_search_radius: int = 1000,
    max_workers: int = 4,
    cache_dir: str = None
) -> Dict[str, Any]:
    """Fetch Street View images along a road network at sampled points.

    Args:
        points: List of (lat, lng, index) tuples representing sample points
        output_dir: Directory to save Street View imagery
        api_key: Google Maps API key with Street View access (from env var if None)
        panorama: Whether to fetch full panoramas instead of single images
        max_search_radius: Maximum search radius in meters
        max_workers: Maximum number of worker threads
        cache_dir: Directory to cache downloaded images

    Returns:
        Dictionary with download results
    """
    # Create integration instance
    integration = GoogleStreetViewIntegration(
        api_key=api_key,
        cache_dir=cache_dir
    )

    # Log the request
    logger.info(f"Fetching Street View imagery along roads ({len(points)} points)")

    # Delegate to the class method
    return integration.fetch_street_view_along_road(
        points=points,
        output_dir=output_dir,
        panorama=panorama,
        max_search_radius=max_search_radius,
        max_workers=max_workers
    )

class GoogleStreetViewIntegration:
    """Class that provides integration with Google Maps Platform's Street View API."""

    def __init__(self,
                 api_key: str = None,
                 base_url: str = None,
                 cache_dir: str = None,
                 retries: int = 3,
                 timeout: int = 30):
        """Initialize the Google Street View integration.

        Args:
            api_key: Google Maps API key with Street View access (from env var if None)
            base_url: Base URL for the Google Street View API
            cache_dir: Directory to cache downloaded images
            retries: Number of retries for failed requests
            timeout: Timeout in seconds for requests
        """
        # Use configuration if available
        if CONFIG_AVAILABLE:
            # Load API key from config or parameter or environment
            self.api_key = api_key or get_api_key("google_maps")
            # Load base URL from config or parameter or default
            self.base_url = base_url or get_base_url("street_view") or "https://maps.googleapis.com/maps/api/streetview"
            # Set cache directory from config or parameter or default
            cache_path = cache_dir or get_cache_dir("street_view")
        else:
            # Load API key from parameter or environment
            self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
            # Use provided base URL or default
            self.base_url = base_url or "https://maps.googleapis.com/maps/api/streetview"
            # Use provided cache directory or default
            cache_path = cache_dir or os.path.expanduser("~/.cache/arcanum/street_view")

        # Check for API key
        if not self.api_key:
            logger.warning("No Google Maps API key provided. Set GOOGLE_MAPS_API_KEY environment variable or configure in integration_config.json.")

        self.retries = retries
        self.timeout = timeout

        # Setup cache directory
        self.cache_dir = Path(cache_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Google Street View cache directory: {self.cache_dir}")

        # API endpoints
        self.image_endpoint = "/image"
        self.metadata_endpoint = "/metadata"

        logger.info("Google Street View Integration initialized")
    
    def get_streetview_url(self, lat: float, lng: float, size: str = "600x300",
                         heading: Optional[int] = None, fov: int = 90,
                         pitch: int = 0, radius: int = 100) -> str:
        """Get the URL for a Street View image.

        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            size: Image size in pixels (e.g. "600x300")
            heading: Optional specific heading in degrees (0-360, default captures panorama)
            fov: Field of view (20-120, default 90)
            pitch: Camera pitch (-90 to 90, default 0)
            radius: Maximum radius in meters to look for Street View imagery

        Returns:
            URL for Street View image with API key included
        """
        # Validate parameters
        if lat < -90 or lat > 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if lng < -180 or lng > 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {lng}")

        # Validate size format
        if not re.match(r'^\d+x\d+$', size):
            logger.warning(f"Invalid size format: {size}, using default 600x300")
            size = "600x300"

        # Validate and constrain fov to allowed range
        if fov < 20 or fov > 120:
            logger.warning(f"FOV must be between 20 and 120, got {fov}. Clamping to allowed range.")
            fov = min(max(fov, 20), 120)

        # Validate and constrain pitch to allowed range
        if pitch < -90 or pitch > 90:
            logger.warning(f"Pitch must be between -90 and 90, got {pitch}. Clamping to allowed range.")
            pitch = min(max(pitch, -90), 90)

        # Validate and constrain heading to allowed range if provided
        if heading is not None:
            if heading < 0 or heading >= 360:
                logger.warning(f"Heading must be between 0 and 359, got {heading}. Normalizing to allowed range.")
                heading = heading % 360

        # Validate radius is positive
        if radius <= 0:
            logger.warning(f"Radius must be positive, got {radius}. Using default 100m.")
            radius = 100

        # Build the Street View URL for the location
        url = f"{self.base_url}{self.image_endpoint}"

        # Add parameters
        params = {
            "location": f"{lat},{lng}",
            "size": size,
            "fov": fov,
            "pitch": pitch,
            "radius": radius,
            "key": self.api_key
        }

        # Add heading if specified
        if heading is not None:
            params["heading"] = heading

        # Append parameters
        url = f"{url}?{urlencode(params)}"

        return url
    
    def get_metadata_url(self, lat: float, lng: float, radius: int = 100) -> str:
        """Get the URL for Street View metadata.

        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            radius: Maximum radius in meters to look for Street View imagery

        Returns:
            URL for Street View metadata with API key included
        """
        # Validate parameters
        if lat < -90 or lat > 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if lng < -180 or lng > 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {lng}")

        # Validate radius is positive
        if radius <= 0:
            logger.warning(f"Radius must be positive, got {radius}. Using default 100m.")
            radius = 100

        # Build the metadata URL
        url = f"{self.base_url}{self.metadata_endpoint}"

        # Add parameters
        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "key": self.api_key
        }

        # Append parameters
        url = f"{url}?{urlencode(params)}"

        return url
    
    def check_street_view_availability(self, lat: float, lng: float, radius: int = 100) -> Dict:
        """Check if Street View imagery is available for a given location.

        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            radius: Maximum radius in meters to look for Street View imagery

        Returns:
            Dictionary with availability information
        """
        url = self.get_metadata_url(lat, lng, radius)
        logger.info(f"Checking Street View availability for location ({lat}, {lng})")

        # Try to fetch with retries
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                # Parse JSON
                metadata = response.json()

                # Cache the metadata
                cache_path = self.cache_dir / f"metadata_{lat}_{lng}.json"
                with open(cache_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"Successfully fetched Street View metadata (cached at {cache_path})")
                return metadata

            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed: {str(e)}. Retrying in {wait_time}s.")
                time.sleep(wait_time)

        logger.error(f"Failed to fetch Street View metadata after {self.retries} attempts.")
        return {"status": "ZERO_RESULTS"}

    def find_nearest_street_view(self, lat: float, lng: float,
                                max_radius: int = 1000,
                                radius_step: int = 100,
                                max_attempts: int = 10) -> Dict:
        """Find the nearest available Street View imagery by expanding search radius.

        Args:
            lat: Latitude of the starting location
            lng: Longitude of the starting location
            max_radius: Maximum radius in meters to expand search
            radius_step: Step size in meters to increase radius each attempt
            max_attempts: Maximum number of attempts with increased radius

        Returns:
            Dictionary with metadata of nearest imagery found or error status
        """
        logger.info(f"Finding nearest Street View imagery for location ({lat}, {lng})")

        # Start with initial radius
        current_radius = radius_step

        # Try with increasing radius until we find imagery or hit max radius
        while current_radius <= max_radius and current_radius // radius_step <= max_attempts:
            logger.info(f"Searching with radius of {current_radius}m")

            # Check for imagery at current radius
            metadata = self.check_street_view_availability(lat, lng, current_radius)

            # If we found imagery, return it
            if metadata.get("status") == "OK":
                found_lat = metadata.get("location", {}).get("lat", lat)
                found_lng = metadata.get("location", {}).get("lng", lng)

                # Calculate distance from original point
                from math import radians, cos, sin, asin, sqrt

                def haversine(lat1, lon1, lat2, lon2):
                    """Calculate the great circle distance between two points in meters"""
                    # Convert decimal degrees to radians
                    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

                    # Haversine formula
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    r = 6371000  # Radius of earth in meters
                    return c * r

                distance = haversine(lat, lng, found_lat, found_lng)
                logger.info(f"Found Street View imagery at ({found_lat}, {found_lng}), {distance:.1f}m away from requested location")

                # Add distance information to metadata
                metadata["search_distance"] = distance
                metadata["original_location"] = {"lat": lat, "lng": lng}

                return metadata

            # Increase radius for next attempt
            current_radius += radius_step

        logger.warning(f"No Street View imagery found within {max_radius}m of location ({lat}, {lng})")
        return {
            "status": "ZERO_RESULTS",
            "error": f"No Street View imagery found within {max_radius}m",
            "original_location": {"lat": lat, "lng": lng}
        }
    
    def download_street_view_image(self, lat: float, lng: float, output_path: str = None,
                                 size: str = "640x640", heading: Optional[int] = None,
                                 fov: int = 90, pitch: int = 0, radius: int = 100) -> Optional[str]:
        """Download a Street View image for a given location.
        
        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            output_path: Path to save the image (if None, uses cache directory)
            size: Image size in pixels (e.g. "640x640")
            heading: Optional specific heading in degrees (0-360, default captures panorama)
            fov: Field of view (20-120, default 90)
            pitch: Camera pitch (-90 to 90, default 0)
            radius: Maximum radius in meters to look for Street View imagery
            
        Returns:
            Path to the downloaded image, or None if failed
        """
        url = self.get_streetview_url(lat, lng, size, heading, fov, pitch, radius)
        
        # Determine output path if not provided
        if not output_path:
            # Generate a filename from the parameters
            heading_str = f"_h{heading}" if heading is not None else ""
            filename = f"streetview_{lat}_{lng}{heading_str}_f{fov}_p{pitch}.jpg"
            output_path = self.cache_dir / filename
        else:
            output_path = Path(output_path)
            
        logger.info(f"Fetching Street View image from {url} to {output_path}")
        
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to fetch with retries
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Check if we got an image (Street View API returns 200 with error image if no imagery)
                if response.headers.get('content-type') != 'image/jpeg' and response.headers.get('content-type') != 'image/png':
                    logger.warning(f"Received non-image response: {response.headers.get('content-type')}")
                    return None
                
                # Save the image
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded Street View image to {output_path}")
                return str(output_path)
                
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed: {str(e)}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
        
        logger.error(f"Failed to fetch Street View image after {self.retries} attempts.")
        return None
    
    def fetch_street_view(self, lat: float, lng: float, output_dir: str,
                        heading: Optional[int] = None, fov: int = 90,
                        pitch: int = 0, radius: int = 100,
                        max_search_radius: int = 1000,
                        find_nearest: bool = True) -> Dict:
        """Fetch Street View imagery for a given location and save to output directory.

        If heading is not specified, fetches a 360° panorama (at headings 0, 90, 180, 270).

        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            output_dir: Directory to save downloaded imagery
            heading: Optional specific heading in degrees (0-360, default captures panorama)
            fov: Field of view (20-120, default 90)
            pitch: Camera pitch (-90 to 90, default 0)
            radius: Maximum radius in meters to look for Street View imagery
            max_search_radius: Maximum radius to search for nearby imagery if none found at exact location
            find_nearest: Whether to search for nearest imagery if none found at exact location

        Returns:
            Dictionary with download results including success status
        """
        try:
            # Create output directory if needed
            os.makedirs(output_dir, exist_ok=True)

            # First check if Street View is available for this location
            metadata = self.check_street_view_availability(lat, lng, radius)

            # If no imagery found and find_nearest is True, search for nearest imagery
            if metadata.get("status") != "OK" and find_nearest:
                logger.info(f"No Street View imagery at exact location ({lat}, {lng}), searching nearby...")
                metadata = self.find_nearest_street_view(lat, lng, max_radius=max_search_radius)

            # If still no imagery found, return error
            if metadata.get("status") != "OK":
                logger.warning(f"No Street View imagery available near location ({lat}, {lng})")
                return {
                    "success": False,
                    "error": f"No Street View imagery available within {max_search_radius}m of this location",
                    "location": (lat, lng)
                }

            # Extract the actual location where imagery was found (may differ from requested location)
            actual_lat = metadata.get("location", {}).get("lat", lat)
            actual_lng = metadata.get("location", {}).get("lng", lng)

            # Save the found location metadata
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Add info about search distance if available
            if "search_distance" in metadata:
                logger.info(f"Using Street View imagery {metadata['search_distance']:.1f}m away from requested location")

            downloaded_paths = []

            # If heading is specified, download just that view
            if heading is not None:
                image_path = os.path.join(output_dir, f"street_view_{heading}.jpg")
                path = self.download_street_view_image(
                    actual_lat, actual_lng, image_path,
                    heading=heading, fov=fov, pitch=pitch
                )

                if path:
                    downloaded_paths.append(path)

            # Otherwise download a panorama (4 images at 90° intervals)
            else:
                for h in [0, 90, 180, 270]:
                    image_path = os.path.join(output_dir, f"street_view_{h}.jpg")
                    path = self.download_street_view_image(
                        actual_lat, actual_lng, image_path,
                        heading=h, fov=fov, pitch=pitch
                    )

                    if path:
                        downloaded_paths.append(path)

            # Return the results
            return {
                "success": len(downloaded_paths) > 0,
                "downloaded_images": len(downloaded_paths),
                "paths": downloaded_paths,
                "location": (actual_lat, actual_lng),
                "original_location": (lat, lng),
                "metadata_path": metadata_path,
                "search_distance": metadata.get("search_distance", 0)
            }

        except Exception as e:
            logger.error(f"Error fetching Street View imagery: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "location": (lat, lng)
            }
    
    def fetch_panorama(self, lat: float, lng: float, output_dir: str,
                     pitch_angles: List[int] = [0, 15, -15],
                     radius: int = 100,
                     max_search_radius: int = 1000,
                     find_nearest: bool = True) -> Dict:
        """Fetch a complete Street View panorama for a given location.

        This captures images at multiple headings and pitch angles to create a full panorama.

        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            output_dir: Directory to save downloaded imagery
            pitch_angles: List of pitch angles to capture
            radius: Maximum radius in meters to look for Street View imagery
            max_search_radius: Maximum radius to search for nearby imagery if none found at exact location
            find_nearest: Whether to search for nearest imagery if none found at exact location

        Returns:
            Dictionary with download results including success status
        """
        try:
            # Create output directory if needed
            os.makedirs(output_dir, exist_ok=True)

            # First check if Street View is available for this location
            metadata = self.check_street_view_availability(lat, lng, radius)

            # If no imagery found and find_nearest is True, search for nearest imagery
            if metadata.get("status") != "OK" and find_nearest:
                logger.info(f"No Street View imagery at exact location ({lat}, {lng}), searching nearby...")
                metadata = self.find_nearest_street_view(lat, lng, max_radius=max_search_radius)

            # If still no imagery found, return error
            if metadata.get("status") != "OK":
                logger.warning(f"No Street View imagery available near location ({lat}, {lng})")
                return {
                    "success": False,
                    "error": f"No Street View imagery available within {max_search_radius}m of this location",
                    "location": (lat, lng)
                }

            # Extract the actual location where imagery was found (may differ from requested location)
            actual_lat = metadata.get("location", {}).get("lat", lat)
            actual_lng = metadata.get("location", {}).get("lng", lng)

            # Save the found location metadata
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Add info about search distance if available
            if "search_distance" in metadata:
                logger.info(f"Using Street View imagery {metadata['search_distance']:.1f}m away from requested location")

            downloaded_paths = []

            # Capture images at 30° intervals for each pitch angle
            for pitch in pitch_angles:
                pitch_dir = os.path.join(output_dir, f"pitch_{pitch}")
                os.makedirs(pitch_dir, exist_ok=True)

                for heading in range(0, 360, 30):
                    image_path = os.path.join(pitch_dir, f"street_view_{heading}.jpg")
                    path = self.download_street_view_image(
                        actual_lat, actual_lng, image_path,
                        heading=heading, fov=90, pitch=pitch
                    )

                    if path:
                        downloaded_paths.append(path)

            # Return the results
            return {
                "success": len(downloaded_paths) > 0,
                "downloaded_images": len(downloaded_paths),
                "paths": downloaded_paths,
                "location": (actual_lat, actual_lng),
                "original_location": (lat, lng),
                "metadata_path": metadata_path,
                "search_distance": metadata.get("search_distance", 0)
            }

        except Exception as e:
            logger.error(f"Error fetching Street View panorama: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "location": (lat, lng)
            }

    def process_point(self, lat: float, lng: float, point_index: int, output_dir: str,
                    panorama: bool = True, max_search_radius: int = 1000) -> Dict[str, Any]:
        """Process a single point by fetching Street View imagery.

        Args:
            lat: Latitude of the point
            lng: Longitude of the point
            point_index: Index of the point for directory naming
            output_dir: Base output directory
            panorama: Whether to fetch full panorama
            max_search_radius: Maximum radius to search for nearby imagery

        Returns:
            Dictionary with results for this point
        """
        try:
            # Create output directory for this point
            point_dir = os.path.join(output_dir, f"point_{point_index:06d}")
            os.makedirs(point_dir, exist_ok=True)

            # Save point location
            with open(os.path.join(point_dir, "location.json"), "w") as f:
                json.dump({
                    "original": {"lat": lat, "lng": lng},
                    "index": point_index
                }, f, indent=2)

            # Fetch Street View imagery
            if panorama:
                logger.info(f"Fetching panorama for point {point_index} at ({lat}, {lng})")
                result = self.fetch_panorama(
                    lat, lng, point_dir,
                    max_search_radius=max_search_radius,
                    find_nearest=True
                )
            else:
                logger.info(f"Fetching Street View for point {point_index} at ({lat}, {lng})")
                result = self.fetch_street_view(
                    lat, lng, point_dir,
                    max_search_radius=max_search_radius,
                    find_nearest=True
                )

            # Add point index to result
            result["point_index"] = point_index

            # Save result summary
            with open(os.path.join(point_dir, "result.json"), "w") as f:
                # Create a serializable result by removing non-serializable items
                serializable_result = {k: v for k, v in result.items()
                                     if k not in ["paths"]}
                json.dump(serializable_result, f, indent=2)

            return result

        except Exception as e:
            logger.error(f"Error processing point {point_index} at ({lat}, {lng}): {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "point_index": point_index,
                "location": (lat, lng)
            }

    def fetch_street_view_along_road(self, points: List[Tuple[float, float, int]],
                                   output_dir: str,
                                   panorama: bool = True,
                                   max_search_radius: int = 1000,
                                   max_workers: int = 4) -> Dict[str, Any]:
        """Fetch Street View images along a road at specified sampling points.

        Args:
            points: List of (lat, lng, index) tuples representing sampling points
            output_dir: Base directory to save imagery
            panorama: Whether to fetch full panorama or regular images
            max_search_radius: Maximum radius to search for nearby imagery
            max_workers: Maximum number of worker threads

        Returns:
            Dictionary with results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process points in parallel
        logger.info(f"Processing {len(points)} points with {max_workers} workers")

        results = {
            "success_count": 0,
            "failed_count": 0,
            "no_imagery_count": 0,
            "total_images": 0,
            "points_with_imagery": []
        }

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_point = {}

            for lat, lng, idx in points:
                # Submit task
                future = executor.submit(
                    self.process_point,
                    lat, lng, idx, output_dir,
                    panorama, max_search_radius
                )

                future_to_point[future] = (lat, lng, idx)

            # Process results as they complete
            start_time = time.time()
            for i, future in enumerate(as_completed(future_to_point), 1):
                point = future_to_point[future]
                lat, lng, idx = point

                try:
                    result = future.result()

                    if result.get("success", False):
                        results["success_count"] += 1
                        results["total_images"] += result.get("downloaded_images", 0)
                        results["points_with_imagery"].append({
                            "point_index": idx,
                            "location": result.get("location"),
                            "original_location": (lat, lng),
                            "image_count": result.get("downloaded_images", 0),
                            "search_distance": result.get("search_distance", 0)
                        })
                    else:
                        if "No Street View imagery" in result.get("error", ""):
                            results["no_imagery_count"] += 1
                        else:
                            results["failed_count"] += 1

                    # Log progress periodically
                    if i % 10 == 0 or i == len(points):
                        elapsed = time.time() - start_time
                        logger.info(f"Processed {i}/{len(points)} points ({i/len(points)*100:.1f}%) in {elapsed:.1f}s")
                        logger.info(f"Success: {results['success_count']}, No imagery: {results['no_imagery_count']}, Failed: {results['failed_count']}")

                except Exception as e:
                    logger.error(f"Error processing point {point}: {str(e)}")
                    results["failed_count"] += 1

        # Calculate elapsed time
        elapsed = time.time() - start_time

        # Add summary information
        results["total_points"] = len(points)
        results["elapsed_seconds"] = elapsed
        results["points_per_second"] = len(points) / elapsed if elapsed > 0 else 0

        # Save summary
        summary_path = os.path.join(output_dir, "street_view_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Completed processing {len(points)} points in {elapsed:.1f}s")
        logger.info(f"Success: {results['success_count']}, No imagery: {results['no_imagery_count']}, Failed: {results['failed_count']}")
        logger.info(f"Downloaded {results['total_images']} Street View images")

        return {
            "success": results["success_count"] > 0,
            "points_processed": len(points),
            "points_with_imagery": results["success_count"],
            "images_downloaded": results["total_images"],
            "summary_path": summary_path,
            "output_dir": output_dir
        }

def main():
    """Main function for testing the module."""
    import argparse

    parser = argparse.ArgumentParser(description="Google Street View Integration")
    parser.add_argument("--api-key", help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)")
    parser.add_argument("--output", default="./street_view_output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Single point mode
    point_parser = subparsers.add_parser("point", help="Fetch Street View for a single point")
    point_parser.add_argument("--lat", type=float, required=True, help="Latitude")
    point_parser.add_argument("--lng", type=float, required=True, help="Longitude")
    point_parser.add_argument("--heading", type=int, help="Specific heading (0-360)")
    point_parser.add_argument("--fov", type=int, default=90, help="Field of view (20-120)")
    point_parser.add_argument("--pitch", type=int, default=0, help="Camera pitch (-90 to 90)")
    point_parser.add_argument("--radius", type=int, default=100, help="Search radius in meters")
    point_parser.add_argument("--panorama", action="store_true", help="Capture full panorama")

    # Road network mode
    road_parser = subparsers.add_parser("road", help="Fetch Street View along a road network")
    road_parser.add_argument("--points-file", required=True, help="Path to JSON file with sampling points")
    road_parser.add_argument("--panorama", action="store_true", help="Capture full panoramas")
    road_parser.add_argument("--max-radius", type=int, default=1000, help="Maximum search radius in meters")
    road_parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize integration
    street_view = GoogleStreetViewIntegration(api_key=args.api_key)

    # If no mode specified, default to 'point' if lat/lng provided
    if not args.mode:
        if hasattr(args, 'lat') and hasattr(args, 'lng'):
            args.mode = 'point'
        else:
            parser.print_help()
            return 1

    # Process based on mode
    if args.mode == "point":
        if args.panorama:
            logger.info(f"Fetching Street View panorama for location ({args.lat}, {args.lng})")
            result = street_view.fetch_panorama(
                args.lat, args.lng, args.output,
                radius=args.radius
            )
        else:
            logger.info(f"Fetching Street View imagery for location ({args.lat}, {args.lng})")
            result = street_view.fetch_street_view(
                args.lat, args.lng, args.output,
                heading=args.heading, fov=args.fov, pitch=args.pitch, radius=args.radius
            )

        if result.get("success", False):
            print(f"Successfully downloaded {result['downloaded_images']} Street View images")
            print(f"Images saved to {args.output}")
        else:
            print(f"Failed to download Street View imagery: {result.get('error', 'Unknown error')}")
            return 1

    elif args.mode == "road":
        # Load the points file
        try:
            with open(args.points_file, 'r') as f:
                points_data = json.load(f)

            # Extract points from GeoJSON
            sample_points = []
            if "features" in points_data:
                # GeoJSON format
                for feature in points_data["features"]:
                    coords = feature["geometry"]["coordinates"]
                    index = feature["properties"]["index"]
                    # Convert from [lon, lat] to (lat, lon, index)
                    sample_points.append((coords[1], coords[0], index))
            elif "points" in points_data:
                # Custom format
                for i, point in enumerate(points_data["points"]):
                    if isinstance(point, list):
                        # [lat, lng] format
                        sample_points.append((point[0], point[1], i))
                    elif isinstance(point, dict):
                        # {"lat": lat, "lng": lng} format
                        sample_points.append((point["lat"], point["lng"], i))

            if not sample_points:
                print(f"No valid points found in {args.points_file}")
                return 1

            print(f"Processing {len(sample_points)} points from {args.points_file}")

            result = street_view.fetch_street_view_along_road(
                sample_points,
                args.output,
                panorama=args.panorama,
                max_search_radius=args.max_radius,
                max_workers=args.workers
            )

            if result.get("success", False):
                print(f"Successfully processed {result['points_processed']} points")
                print(f"Found imagery at {result['points_with_imagery']} locations")
                print(f"Downloaded {result['images_downloaded']} Street View images")
                print(f"Results saved to {result['summary_path']}")
            else:
                print(f"Failed to process points: {result.get('error', 'Unknown error')}")
                return 1

        except Exception as e:
            print(f"Error processing points file: {str(e)}")
            return 1

    else:
        parser.print_help()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())