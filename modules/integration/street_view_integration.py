#!/usr/bin/env python3
"""
Google Street View Integration Module
------------------------------------
This module provides integration with Google Maps Platform's Street View API,
allowing for fetching and processing of Street View imagery within the Arcanum system.

Implementation follows the official Google Maps Platform documentation:
https://developers.google.com/maps/documentation/tile/streetview
"""

import os
import sys
import logging
import json
import requests
import time
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class GoogleStreetViewIntegration:
    """Class that provides integration with Google Maps Platform's Street View API."""

    def __init__(self,
                api_key: str = None,
                cache_dir: str = None,
                retries: int = 3,
                timeout: int = 30):
        """Initialize the Google Street View integration.

        Args:
            api_key: Google Maps API key with Street View access (from env var if None)
            cache_dir: Directory to cache downloaded images
            retries: Number of retries for failed requests
            timeout: Timeout in seconds for requests
        """
        # Load API key from environment if not provided
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            logger.warning("No Google Maps API key provided. Set GOOGLE_MAPS_API_KEY environment variable.")
        
        # Set up base URLs for the new Tile API
        self.base_url = "https://tile.googleapis.com/v1/streetview"
        self.tiles_endpoint = "/tiles"
        self.pano_ids_endpoint = "/panoIds"
        self.thumbnail_endpoint = "/thumbnail"
        self.metadata_endpoint = "/metadata"
        
        self.retries = retries
        self.timeout = timeout
        
        # Create a session token for this integration instance
        self.session_token = f"arcanum_session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        logger.info(f"Created Street View session token: {self.session_token}")
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(os.path.expanduser("~/.cache/arcanum/street_view"))
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Google Street View cache directory: {self.cache_dir}")
        
        logger.info("Google Street View Integration initialized")
    
    def get_pano_ids_url(self, lat: float, lng: float, radius: int = 100) -> str:
        """Get the URL for finding Street View panorama IDs near a location.
        
        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            radius: Maximum radius in meters to look for Street View imagery
            
        Returns:
            URL for Street View panorama IDs with API key included
        """
        # Validate parameters
        if lat < -90 or lat > 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        if lng < -180 or lng > 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {lng}")
        
        # Build the URL
        url = f"{self.base_url}{self.pano_ids_endpoint}"
        
        # Add parameters
        params = {
            "location": f"{lat},{lng}",
            "radiusMeters": radius,
            "key": self.api_key,
            "sessionToken": self.session_token
        }
        
        # Append parameters
        url = f"{url}?{urlencode(params)}"
        
        return url
    
    def get_metadata_url(self, pano_id: str) -> str:
        """Get the URL for Street View panorama metadata.
        
        Args:
            pano_id: Panorama ID
            
        Returns:
            URL for Street View metadata with API key included
        """
        # Build the URL
        url = f"{self.base_url}{self.metadata_endpoint}"
        
        # Add parameters
        params = {
            "panoId": pano_id,
            "key": self.api_key,
            "sessionToken": self.session_token
        }
        
        # Append parameters
        url = f"{url}?{urlencode(params)}"
        
        return url
    
    def get_thumbnail_url(self, pano_id: str, size: str = "640x640", 
                         heading: float = 0, pitch: float = 0, fov: float = 90) -> str:
        """Get the URL for a Street View thumbnail image.
        
        Args:
            pano_id: Panorama ID
            size: Image size in pixels (width x height)
            heading: Camera heading in degrees (0-360)
            pitch: Camera pitch in degrees (-90 to 90)
            fov: Field of view in degrees (10-120)
            
        Returns:
            URL for Street View thumbnail with API key included
        """
        # Build the URL
        url = f"{self.base_url}{self.thumbnail_endpoint}"
        
        # Add parameters
        params = {
            "panoId": pano_id,
            "size": size,
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "key": self.api_key,
            "sessionToken": self.session_token
        }
        
        # Append parameters
        url = f"{url}?{urlencode(params)}"
        
        return url
    
    def get_tile_url(self, pano_id: str, zoom: int, x: int, y: int) -> str:
        """Get the URL for a Street View tile.
        
        Args:
            pano_id: Panorama ID
            zoom: Zoom level (0-5)
            x: Tile x coordinate
            y: Tile y coordinate
            
        Returns:
            URL for Street View tile with API key included
        """
        # Build the URL according to the new tiles API format
        url = f"{self.base_url}{self.tiles_endpoint}/{zoom}/{x}/{y}"
        
        # Add parameters
        params = {
            "panoId": pano_id,
            "key": self.api_key,
            "sessionToken": self.session_token
        }
        
        # Append parameters
        url = f"{url}?{urlencode(params)}"
        
        return url
    
    def find_panoramas_near_location(self, lat: float, lng: float, radius: int = 100) -> Dict:
        """Find Street View panoramas near a given location.
        
        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            radius: Maximum radius in meters to look for Street View imagery
            
        Returns:
            Dictionary with panorama information
        """
        url = self.get_pano_ids_url(lat, lng, radius)
        logger.info(f"Finding panoramas near location ({lat}, {lng}) within {radius}m")
        
        # Try to fetch with retries
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Parse JSON
                pano_data = response.json()
                
                # Cache the data
                cache_path = self.cache_dir / f"pano_ids_{lat}_{lng}_{radius}.json"
                with open(cache_path, "w") as f:
                    json.dump(pano_data, f, indent=2)
                
                logger.info(f"Successfully fetched panorama data (cached at {cache_path})")
                
                # Add query location to result
                if "panoInfos" in pano_data:
                    pano_data["query_location"] = {"lat": lat, "lng": lng}
                    pano_data["query_radius"] = radius
                
                return pano_data
                
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed: {str(e)}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
        
        logger.error(f"Failed to find panoramas after {self.retries} attempts.")
        return {"error": f"Failed to find panoramas after {self.retries} attempts.",
                "panoInfos": []}
    
    def get_panorama_metadata(self, pano_id: str) -> Dict:
        """Get metadata for a specific panorama.
        
        Args:
            pano_id: Panorama ID
            
        Returns:
            Dictionary with panorama metadata
        """
        url = self.get_metadata_url(pano_id)
        logger.info(f"Getting metadata for panorama {pano_id}")
        
        # Try to fetch with retries
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Parse JSON
                metadata = response.json()
                
                # Cache the metadata
                cache_path = self.cache_dir / f"metadata_{pano_id}.json"
                with open(cache_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Successfully fetched panorama metadata (cached at {cache_path})")
                return metadata
                
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed: {str(e)}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
        
        logger.error(f"Failed to fetch panorama metadata after {self.retries} attempts.")
        return {"error": f"Failed to fetch panorama metadata after {self.retries} attempts."}
    
    def download_panorama_thumbnail(self, pano_id: str, output_path: str,
                                  heading: float = 0, pitch: float = 0, 
                                  fov: float = 90, size: str = "640x640") -> Optional[str]:
        """Download a thumbnail image for a panorama.
        
        Args:
            pano_id: Panorama ID
            output_path: Path to save the image
            heading: Camera heading in degrees (0-360)
            pitch: Camera pitch in degrees (-90 to 90)
            fov: Field of view in degrees (10-120)
            size: Image size in pixels (width x height)
            
        Returns:
            Path to the downloaded image, or None if failed
        """
        url = self.get_thumbnail_url(pano_id, size, heading, pitch, fov)
        
        # Create parent directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading panorama thumbnail from {url} to {output_path}")
        
        # Try to fetch with retries
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Check if we got an image
                if response.headers.get('content-type') not in ['image/jpeg', 'image/png']:
                    logger.warning(f"Received non-image response: {response.headers.get('content-type')}")
                    return None
                
                # Save the image
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded panorama thumbnail to {output_path}")
                return output_path
                
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed: {str(e)}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
        
        logger.error(f"Failed to download panorama thumbnail after {self.retries} attempts.")
        return None
    
    def download_panorama_tile(self, pano_id: str, zoom: int, x: int, y: int, output_path: str) -> Optional[str]:
        """Download a tile from a panorama.
        
        Args:
            pano_id: Panorama ID
            zoom: Zoom level (0-5)
            x: Tile x coordinate
            y: Tile y coordinate
            output_path: Path to save the tile
            
        Returns:
            Path to the downloaded tile, or None if failed
        """
        url = self.get_tile_url(pano_id, zoom, x, y)
        
        # Create parent directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading panorama tile from {url} to {output_path}")
        
        # Try to fetch with retries
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Check if we got an image
                if response.headers.get('content-type') not in ['image/jpeg', 'image/png']:
                    logger.warning(f"Received non-image response: {response.headers.get('content-type')}")
                    return None
                
                # Save the image
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded panorama tile to {output_path}")
                return output_path
                
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt+1}/{self.retries} failed: {str(e)}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
        
        logger.error(f"Failed to download panorama tile after {self.retries} attempts.")
        return None
    
    def download_panorama_views(self, pano_id: str, output_dir: str, 
                             headings: List[float], pitches: List[float] = [0],
                             fov: float = 90, size: str = "640x640") -> Dict:
        """Download multiple views from a panorama at different headings and pitches.
        
        Args:
            pano_id: Panorama ID
            output_dir: Directory to save the images
            headings: List of headings in degrees
            pitches: List of pitches in degrees
            fov: Field of view in degrees (10-120)
            size: Image size in pixels (width x height)
            
        Returns:
            Dictionary with download results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get and save panorama metadata
        metadata = self.get_panorama_metadata(pano_id)
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        downloaded_paths = []
        
        # Download views for each heading and pitch
        for pitch in pitches:
            for heading in headings:
                # Generate filename
                filename = f"pano_{pano_id}_h{int(heading)}_p{int(pitch)}_fov{int(fov)}.jpg"
                output_path = os.path.join(output_dir, filename)
                
                # Download thumbnail
                path = self.download_panorama_thumbnail(
                    pano_id, output_path, heading, pitch, fov, size
                )
                
                if path:
                    downloaded_paths.append(path)
        
        # Return results
        return {
            "success": len(downloaded_paths) > 0,
            "pano_id": pano_id,
            "downloaded_views": len(downloaded_paths),
            "paths": downloaded_paths,
            "metadata_path": metadata_path,
            "copyright": metadata.get("copyright", "")
        }
    
    def download_complete_panorama(self, pano_id: str, output_dir: str, zoom: int = 2) -> Dict:
        """Download all tiles needed to recreate a complete 360° panorama.
        
        Args:
            pano_id: Panorama ID
            output_dir: Directory to save the tiles
            zoom: Zoom level (0-5, with 0 being lowest resolution)
            
        Returns:
            Dictionary with download results
        """
        # Create output directory
        tiles_dir = os.path.join(output_dir, f"tiles_z{zoom}")
        os.makedirs(tiles_dir, exist_ok=True)
        
        # Get and save panorama metadata
        metadata = self.get_panorama_metadata(pano_id)
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Calculate number of tiles at the specified zoom level
        # Each zoom level has 2^zoom tiles horizontally and 2^(zoom-1) tiles vertically
        horizontal_tiles = 2 ** zoom
        vertical_tiles = 2 ** (zoom - 1) if zoom > 0 else 1
        
        downloaded_paths = []
        
        # Download all tiles
        for y in range(vertical_tiles):
            for x in range(horizontal_tiles):
                # Generate filename
                filename = f"tile_{pano_id}_z{zoom}_x{x}_y{y}.jpg"
                output_path = os.path.join(tiles_dir, filename)
                
                # Download tile
                path = self.download_panorama_tile(
                    pano_id, zoom, x, y, output_path
                )
                
                if path:
                    downloaded_paths.append(path)
        
        # Return results
        return {
            "success": len(downloaded_paths) > 0,
            "pano_id": pano_id,
            "zoom": zoom,
            "downloaded_tiles": len(downloaded_paths),
            "total_tiles": horizontal_tiles * vertical_tiles,
            "paths": downloaded_paths,
            "metadata_path": metadata_path,
            "copyright": metadata.get("copyright", "")
        }
    
    def fetch_street_view(self, lat: float, lng: float, output_dir: str,
                       headings: Optional[List[float]] = None,
                       download_tiles: bool = False,
                       tile_zoom: int = 2,
                       radius: int = 100) -> Dict:
        """Fetch Street View imagery for a given location and save to output directory.
        
        Args:
            lat: Latitude of the location
            lng: Longitude of the location
            output_dir: Directory to save downloaded imagery
            headings: Optional list of headings to capture (default is 4 cardinal directions)
            download_tiles: Whether to download individual tiles for the panorama
            tile_zoom: Zoom level for tiles (0-5)
            radius: Maximum radius in meters to look for Street View imagery
            
        Returns:
            Dictionary with download results
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Use default headings (cardinal directions) if not specified
            if headings is None:
                headings = [0, 90, 180, 270]
            
            # Find panoramas near the location
            pano_data = self.find_panoramas_near_location(lat, lng, radius)
            
            # Save original request details
            with open(os.path.join(output_dir, "request.json"), "w") as f:
                json.dump({
                    "lat": lat,
                    "lng": lng,
                    "radius": radius,
                    "headings": headings,
                    "download_tiles": download_tiles,
                    "tile_zoom": tile_zoom
                }, f, indent=2)
            
            # Check if we found any panoramas
            if "panoInfos" not in pano_data or not pano_data["panoInfos"]:
                logger.warning(f"No panoramas found near location ({lat}, {lng}) within {radius}m")
                return {
                    "success": False,
                    "error": f"No Street View imagery available within {radius}m",
                    "location": {"lat": lat, "lng": lng}
                }
            
            # Save the panorama search results
            with open(os.path.join(output_dir, "panoramas.json"), "w") as f:
                json.dump(pano_data, f, indent=2)
            
            # Use the closest panorama
            closest_pano = pano_data["panoInfos"][0]
            pano_id = closest_pano["panoId"]
            
            # Create a directory for this panorama
            pano_dir = os.path.join(output_dir, f"pano_{pano_id}")
            os.makedirs(pano_dir, exist_ok=True)
            
            logger.info(f"Processing panorama {pano_id}")
            
            # Download the views at specified headings
            views_result = self.download_panorama_views(
                pano_id, pano_dir, headings=headings
            )
            
            # Download tiles if requested
            tiles_result = None
            if download_tiles:
                tiles_result = self.download_complete_panorama(
                    pano_id, pano_dir, zoom=tile_zoom
                )
            
            # Return the results
            return {
                "success": views_result["success"],
                "pano_id": pano_id,
                "location": closest_pano.get("location", {"lat": lat, "lng": lng}),
                "query_location": {"lat": lat, "lng": lng},
                "downloaded_views": views_result.get("downloaded_views", 0),
                "downloaded_tiles": tiles_result.get("downloaded_tiles", 0) if tiles_result else 0,
                "paths": views_result.get("paths", []),
                "copyright": views_result.get("copyright", ""),
                "output_directory": pano_dir
            }
            
        except Exception as e:
            logger.error(f"Error fetching Street View imagery: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "location": {"lat": lat, "lng": lng}
            }
    
    def process_point(self, lat: float, lng: float, point_index: int, output_dir: str,
                    download_tiles: bool = False, tile_zoom: int = 2, radius: int = 100) -> Dict:
        """Process a single point by fetching Street View imagery.
        
        Args:
            lat: Latitude of the point
            lng: Longitude of the point
            point_index: Index of the point for directory naming
            output_dir: Base output directory
            download_tiles: Whether to download individual tiles
            tile_zoom: Zoom level for tiles (0-5)
            radius: Maximum radius in meters to look for Street View imagery
            
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
                    "location": {"lat": lat, "lng": lng},
                    "index": point_index
                }, f, indent=2)
            
            # Fetch Street View imagery
            logger.info(f"Fetching Street View for point {point_index} at ({lat}, {lng})")
            result = self.fetch_street_view(
                lat, lng, point_dir,
                download_tiles=download_tiles,
                tile_zoom=tile_zoom,
                radius=radius
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
                "location": {"lat": lat, "lng": lng}
            }
    
    def fetch_street_view_along_road(self, points: List[Tuple[float, float, int]],
                                  output_dir: str,
                                  download_tiles: bool = False,
                                  tile_zoom: int = 2,
                                  radius: int = 100,
                                  max_workers: int = 4) -> Dict:
        """Fetch Street View images along a road at specified sampling points.
        
        Args:
            points: List of (lat, lng, index) tuples representing sampling points
            output_dir: Base directory to save imagery
            download_tiles: Whether to download individual tiles
            tile_zoom: Zoom level for tiles (0-5)
            radius: Maximum radius in meters to look for Street View imagery
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
            "total_views": 0,
            "total_tiles": 0,
            "points_with_imagery": []
        }
        
        # Create points metadata file
        with open(os.path.join(output_dir, "points.json"), "w") as f:
            json.dump({
                "points": [(lat, lng, idx) for lat, lng, idx in points],
                "count": len(points),
                "download_tiles": download_tiles,
                "tile_zoom": tile_zoom,
                "radius": radius
            }, f, indent=2)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_point = {}
            
            for lat, lng, idx in points:
                # Submit task
                future = executor.submit(
                    self.process_point,
                    lat, lng, idx, output_dir,
                    download_tiles, tile_zoom, radius
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
                        results["total_views"] += result.get("downloaded_views", 0)
                        results["total_tiles"] += result.get("downloaded_tiles", 0)
                        results["points_with_imagery"].append({
                            "point_index": idx,
                            "location": {"lat": lat, "lng": lng},
                            "pano_id": result.get("pano_id"),
                            "view_count": result.get("downloaded_views", 0),
                            "tile_count": result.get("downloaded_tiles", 0)
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
        logger.info(f"Downloaded {results['total_views']} views and {results['total_tiles']} tiles")
        
        return {
            "success": results["success_count"] > 0,
            "points_processed": len(points),
            "points_with_imagery": results["success_count"],
            "views_downloaded": results["total_views"],
            "tiles_downloaded": results["total_tiles"],
            "summary_path": summary_path,
            "output_dir": output_dir
        }

# Module level functions for direct import
def fetch_street_view(lat: float, lng: float, output_dir: str, api_key: str = None,
                    headings: Optional[List[float]] = None, radius: int = 100,
                    download_tiles: bool = False, tile_zoom: int = 2,
                    cache_dir: str = None) -> Dict:
    """Fetch Street View imagery for a given location and save to output directory.

    This is a module-level function that creates a GoogleStreetViewIntegration instance
    and delegates to its fetch_street_view method.

    Args:
        lat: Latitude of the location
        lng: Longitude of the location
        output_dir: Directory to save downloaded imagery
        api_key: Google Maps API key with Street View access (from env var if None)
        headings: Optional list of headings to capture (default is 4 cardinal directions)
        radius: Maximum radius in meters to look for Street View imagery
        download_tiles: Whether to download individual tiles
        tile_zoom: Zoom level for tiles (0-5)
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
    return integration.fetch_street_view(
        lat, lng, output_dir, 
        headings=headings, 
        download_tiles=download_tiles, 
        tile_zoom=tile_zoom, 
        radius=radius
    )

def fetch_street_view_along_roads(
    points: List[Tuple[float, float, int]],
    output_dir: str,
    api_key: str = None,
    download_tiles: bool = False,
    tile_zoom: int = 2,
    radius: int = 100,
    max_workers: int = 4,
    cache_dir: str = None
) -> Dict[str, Any]:
    """Fetch Street View images along a road network at sampled points.

    Args:
        points: List of (lat, lng, index) tuples representing sample points
        output_dir: Directory to save Street View imagery
        api_key: Google Maps API key with Street View access (from env var if None)
        download_tiles: Whether to download complete panorama tiles
        tile_zoom: Zoom level for panorama tiles (0-5)
        radius: Maximum radius in meters to look for Street View imagery
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
        download_tiles=download_tiles,
        tile_zoom=tile_zoom,
        radius=radius,
        max_workers=max_workers
    )

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
    point_parser.add_argument("--radius", type=int, default=100, help="Search radius in meters")
    point_parser.add_argument("--tiles", action="store_true", help="Download complete panorama tiles")
    point_parser.add_argument("--zoom", type=int, default=2, help="Zoom level for tiles (0-5)")

    # Road network mode
    road_parser = subparsers.add_parser("road", help="Fetch Street View along a road network")
    road_parser.add_argument("--points-file", required=True, help="Path to JSON file with sampling points")
    road_parser.add_argument("--tiles", action="store_true", help="Download complete panorama tiles")
    road_parser.add_argument("--zoom", type=int, default=2, help="Zoom level for tiles (0-5)")
    road_parser.add_argument("--radius", type=int, default=100, help="Search radius in meters")
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
        logger.info(f"Fetching Street View imagery for location ({args.lat}, {args.lng})")
        result = street_view.fetch_street_view(
            args.lat, args.lng, args.output,
            download_tiles=args.tiles,
            tile_zoom=args.zoom,
            radius=args.radius
        )

        if result.get("success", False):
            print(f"Successfully downloaded {result['downloaded_views']} views")
            if args.tiles:
                print(f"Downloaded {result['downloaded_tiles']} panorama tiles")
            print(f"Images saved to {result['output_directory']}")
        else:
            print(f"Failed to download Street View imagery: {result.get('error', 'Unknown error')}")
            return 1

    elif args.mode == "road":
        # Load the points file
        try:
            with open(args.points_file, 'r') as f:
                points_data = json.load(f)

            # Extract points from file
            sample_points = []
            if "features" in points_data:
                # GeoJSON format
                for i, feature in enumerate(points_data["features"]):
                    coords = feature["geometry"]["coordinates"]
                    # Get index from properties or use sequence number
                    index = feature["properties"].get("index", i)
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
                download_tiles=args.tiles,
                tile_zoom=args.zoom,
                radius=args.radius,
                max_workers=args.workers
            )

            if result.get("success", False):
                print(f"Successfully processed {result['points_processed']} points")
                print(f"Found imagery at {result['points_with_imagery']} locations")
                print(f"Downloaded {result['views_downloaded']} views and {result['tiles_downloaded']} tiles")
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