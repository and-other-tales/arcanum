#!/usr/bin/env python3
"""
Tile Server Integration
----------------------
Manages uploading, configuration, and integration of Unity assets with tile servers.
Supports various server types including Google Cloud Storage, AWS S3, and standard HTTP servers.
"""

import os
import sys
import logging
import json
import re
import shutil
import time
import zipfile
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import uuid
import hashlib

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TileServerIntegration:
    """
    Manages the integration with various tile server types for Unity assets.
    """
    
    def __init__(self, 
                output_dir: str,
                server_url: str,
                credentials_path: str = None,
                server_type: str = None):
        """
        Initialize the tile server integration.
        
        Args:
            output_dir: Directory containing files to upload
            server_url: URL of the tile server (gs://, s3://, or http://)
            credentials_path: Path to credentials file
            server_type: Type of server (gcs, s3, http) - auto-detected if None
        """
        self.output_dir = Path(output_dir)
        self.server_url = server_url
        self.credentials_path = credentials_path
        
        # Detect server type from URL if not provided
        if server_type is None:
            if server_url.startswith("gs://"):
                self.server_type = "gcs"
            elif server_url.startswith("s3://"):
                self.server_type = "s3"
            elif server_url.startswith("http://") or server_url.startswith("https://"):
                self.server_type = "http"
            else:
                self.server_type = "local"
                logger.warning(f"Unknown server URL format: {server_url}. Assuming local path.")
        else:
            self.server_type = server_type
        
        # Create output directories
        self.manifest_dir = self.output_dir / "manifests"
        os.makedirs(self.manifest_dir, exist_ok=True)
        
        # Tile server configuration
        self.tile_size = 256
        self.max_zoom_level = 20
        self.min_zoom_level = 0
        
        logger.info(f"TileServerIntegration initialized with server type: {self.server_type}")
        logger.info(f"Server URL: {server_url}")
    
    def upload_assets(self, 
                     assets_dir: str = None, 
                     zip_files: bool = False,
                     include_pattern: str = None,
                     exclude_pattern: str = None) -> Dict:
        """
        Upload assets to the tile server.
        
        Args:
            assets_dir: Directory containing assets to upload (defaults to output_dir)
            zip_files: Whether to zip files by type before uploading
            include_pattern: Regex pattern for files to include
            exclude_pattern: Regex pattern for files to exclude
            
        Returns:
            Dictionary with upload results
        """
        # Use assets_dir if provided, otherwise use output_dir
        assets_dir = Path(assets_dir) if assets_dir else self.output_dir
        
        # Verify assets directory exists
        if not assets_dir.exists() or not assets_dir.is_dir():
            logger.error(f"Assets directory not found: {assets_dir}")
            return {"success": False, "error": "Assets directory not found"}
        
        # Create include/exclude patterns
        include_re = re.compile(include_pattern) if include_pattern else None
        exclude_re = re.compile(exclude_pattern) if exclude_pattern else None
        
        # Collect files to upload
        files_to_upload = []
        for root, dirs, files in os.walk(assets_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(assets_dir)
                
                # Check patterns
                if include_re and not include_re.search(str(rel_path)):
                    continue
                if exclude_re and exclude_re.search(str(rel_path)):
                    continue
                
                files_to_upload.append((file_path, rel_path))
        
        # Group files by type if zipping
        if zip_files:
            return self._upload_zipped_assets(assets_dir, files_to_upload)
        else:
            return self._upload_individual_assets(files_to_upload)
    
    def _upload_zipped_assets(self, assets_dir: Path, files_to_upload: List[Tuple[Path, Path]]) -> Dict:
        """
        Zip files by type and upload the zip files.
        
        Args:
            assets_dir: Base directory for assets
            files_to_upload: List of (file_path, rel_path) tuples
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Group files by type
            file_groups = {
                "models": [],
                "textures": [],
                "materials": [],
                "prefabs": [],
                "other": []
            }
            
            for file_path, rel_path in files_to_upload:
                path_str = str(rel_path).lower()
                
                if any(path_str.startswith(prefix) for prefix in ["models/", "model/"]):
                    file_groups["models"].append((file_path, rel_path))
                elif any(path_str.startswith(prefix) for prefix in ["textures/", "texture/"]):
                    file_groups["textures"].append((file_path, rel_path))
                elif any(path_str.startswith(prefix) for prefix in ["materials/", "material/"]):
                    file_groups["materials"].append((file_path, rel_path))
                elif any(path_str.startswith(prefix) for prefix in ["prefabs/", "prefab/"]):
                    file_groups["prefabs"].append((file_path, rel_path))
                else:
                    file_groups["other"].append((file_path, rel_path))
            
            # Create zip files
            zip_files = []
            
            for group_name, files in file_groups.items():
                if not files:
                    continue
                
                zip_path = self.output_dir / f"{group_name}.zip"
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path, rel_path in files:
                        zipf.write(file_path, str(rel_path))
                
                zip_files.append((zip_path, Path(f"{group_name}.zip")))
                logger.info(f"Created {group_name}.zip with {len(files)} files")
            
            # Upload zip files
            upload_result = self._upload_individual_assets(zip_files)
            
            # Include manifest with information about the groups
            manifest = {
                "created": time.time(),
                "server_url": self.server_url,
                "groups": {},
                "total_files": sum(len(files) for files in file_groups.values())
            }
            
            for group_name, files in file_groups.items():
                if files:
                    manifest["groups"][group_name] = {
                        "file_count": len(files),
                        "zip_name": f"{group_name}.zip",
                        "files": [str(rel_path) for _, rel_path in files]
                    }
            
            # Save manifest
            manifest_path = self.manifest_dir / "zip_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Add manifest to upload result
            upload_result["manifest"] = manifest
            
            return upload_result
        
        except Exception as e:
            logger.error(f"Error zipping and uploading assets: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _upload_individual_assets(self, files_to_upload: List[Tuple[Path, Path]]) -> Dict:
        """
        Upload individual files to the server.
        
        Args:
            files_to_upload: List of (file_path, rel_path) tuples
            
        Returns:
            Dictionary with upload results
        """
        # Call the appropriate upload method based on server type
        if self.server_type == "gcs":
            return self._upload_to_gcs(files_to_upload)
        elif self.server_type == "s3":
            return self._upload_to_s3(files_to_upload)
        elif self.server_type == "http":
            return self._upload_to_http(files_to_upload)
        else:
            return self._copy_to_local(files_to_upload)
    
    def _upload_to_gcs(self, files_to_upload: List[Tuple[Path, Path]]) -> Dict:
        """
        Upload files to Google Cloud Storage.
        
        Args:
            files_to_upload: List of (file_path, rel_path) tuples
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Check if google-cloud-storage is installed
            try:
                from google.cloud import storage
                from google.oauth2 import service_account
            except ImportError:
                logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
                return {"success": False, "error": "google-cloud-storage not installed"}
            
            # Parse GCS URL
            match = re.match(r'gs://([^/]+)(?:/(.*))?', self.server_url)
            if not match:
                logger.error(f"Invalid GCS URL: {self.server_url}")
                return {"success": False, "error": f"Invalid GCS URL: {self.server_url}"}
            
            bucket_name = match.group(1)
            base_prefix = match.group(2) or ""
            if base_prefix and not base_prefix.endswith("/"):
                base_prefix += "/"
            
            # Initialize GCS client
            if self.credentials_path and os.path.exists(self.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
                client = storage.Client(credentials=credentials)
            else:
                client = storage.Client()
            
            # Get the bucket
            try:
                bucket = client.get_bucket(bucket_name)
            except Exception as e:
                logger.error(f"Error accessing GCS bucket {bucket_name}: {str(e)}")
                return {"success": False, "error": f"Error accessing GCS bucket: {str(e)}"}
            
            # Upload files
            uploaded_files = []
            start_time = time.time()
            
            # Determine MIME types
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".obj": "text/plain",
                ".mtl": "text/plain",
                ".fbx": "application/octet-stream",
                ".mat": "application/json",
                ".json": "application/json",
                ".zip": "application/zip",
                ".unity": "application/octet-stream",
                ".prefab": "application/json"
            }
            
            for file_path, rel_path in files_to_upload:
                try:
                    # Determine content type
                    ext = file_path.suffix.lower()
                    content_type = mime_types.get(ext, "application/octet-stream")
                    
                    # Create blob name (target path in GCS)
                    blob_name = base_prefix + str(rel_path).replace("\\", "/")
                    
                    # Create the blob
                    blob = bucket.blob(blob_name)
                    
                    # Set content type
                    blob.content_type = content_type
                    
                    # Set cache control based on file type
                    if ext in (".jpg", ".jpeg", ".png"):
                        # Longer caching for images
                        blob.cache_control = "public, max-age=86400"
                    else:
                        # Shorter caching for other files
                        blob.cache_control = "public, max-age=3600"
                    
                    # Upload the file
                    blob.upload_from_filename(str(file_path))
                    
                    # Get public URL
                    public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
                    
                    uploaded_files.append({
                        "local_path": str(file_path),
                        "server_path": blob_name,
                        "public_url": public_url
                    })
                    
                    logger.info(f"Uploaded {file_path} to {blob_name}")
                
                except Exception as e:
                    logger.error(f"Error uploading {file_path}: {str(e)}")
            
            duration = time.time() - start_time
            
            # Create manifest
            manifest = {
                "created": time.time(),
                "server_type": "gcs",
                "server_url": self.server_url,
                "bucket": bucket_name,
                "base_prefix": base_prefix,
                "uploaded_files": len(uploaded_files),
                "duration_seconds": duration,
                "files": uploaded_files
            }
            
            # Save manifest
            manifest_path = self.manifest_dir / "gcs_upload_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return {
                "success": True,
                "uploaded_files": len(uploaded_files),
                "duration_seconds": duration,
                "manifest_path": str(manifest_path),
                "manifest": manifest
            }
        
        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _upload_to_s3(self, files_to_upload: List[Tuple[Path, Path]]) -> Dict:
        """
        Upload files to Amazon S3.
        
        Args:
            files_to_upload: List of (file_path, rel_path) tuples
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Check if boto3 is installed
            try:
                import boto3
                from botocore.exceptions import ClientError
            except ImportError:
                logger.error("boto3 not installed. Run: pip install boto3")
                return {"success": False, "error": "boto3 not installed"}
            
            # Parse S3 URL
            match = re.match(r's3://([^/]+)(?:/(.*))?', self.server_url)
            if not match:
                logger.error(f"Invalid S3 URL: {self.server_url}")
                return {"success": False, "error": f"Invalid S3 URL: {self.server_url}"}
            
            bucket_name = match.group(1)
            base_prefix = match.group(2) or ""
            if base_prefix and not base_prefix.endswith("/"):
                base_prefix += "/"
            
            # Initialize S3 client
            session = boto3.Session()
            s3_client = session.client('s3')
            
            # Check if bucket exists
            try:
                s3_client.head_bucket(Bucket=bucket_name)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code == '404':
                    logger.error(f"S3 bucket {bucket_name} does not exist")
                    return {"success": False, "error": f"S3 bucket {bucket_name} does not exist"}
                else:
                    logger.error(f"Error accessing S3 bucket {bucket_name}: {str(e)}")
                    return {"success": False, "error": f"Error accessing S3 bucket: {str(e)}"}
            
            # Upload files
            uploaded_files = []
            start_time = time.time()
            
            # Determine MIME types
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".obj": "text/plain",
                ".mtl": "text/plain",
                ".fbx": "application/octet-stream",
                ".mat": "application/json",
                ".json": "application/json",
                ".zip": "application/zip",
                ".unity": "application/octet-stream",
                ".prefab": "application/json"
            }
            
            for file_path, rel_path in files_to_upload:
                try:
                    # Determine content type
                    ext = file_path.suffix.lower()
                    content_type = mime_types.get(ext, "application/octet-stream")
                    
                    # Create object key (target path in S3)
                    object_key = base_prefix + str(rel_path).replace("\\", "/")
                    
                    # Prepare extra args
                    extra_args = {
                        'ContentType': content_type
                    }
                    
                    # Set cache control based on file type
                    if ext in (".jpg", ".jpeg", ".png"):
                        # Longer caching for images
                        extra_args['CacheControl'] = "public, max-age=86400"
                    else:
                        # Shorter caching for other files
                        extra_args['CacheControl'] = "public, max-age=3600"
                    
                    # Upload the file
                    s3_client.upload_file(
                        str(file_path),
                        bucket_name,
                        object_key,
                        ExtraArgs=extra_args
                    )
                    
                    # Get public URL
                    public_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
                    
                    uploaded_files.append({
                        "local_path": str(file_path),
                        "server_path": object_key,
                        "public_url": public_url
                    })
                    
                    logger.info(f"Uploaded {file_path} to {object_key}")
                
                except Exception as e:
                    logger.error(f"Error uploading {file_path}: {str(e)}")
            
            duration = time.time() - start_time
            
            # Create manifest
            manifest = {
                "created": time.time(),
                "server_type": "s3",
                "server_url": self.server_url,
                "bucket": bucket_name,
                "base_prefix": base_prefix,
                "uploaded_files": len(uploaded_files),
                "duration_seconds": duration,
                "files": uploaded_files
            }
            
            # Save manifest
            manifest_path = self.manifest_dir / "s3_upload_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return {
                "success": True,
                "uploaded_files": len(uploaded_files),
                "duration_seconds": duration,
                "manifest_path": str(manifest_path),
                "manifest": manifest
            }
        
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _upload_to_http(self, files_to_upload: List[Tuple[Path, Path]]) -> Dict:
        """
        Upload files to an HTTP server.
        
        Args:
            files_to_upload: List of (file_path, rel_path) tuples
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Check if requests is installed
            try:
                import requests
            except ImportError:
                logger.error("requests not installed. Run: pip install requests")
                return {"success": False, "error": "requests not installed"}
            
            # Ensure server URL ends with /
            server_url = self.server_url
            if not server_url.endswith("/"):
                server_url += "/"
            
            # Load credentials if provided
            auth = None
            headers = {}
            
            if self.credentials_path and os.path.exists(self.credentials_path):
                try:
                    with open(self.credentials_path, 'r') as f:
                        creds = json.load(f)
                    
                    if "username" in creds and "password" in creds:
                        auth = (creds["username"], creds["password"])
                    
                    if "token" in creds:
                        headers["Authorization"] = f"Bearer {creds['token']}"
                    
                    if "api_key" in creds:
                        headers["X-API-Key"] = creds["api_key"]
                
                except Exception as e:
                    logger.error(f"Error loading credentials: {str(e)}")
                    return {"success": False, "error": f"Error loading credentials: {str(e)}"}
            
            # Upload files
            uploaded_files = []
            start_time = time.time()
            
            # Determine MIME types
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".obj": "text/plain",
                ".mtl": "text/plain",
                ".fbx": "application/octet-stream",
                ".mat": "application/json",
                ".json": "application/json",
                ".zip": "application/zip",
                ".unity": "application/octet-stream",
                ".prefab": "application/json"
            }
            
            for file_path, rel_path in files_to_upload:
                try:
                    # Determine content type
                    ext = file_path.suffix.lower()
                    content_type = mime_types.get(ext, "application/octet-stream")
                    
                    # Create URL
                    url = server_url + str(rel_path).replace("\\", "/")
                    
                    # Set headers
                    file_headers = headers.copy()
                    file_headers["Content-Type"] = content_type
                    
                    # Set cache control based on file type
                    if ext in (".jpg", ".jpeg", ".png"):
                        # Longer caching for images
                        file_headers["Cache-Control"] = "public, max-age=86400"
                    else:
                        # Shorter caching for other files
                        file_headers["Cache-Control"] = "public, max-age=3600"
                    
                    # Upload the file
                    with open(file_path, 'rb') as f:
                        response = requests.put(url, data=f, headers=file_headers, auth=auth)
                    
                    # Check response
                    if response.status_code in (200, 201, 204):
                        uploaded_files.append({
                            "local_path": str(file_path),
                            "server_path": str(rel_path),
                            "public_url": url,
                            "status_code": response.status_code
                        })
                        
                        logger.info(f"Uploaded {file_path} to {url}")
                    else:
                        logger.warning(f"Failed to upload {file_path} to {url}: {response.status_code}")
                
                except Exception as e:
                    logger.error(f"Error uploading {file_path}: {str(e)}")
            
            duration = time.time() - start_time
            
            # Create manifest
            manifest = {
                "created": time.time(),
                "server_type": "http",
                "server_url": server_url,
                "uploaded_files": len(uploaded_files),
                "duration_seconds": duration,
                "files": uploaded_files
            }
            
            # Save manifest
            manifest_path = self.manifest_dir / "http_upload_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return {
                "success": True,
                "uploaded_files": len(uploaded_files),
                "duration_seconds": duration,
                "manifest_path": str(manifest_path),
                "manifest": manifest
            }
        
        except Exception as e:
            logger.error(f"Error uploading to HTTP server: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _copy_to_local(self, files_to_upload: List[Tuple[Path, Path]]) -> Dict:
        """
        Copy files to a local directory.
        
        Args:
            files_to_upload: List of (file_path, rel_path) tuples
            
        Returns:
            Dictionary with copy results
        """
        try:
            # Parse local path
            dest_dir = Path(self.server_url)
            
            # Create destination directory if it doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy files
            copied_files = []
            start_time = time.time()
            
            for file_path, rel_path in files_to_upload:
                try:
                    # Create destination path
                    dest_path = dest_dir / rel_path
                    
                    # Create parent directory if it doesn't exist
                    os.makedirs(dest_path.parent, exist_ok=True)
                    
                    # Copy the file
                    shutil.copy2(file_path, dest_path)
                    
                    copied_files.append({
                        "local_path": str(file_path),
                        "server_path": str(dest_path)
                    })
                    
                    logger.info(f"Copied {file_path} to {dest_path}")
                
                except Exception as e:
                    logger.error(f"Error copying {file_path}: {str(e)}")
            
            duration = time.time() - start_time
            
            # Create manifest
            manifest = {
                "created": time.time(),
                "server_type": "local",
                "server_url": str(dest_dir),
                "copied_files": len(copied_files),
                "duration_seconds": duration,
                "files": copied_files
            }
            
            # Save manifest
            manifest_path = self.manifest_dir / "local_copy_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return {
                "success": True,
                "copied_files": len(copied_files),
                "duration_seconds": duration,
                "manifest_path": str(manifest_path),
                "manifest": manifest
            }
        
        except Exception as e:
            logger.error(f"Error copying to local directory: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_unity_server_config(self, manifest_path: str = None) -> Dict:
        """
        Generate a Unity configuration file for the server.
        
        Args:
            manifest_path: Path to the upload manifest file (auto-detected if None)
            
        Returns:
            Dictionary with configuration generation results
        """
        try:
            # If manifest path not provided, try to find the latest one
            if manifest_path is None:
                manifest_files = list(self.manifest_dir.glob("*_manifest.json"))
                if not manifest_files:
                    logger.error("No upload manifest files found")
                    return {"success": False, "error": "No upload manifest files found"}
                
                # Sort by modification time (newest first)
                manifest_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                manifest_path = manifest_files[0]
            
            # Load manifest
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Create Unity configuration
            unity_config = {
                "serverType": manifest.get("server_type", self.server_type),
                "serverUrl": manifest.get("server_url", self.server_url),
                "assets": {
                    "prefabs": {},
                    "materials": {},
                    "textures": {},
                    "models": {}
                }
            }
            
            # Add server-specific configuration
            if self.server_type == "gcs":
                unity_config["googleCloud"] = {
                    "bucketName": manifest.get("bucket", ""),
                    "basePrefix": manifest.get("base_prefix", "")
                }
            elif self.server_type == "s3":
                unity_config["amazonS3"] = {
                    "bucketName": manifest.get("bucket", ""),
                    "basePrefix": manifest.get("base_prefix", ""),
                    "region": "us-east-1"  # Default region
                }
            
            # Process files from manifest
            if "files" in manifest:
                for file_info in manifest["files"]:
                    if "server_path" not in file_info:
                        continue
                    
                    server_path = file_info["server_path"]
                    public_url = file_info.get("public_url", f"{self.server_url}/{server_path}")
                    
                    # Categorize by file type
                    if server_path.endswith(".prefab"):
                        unity_config["assets"]["prefabs"][os.path.basename(server_path)] = public_url
                    elif server_path.endswith(".mat"):
                        unity_config["assets"]["materials"][os.path.basename(server_path)] = public_url
                    elif server_path.endswith((".jpg", ".jpeg", ".png")):
                        unity_config["assets"]["textures"][os.path.basename(server_path)] = public_url
                    elif server_path.endswith((".obj", ".fbx")):
                        unity_config["assets"]["models"][os.path.basename(server_path)] = public_url
            
            # Add main city prefab if available
            city_prefab = unity_config["assets"]["prefabs"].get("arcanum_city.prefab")
            if city_prefab:
                unity_config["mainCityPrefab"] = city_prefab
            
            # Save configuration
            config_path = self.output_dir / "unity_server_config.json"
            with open(config_path, 'w') as f:
                json.dump(unity_config, f, indent=2)
            
            logger.info(f"Generated Unity server configuration at {config_path}")
            
            # Generate Unity C# class
            unity_cs_path = self.output_dir / "ArcanumServerConfig.cs"
            self._generate_unity_cs_class(unity_config, unity_cs_path)
            
            return {
                "success": True,
                "config_path": str(config_path),
                "unity_cs_path": str(unity_cs_path),
                "config": unity_config
            }
        
        except Exception as e:
            logger.error(f"Error generating Unity server configuration: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _generate_unity_cs_class(self, config: Dict, output_path: Path) -> bool:
        """
        Generate a Unity C# class with async loading, caching and error handling.
        
        Args:
            config: Server configuration dictionary
            output_path: Path to save the C# class
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create C# class template
            cs_code = """using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace Arcanum
{
    /// <summary>
    /// Server configuration supporting async loading and caching
    /// </summary>
    [Serializable]
    public class ArcanumServerConfig
    {
        // Server connection configuration
        public string serverType;
        public string serverUrl;
        public string mainCityPrefab;

        // Configuration version for backward compatibility
        public string configVersion = "1.0";

        // Connection settings
        public int maxRetryAttempts = 3;
        public float connectionTimeout = 30f;
        
        /// <summary>
        /// Asset collection with efficient caching and loading capabilities
        /// </summary>
        [Serializable]
        public class AssetCollection
        {
            // Asset URL dictionaries
            public Dictionary<string, string> prefabs = new Dictionary<string, string>();
            public Dictionary<string, string> materials = new Dictionary<string, string>();
            public Dictionary<string, string> textures = new Dictionary<string, string>();
            public Dictionary<string, string> models = new Dictionary<string, string>();

            // Non-serialized runtime cache
            [NonSerialized] private Dictionary<string, UnityEngine.Object> assetCache =
                new Dictionary<string, UnityEngine.Object>();

            /// <summary>
            /// Checks if asset exists in the collection
            /// </summary>
            public bool HasAsset(string assetName, string assetType = null)
            {
                if (assetType == "prefab") return prefabs.ContainsKey(assetName);
                if (assetType == "material") return materials.ContainsKey(assetName);
                if (assetType == "texture") return textures.ContainsKey(assetName);
                if (assetType == "model") return models.ContainsKey(assetName);

                // Check all types if not specified
                return prefabs.ContainsKey(assetName) ||
                       materials.ContainsKey(assetName) ||
                       textures.ContainsKey(assetName) ||
                       models.ContainsKey(assetName);
            }

            /// <summary>
            /// Gets a cached asset or returns null
            /// </summary>
            public T GetCachedAsset<T>(string key) where T : UnityEngine.Object
            {
                if (assetCache.TryGetValue(key, out var asset) && asset is T typedAsset)
                {
                    return typedAsset;
                }
                return null;
            }

            /// <summary>
            /// Caches an asset for future use
            /// </summary>
            public void CacheAsset(string key, UnityEngine.Object asset)
            {
                assetCache[key] = asset;
            }
        }
        
        public AssetCollection assets = new AssetCollection();
        
        """
            
            # Add server-specific classes
            if self.server_type == "gcs":
                cs_code += """[Serializable]
        public class GoogleCloudConfig
        {
            public string bucketName;
            public string basePrefix;
        }
        
        public GoogleCloudConfig googleCloud = new GoogleCloudConfig();
        """
            elif self.server_type == "s3":
                cs_code += """[Serializable]
        public class AmazonS3Config
        {
            public string bucketName;
            public string basePrefix;
            public string region;
        }
        
        public AmazonS3Config amazonS3 = new AmazonS3Config();
        """
            
            # Add static instance and loading methods
            cs_code += """
        private static ArcanumServerConfig _instance;
        
        public static ArcanumServerConfig Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = LoadConfig();
                }
                return _instance;
            }
        }
        
        /// <summary>
        /// Load server configuration with fallback and error recovery
        /// </summary>
        private static ArcanumServerConfig LoadConfig()
        {
            // Try to load from Resources folder first
            TextAsset configAsset = Resources.Load<TextAsset>("ArcanumServerConfig");
            if (configAsset != null)
            {
                try
                {
                    ArcanumServerConfig config = JsonUtility.FromJson<ArcanumServerConfig>(configAsset.text);
                    Debug.Log("Loaded ArcanumServerConfig from Resources");

                    // Validate configuration
                    if (string.IsNullOrEmpty(config.serverUrl))
                    {
                        Debug.LogWarning("Loaded config has empty serverUrl - using default");
                        config.serverUrl = "Arcanum/Assets";
                    }

                    return config;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error parsing ArcanumServerConfig: {ex.Message}");
                }
            }

            // Try to load from persistent storage next
            string persistentPath = System.IO.Path.Combine(
                Application.persistentDataPath,
                "ArcanumServerConfig.json"
            );

            if (System.IO.File.Exists(persistentPath))
            {
                try
                {
                    string json = System.IO.File.ReadAllText(persistentPath);
                    ArcanumServerConfig config = JsonUtility.FromJson<ArcanumServerConfig>(json);
                    Debug.Log("Loaded ArcanumServerConfig from persistent storage");
                    return config;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error loading from persistent storage: {ex.Message}");
                }
            }

            // Fallback to default config with logging
            Debug.LogWarning("Using default ArcanumServerConfig - no valid configuration found");
            return CreateDefaultConfig();
        }
        
        private static ArcanumServerConfig CreateDefaultConfig()
        {
            ArcanumServerConfig config = new ArcanumServerConfig();
            config.serverType = "local";
            config.serverUrl = "Arcanum/Assets";
            return config;
        }
        
        /// <summary>
        /// Gets the URL for an asset with type detection and error handling
        /// </summary>
        public string GetAssetUrl(string assetName, string assetType = null)
        {
            // Use explicit asset type if provided
            if (assetType != null)
            {
                switch (assetType.ToLowerInvariant())
                {
                    case "prefab":
                        if (assets.prefabs.TryGetValue(assetName, out string prefabUrl))
                            return prefabUrl;
                        break;

                    case "material":
                        if (assets.materials.TryGetValue(assetName, out string materialUrl))
                            return materialUrl;
                        break;

                    case "texture":
                        if (assets.textures.TryGetValue(assetName, out string textureUrl))
                            return textureUrl;
                        break;

                    case "model":
                        if (assets.models.TryGetValue(assetName, out string modelUrl))
                            return modelUrl;
                        break;
                }
            }
            else
            {
                // Auto-detect asset type if not specified
                if (assets.prefabs.TryGetValue(assetName, out string prefabUrl))
                    return prefabUrl;

                if (assets.materials.TryGetValue(assetName, out string materialUrl))
                    return materialUrl;

                if (assets.textures.TryGetValue(assetName, out string textureUrl))
                    return textureUrl;

                if (assets.models.TryGetValue(assetName, out string modelUrl))
                    return modelUrl;
            }

            // Asset not found - log warning and return default path
            Debug.LogWarning($"Asset not found in configuration: {assetName} (type: {assetType ?? "auto"})");
            return $"{serverUrl}/{assetName}";
        }

        /// <summary>
        /// Asynchronously loads an asset from the server with caching and error handling
        /// </summary>
        public async Task<T> LoadAssetAsync<T>(string assetName, string assetType = null) where T : UnityEngine.Object
        {
            // Check if we already have this asset cached
            T cachedAsset = assets.GetCachedAsset<T>(assetName);
            if (cachedAsset != null)
                return cachedAsset;

            // Get the asset URL
            string url = GetAssetUrl(assetName, assetType);

            // Determine the asset type based on the generic parameter
            if (typeof(T) == typeof(Texture2D))
            {
                Texture2D texture = await LoadTextureAsync(url);
                assets.CacheAsset(assetName, texture);
                return texture as T;
            }

            throw new System.NotSupportedException($"Async loading not supported for type {typeof(T).Name}");
        }

        /// <summary>
        /// Loads a texture from a URL with retry logic and error handling
        /// </summary>
        private async Task<Texture2D> LoadTextureAsync(string url)
        {
            for (int attempt = 0; attempt < maxRetryAttempts; attempt++)
            {
                try
                {
                    using (UnityWebRequest request = UnityWebRequestTexture.GetTexture(url))
                    {
                        request.timeout = (int)connectionTimeout;
                        var operation = request.SendWebRequest();

                        while (!operation.isDone)
                            await Task.Delay(10);

                        if (request.result == UnityWebRequest.Result.Success)
                        {
                            return DownloadHandlerTexture.GetContent(request);
                        }
                        else
                        {
                            Debug.LogWarning($"Failed to load texture on attempt {attempt+1}: {request.error}");

                            // Wait before retrying (exponential backoff)
                            await Task.Delay((attempt + 1) * 500);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Exception loading texture on attempt {attempt+1}: {ex.Message}");

                    // Wait before retrying
                    await Task.Delay((attempt + 1) * 1000);
                }
            }

            // All attempts failed - return fallback texture
            Debug.LogError($"Failed to load texture after {maxRetryAttempts} attempts: {url}");
            return CreateFallbackTexture();
        }

        /// <summary>
        /// Creates a simple checkerboard pattern fallback texture
        /// </summary>
        private Texture2D CreateFallbackTexture()
        {
            Texture2D fallbackTexture = new Texture2D(256, 256);
            Color[] pixels = new Color[256 * 256];

            // Create simple checkerboard pattern
            for (int y = 0; y < 256; y++)
            {
                for (int x = 0; x < 256; x++)
                {
                    bool isEvenX = (x / 32) % 2 == 0;
                    bool isEvenY = (y / 32) % 2 == 0;

                    if (isEvenX == isEvenY)
                        pixels[y * 256 + x] = new Color(0.8f, 0.3f, 0.3f); // Error color (red-ish)
                    else
                        pixels[y * 256 + x] = new Color(0.3f, 0.3f, 0.3f); // Dark gray
                }
            }

            fallbackTexture.SetPixels(pixels);
            fallbackTexture.Apply();
            return fallbackTexture;
        }
    }
}
"""
            
            # Write C# class to file
            with open(output_path, 'w') as f:
                f.write(cs_code)
            
            logger.info(f"Generated Unity C# class at {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error generating Unity C# class: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tile Server Integration")
    parser.add_argument("--output", default="./arcanum_output", help="Output directory")
    parser.add_argument("--server", required=True, help="Server URL (gs://, s3://, http://, or local path)")
    parser.add_argument("--credentials", help="Path to credentials file")
    parser.add_argument("--server-type", choices=["gcs", "s3", "http", "local"], help="Server type (auto-detected if not specified)")
    parser.add_argument("--zip", action="store_true", help="Zip files by type before uploading")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize tile server integration
    tile_server = TileServerIntegration(
        args.output,
        args.server,
        args.credentials,
        args.server_type
    )
    
    # Upload assets
    upload_result = tile_server.upload_assets(args.output, args.zip)
    
    if upload_result["success"]:
        print(f"Uploaded {upload_result.get('uploaded_files', 0)} files to server")
        
        # Generate Unity configuration
        config_result = tile_server.generate_unity_server_config()
        
        if config_result["success"]:
            print(f"Generated Unity server configuration at {config_result.get('config_path')}")
            print(f"Generated Unity C# class at {config_result.get('unity_cs_path')}")
        else:
            print(f"Failed to generate Unity server configuration: {config_result.get('error')}")
    else:
        print(f"Failed to upload assets: {upload_result.get('error')}")