#!/usr/bin/env python3
"""
Unit tests for the Arcanum storage module.
"""

import os
import sys
import pytest
import shutil
import tempfile
import json
from unittest.mock import patch, MagicMock, call

# Import the module to test
try:
    from modules.storage.storage import ArcanumStorageManager
    from modules.storage.storage_integration import ArcanumStorageIntegration
    from modules.storage import upload_image, upload_dir, upload
except ImportError:
    # Skip tests if module not available
    pytestmark = pytest.mark.skip(reason="Storage module not available")


class TestArcanumStorageManager:
    """Test the ArcanumStorageManager class."""

    def test_init(self, temp_dir):
        """Test initialization with default parameters."""
        manager = ArcanumStorageManager(local_root_dir=temp_dir)
        assert manager.local_root == temp_dir
        assert manager.gcs_bucket_name == "arcanum-maps"
        assert manager.cdn_url == "https://arcanum.fortunestold.co"
        assert manager.cleanup_originals is True
        assert manager.max_local_cache_size_gb == 10.0

    def test_init_with_params(self, temp_dir):
        """Test initialization with custom parameters."""
        manager = ArcanumStorageManager(
            local_root_dir=temp_dir,
            gcs_bucket_name="test-bucket",
            cdn_url="https://test.example.com",
            cleanup_originals=False,
            max_local_cache_size_gb=5.0
        )
        assert manager.local_root == temp_dir
        assert manager.gcs_bucket_name == "test-bucket"
        assert manager.cdn_url == "https://test.example.com"
        assert manager.cleanup_originals is False
        assert manager.max_local_cache_size_gb == 5.0

    def test_create_directory_structure(self, temp_dir):
        """Test creation of directory structure."""
        manager = ArcanumStorageManager(local_root_dir=temp_dir)
        
        # Check if directories were created
        assert os.path.isdir(os.path.join(temp_dir, "raw_data"))
        assert os.path.isdir(os.path.join(temp_dir, "processed"))
        assert os.path.isdir(os.path.join(temp_dir, "temp"))
        assert os.path.isdir(os.path.join(temp_dir, "cache"))

    @pytest.mark.parametrize("extension,expected", [
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
        (".png", "image/png"),
        (".gif", "image/gif"),
        (".svg", "image/svg+xml"),
        (".json", "application/json"),
        (".html", "text/html"),
        (".css", "text/css"),
        (".js", "application/javascript"),
        (".unknown", "application/octet-stream"),
    ])
    def test_get_content_type(self, temp_dir, extension, expected):
        """Test getting content type from file extension."""
        manager = ArcanumStorageManager(local_root_dir=temp_dir)
        assert manager._get_content_type(extension) == expected

    def test_extract_coordinates_from_filename(self, temp_dir):
        """Test extracting coordinates from filename."""
        manager = ArcanumStorageManager(local_root_dir=temp_dir)
        
        # Test with standard format
        assert manager._extract_coordinates_from_filename("tile_10_20.jpg", 0) == (10, 20)
        
        # Test with z value
        assert manager._extract_coordinates_from_filename("tile_10_20_z5.jpg", 5) == (10, 20)
        
        # Test with different separator
        assert manager._extract_coordinates_from_filename("tile-10-20.jpg", 0) == (10, 20)
        
        # Test with non-standard format, should return default
        assert manager._extract_coordinates_from_filename("image.jpg", 0) == (0, 0)

    @patch("modules.storage.storage.ArcanumStorageManager.upload_to_gcs")
    def test_process_tile(self, mock_upload, temp_dir, sample_image_path):
        """Test processing a tile."""
        manager = ArcanumStorageManager(local_root_dir=temp_dir)
        
        # Create a mock transform function
        def mock_transform(src_path, dst_path):
            shutil.copy(src_path, dst_path)
            return dst_path
        
        # Mock GCS upload
        mock_upload.return_value = "https://test.example.com/test.jpg"
        
        # Call process_tile
        result = manager.process_tile(
            source_path=sample_image_path,
            tile_type="jpg",
            x=10,
            y=20,
            z=0,
            tileset_name="test",
            transform_function=mock_transform
        )
        
        # Check result
        assert result["success"] is True
        assert "processed_path" in result
        assert "gcs_path" in result
        assert "cdn_url" in result
        assert result["cdn_url"] == "https://test.example.com/test.jpg"
        
        # Check that upload_to_gcs was called
        mock_upload.assert_called_once()

    @patch("modules.storage.storage.ArcanumStorageManager.upload_to_gcs")
    def test_process_tile_error(self, mock_upload, temp_dir):
        """Test processing a tile with an error."""
        manager = ArcanumStorageManager(local_root_dir=temp_dir)
        
        # Create a mock transform function that raises an exception
        def mock_transform(src_path, dst_path):
            raise Exception("Test error")
        
        # Call process_tile with a non-existent file
        result = manager.process_tile(
            source_path="/non/existent/file.jpg",
            tile_type="jpg",
            x=10,
            y=20,
            z=0,
            tileset_name="test",
            transform_function=mock_transform
        )
        
        # Check result
        assert result["success"] is False
        assert "error" in result
        
        # Check that upload_to_gcs was not called
        mock_upload.assert_not_called()


class TestArcanumStorageIntegration:
    """Test the ArcanumStorageIntegration class."""

    @pytest.mark.skip(reason="Requires ComfyUI which may not be available")
    def test_init(self, temp_dir):
        """Test initialization with default parameters."""
        integration = ArcanumStorageIntegration(local_root_dir=temp_dir)
        assert integration.storage_manager.local_root == temp_dir
        assert integration.storage_manager.gcs_bucket_name == "arcanum-maps"
        assert integration.storage_manager.cdn_url == "https://arcanum.fortunestold.co"
        assert integration.storage_manager.cleanup_originals is True

    @pytest.mark.skip(reason="Requires ComfyUI which may not be available")
    def test_init_with_params(self, temp_dir):
        """Test initialization with custom parameters."""
        integration = ArcanumStorageIntegration(
            comfyui_path=None,
            local_root_dir=temp_dir,
            gcs_bucket_name="test-bucket",
            cdn_url="https://test.example.com",
            cleanup_originals=False,
            max_local_cache_size_gb=5.0
        )
        assert integration.storage_manager.local_root == temp_dir
        assert integration.storage_manager.gcs_bucket_name == "test-bucket"
        assert integration.storage_manager.cdn_url == "https://test.example.com"
        assert integration.storage_manager.cleanup_originals is False
        assert integration.storage_manager.max_local_cache_size_gb == 5.0

    @patch("modules.storage.storage_integration.ArcanumStorageIntegration.transform_and_upload_image")
    def test_transform_satellite_images(self, mock_transform, temp_dir):
        """Test transforming satellite images."""
        with patch("modules.storage.storage_integration.ArcanumStorageManager"):
            with patch("modules.storage.storage_integration.ArcanumStyleTransformer"):
                integration = ArcanumStorageIntegration(local_root_dir=temp_dir)
                
                # Mock the transform method
                mock_transform.return_value = {"success": True}
                
                # Create a test directory with a test image
                test_dir = os.path.join(temp_dir, "test_satellite")
                os.makedirs(test_dir, exist_ok=True)
                with open(os.path.join(test_dir, "test.jpg"), "w") as f:
                    f.write("test")
                
                # Call transform_satellite_images
                result = integration.transform_satellite_images(test_dir)
                
                # Check that transform_and_upload_directory was called
                assert mock_transform.called


class TestModuleFunctions:
    """Test the module-level functions."""

    @patch("modules.storage.ArcanumStorageIntegration")
    def test_upload_image(self, mock_integration, temp_dir, sample_image_path):
        """Test the upload_image function."""
        # Set up the mock
        mock_instance = mock_integration.return_value
        mock_instance.transform_and_upload_image.return_value = {"success": True}
        
        # Call the function
        result = upload_image(sample_image_path)
        
        # Check the result
        assert mock_integration.called
        assert mock_instance.transform_and_upload_image.called
        assert mock_instance.transform_and_upload_image.call_args[0][0] == sample_image_path

    @patch("modules.storage.ArcanumStorageIntegration")
    def test_upload_dir(self, mock_integration, temp_dir):
        """Test the upload_dir function."""
        # Set up the mock
        mock_instance = mock_integration.return_value
        mock_instance.transform_and_upload_directory.return_value = [{"success": True}]
        
        # Call the function
        result = upload_dir(temp_dir)
        
        # Check the result
        assert mock_integration.called
        assert mock_instance.transform_and_upload_directory.called
        assert mock_instance.transform_and_upload_directory.call_args[0][0] == temp_dir

    @patch("modules.storage.uploader.transform_and_upload")
    def test_upload(self, mock_transform, temp_dir):
        """Test the upload function."""
        # Set up the mock
        mock_transform.return_value = {"success": True}
        
        # Call the function
        result = upload(temp_dir)
        
        # Check the result
        assert mock_transform.called
        assert mock_transform.call_args[0][0] == temp_dir