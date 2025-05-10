#!/usr/bin/env python3
"""
Integration tests for the Arcanum storage module.
"""

import os
import sys
import pytest
import shutil
import tempfile
import json
from unittest.mock import patch, MagicMock

# Import the module to test
try:
    from modules.storage.storage import ArcanumStorageManager
    from modules.storage.storage_integration import ArcanumStorageIntegration
    from modules.storage import upload_image, upload_dir, upload
except ImportError:
    # Skip tests if module not available
    pytestmark = pytest.mark.skip(reason="Storage module not available")


@pytest.mark.integration
class TestArcanumStorageManagerIntegration:
    """Integration tests for the ArcanumStorageManager class."""

    def test_process_tile_with_file_system(self, temp_dir, sample_image_path):
        """Test processing a tile using the file system."""
        # Create a storage manager
        manager = ArcanumStorageManager(local_root_dir=temp_dir)
        
        # Create a simple transform function
        def transform_function(src_path, dst_path):
            shutil.copy(src_path, dst_path)
            return dst_path
        
        # Process the tile
        result = manager.process_tile(
            source_path=sample_image_path,
            tile_type="jpg",
            x=10,
            y=20,
            z=0,
            tileset_name="test",
            transform_function=transform_function
        )
        
        # Check the result
        assert result["success"] is True
        assert "processed_path" in result
        assert os.path.exists(result["processed_path"])
        assert os.path.getsize(result["processed_path"]) > 0
    
    def test_process_image_batch_with_file_system(self, temp_dir):
        """Test processing a batch of images using the file system."""
        # Create a storage manager
        manager = ArcanumStorageManager(local_root_dir=temp_dir)
        
        # Create a simple transform function
        def transform_function(src_path, dst_path):
            shutil.copy(src_path, dst_path)
            return dst_path
        
        # Create source directory with sample images
        source_dir = os.path.join(temp_dir, "source")
        os.makedirs(source_dir, exist_ok=True)
        
        # Create some sample images
        sample_images = []
        for i in range(3):
            img_path = os.path.join(source_dir, f"image_{i}.jpg")
            with open(img_path, "w") as f:
                f.write(f"test_image_{i}")
            sample_images.append(img_path)
        
        # Process the batch
        results = manager.process_image_batch(
            source_dir=source_dir,
            target_tileset="test",
            transform_function=transform_function,
            file_pattern="*.jpg",
            z_level=0
        )
        
        # Check the results
        assert len(results) == 3
        for result in results:
            assert result["success"] is True
            assert "processed_path" in result
            assert os.path.exists(result["processed_path"])
            assert os.path.getsize(result["processed_path"]) > 0
    
    def test_manage_cache_size(self, temp_dir):
        """Test managing cache size."""
        # Create a storage manager with a small cache size
        manager = ArcanumStorageManager(
            local_root_dir=temp_dir,
            max_local_cache_size_gb=0.0001  # Very small, ~100KB
        )
        
        # Create cache directory with some files
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create some large cache files (at least 10KB each)
        for i in range(20):
            file_path = os.path.join(cache_dir, f"cache_file_{i}.dat")
            with open(file_path, "wb") as f:
                f.write(b"0" * 10240)  # 10KB
        
        # Run cache management
        result = manager._manage_cache_size()
        
        # Check that cache is now smaller
        assert result["bytes_removed"] > 0
        assert result["files_removed"] > 0
        assert os.path.exists(cache_dir)  # Directory still exists
        
        # Check current cache size
        current_size = 0
        for file_name in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, file_name)
            if os.path.isfile(file_path):
                current_size += os.path.getsize(file_path)
        
        # Cache should be smaller than our target
        assert current_size < 0.0001 * 1024 * 1024 * 1024


@pytest.mark.integration
@pytest.mark.skip(reason="Requires ComfyUI which may not be available")
class TestArcanumStorageIntegrationWithFileSystem:
    """Integration tests for the ArcanumStorageIntegration class with the file system."""

    def test_transform_and_upload_image(self, temp_dir, sample_image_path):
        """Test transforming and uploading an image."""
        # Mock the style transformer
        with patch("modules.storage.storage_integration.ArcanumStyleTransformer") as mock_transformer:
            # Set up the mock
            instance = mock_transformer.return_value
            instance.transform_image.side_effect = lambda image_path, output_path, **kwargs: shutil.copy(image_path, output_path) or output_path
            
            # Create integration
            integration = ArcanumStorageIntegration(local_root_dir=temp_dir)
            
            # Transform and upload
            result = integration.transform_and_upload_image(
                image_path=sample_image_path,
                tileset_name="test"
            )
            
            # Check result
            assert result["success"] is True
            assert "processed_path" in result
            assert os.path.exists(result["processed_path"])
            assert os.path.getsize(result["processed_path"]) > 0
    
    def test_transform_and_upload_directory(self, temp_dir):
        """Test transforming and uploading a directory."""
        # Mock the style transformer
        with patch("modules.storage.storage_integration.ArcanumStyleTransformer") as mock_transformer:
            # Set up the mock
            instance = mock_transformer.return_value
            instance.transform_image.side_effect = lambda image_path, output_path, **kwargs: shutil.copy(image_path, output_path) or output_path
            instance.get_material_type.return_value = "general"
            
            # Create source directory with sample images
            source_dir = os.path.join(temp_dir, "source")
            os.makedirs(source_dir, exist_ok=True)
            
            # Create some sample images
            sample_images = []
            for i in range(3):
                img_path = os.path.join(source_dir, f"image_{i}.jpg")
                with open(img_path, "w") as f:
                    f.write(f"test_image_{i}")
                sample_images.append(img_path)
            
            # Create integration
            integration = ArcanumStorageIntegration(local_root_dir=temp_dir)
            
            # Transform and upload
            results = integration.transform_and_upload_directory(
                input_dir=source_dir,
                tileset_name="test"
            )
            
            # Check results
            assert len(results) == 3
            for result in results:
                assert result["success"] is True
                assert "processed_path" in result
                assert os.path.exists(result["processed_path"])
                assert os.path.getsize(result["processed_path"]) > 0


@pytest.mark.integration
class TestModuleFunctionsWithFileSystem:
    """Integration tests for the module-level functions with the file system."""

    def test_upload_image_filesystem(self, temp_dir, sample_image_path):
        """Test uploading an image with file system."""
        # Mock the style transformer
        with patch("modules.storage.ArcanumStyleTransformer") as mock_transformer:
            # Set up the mock
            instance = mock_transformer.return_value
            instance.transform_image.side_effect = lambda image_path, output_path, **kwargs: shutil.copy(image_path, output_path) or output_path
            
            # Patch the class to use our temp dir
            with patch("modules.storage.ArcanumStorageIntegration") as mock_class:
                mock_class.return_value.storage_manager.local_root = temp_dir
                mock_class.return_value.transform_and_upload_image.side_effect = lambda image_path, **kwargs: {
                    "success": True,
                    "processed_path": os.path.join(temp_dir, "processed.jpg"),
                    "local_path": os.path.join(temp_dir, "local.jpg")
                }
                
                # Create the destination file
                with open(os.path.join(temp_dir, "processed.jpg"), "w") as f:
                    f.write("test")
                
                # Call function
                result = upload_image(sample_image_path, output_dir=temp_dir)
                
                # Check result
                assert result["success"] is True
                assert "processed_path" in result
    
    def test_upload_dir_filesystem(self, temp_dir):
        """Test uploading a directory with file system."""
        # Mock the style transformer
        with patch("modules.storage.ArcanumStyleTransformer") as mock_transformer:
            # Set up the mock
            instance = mock_transformer.return_value
            instance.transform_image.side_effect = lambda image_path, output_path, **kwargs: shutil.copy(image_path, output_path) or output_path
            
            # Create source directory with sample images
            source_dir = os.path.join(temp_dir, "source")
            os.makedirs(source_dir, exist_ok=True)
            
            # Create some sample images
            for i in range(3):
                img_path = os.path.join(source_dir, f"image_{i}.jpg")
                with open(img_path, "w") as f:
                    f.write(f"test_image_{i}")
            
            # Patch the class to use our temp dir
            with patch("modules.storage.ArcanumStorageIntegration") as mock_class:
                mock_class.return_value.storage_manager.local_root = temp_dir
                mock_class.return_value.transform_and_upload_directory.return_value = [
                    {
                        "success": True,
                        "processed_path": os.path.join(temp_dir, f"processed_{i}.jpg"),
                        "local_path": os.path.join(temp_dir, f"local_{i}.jpg")
                    }
                    for i in range(3)
                ]
                
                # Create the destination files
                for i in range(3):
                    with open(os.path.join(temp_dir, f"processed_{i}.jpg"), "w") as f:
                        f.write(f"test_processed_{i}")
                
                # Call function
                result = upload_dir(source_dir, output_dir=temp_dir)
                
                # Check result
                assert len(result) == 3
                for r in result:
                    assert r["success"] is True
                    assert "processed_path" in r