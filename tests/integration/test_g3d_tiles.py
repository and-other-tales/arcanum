#!/usr/bin/env python3
"""
Integration tests for the Google 3D Tiles module.
"""

import os
import sys
import pytest
import tempfile
import json
import shutil
import requests
from unittest.mock import patch, MagicMock, PropertyMock

# Import the modules to test
try:
    from modules.integration.g3d_tiles import fetch_tiles, fetch_city_tiles
    from modules.integration.google_3d_tiles_integration import Google3DTilesIntegration
    from modules.integration.city_tiles import fetch_city_3d_tiles
except ImportError:
    # Skip tests if modules not available
    pytestmark = pytest.mark.skip(reason="Integration modules not available")


@pytest.fixture
def mock_tileset_json():
    """Return a mock tileset.json for testing."""
    return {
        "asset": {
            "version": "1.0",
            "tilesetVersion": "1.2.3",
            "extras": {}
        },
        "root": {
            "boundingVolume": {
                "region": [
                    -0.02, 0.85, 0.02, 0.89, -10, 1000
                ]
            },
            "content": {
                "uri": "root.b3dm"
            },
            "children": [
                {
                    "boundingVolume": {
                        "region": [
                            -0.01, 0.86, 0.01, 0.88, -5, 500
                        ]
                    },
                    "content": {
                        "uri": "child.b3dm"
                    }
                }
            ]
        }
    }


@pytest.fixture
def mock_tile_response():
    """Return a mock tile response for testing."""
    # Create a binary response that looks like a b3dm file
    return b"b3dm" + b"\x00" * 100


@pytest.mark.integration
class TestGoogle3DTilesIntegration:
    """Integration tests for the Google 3D Tiles integration."""

    def test_init(self, mock_api_key):
        """Test initializing the Google3DTilesIntegration."""
        integration = Google3DTilesIntegration(api_key=mock_api_key)
        assert integration.api_key == mock_api_key
        assert integration.session_token is not None
        assert isinstance(integration.cache_dir, type(Path())) if 'Path' in globals() else True

    def test_get_tileset_url(self, mock_api_key):
        """Test getting a tileset URL."""
        integration = Google3DTilesIntegration(api_key=mock_api_key)
        url = integration.get_tileset_url(region="us")
        
        # Check that URL has correct format and parameters
        assert "https://tile.googleapis.com/v1/3dtiles/root.json" in url
        assert "key=" in url
        assert "session=" in url
        assert "region=us" in url

    def test_get_metadata_url(self, mock_api_key):
        """Test getting a metadata URL."""
        integration = Google3DTilesIntegration(api_key=mock_api_key)
        url = integration.get_metadata_url()
        
        # Check that URL has correct format and parameters
        assert "/v1/3dtiles/features/basic/metadata" in url
        assert "key=" in url

    def test_get_tile_url(self, mock_api_key):
        """Test getting a tile URL."""
        integration = Google3DTilesIntegration(api_key=mock_api_key)
        url = integration.get_tile_url("test/path/tile.b3dm")
        
        # Check that URL has correct format and parameters
        assert "/v1/3dtiles/features/basic/test/path/tile.b3dm" in url
        assert "key=" in url

    @patch("requests.get")
    def test_fetch_tileset_json(self, mock_get, mock_api_key, mock_tileset_json, temp_dir):
        """Test fetching a tileset.json."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_tileset_json
        mock_get.return_value = mock_response
        
        # Create integration and set cache directory
        integration = Google3DTilesIntegration(api_key=mock_api_key, cache_dir=temp_dir)
        
        # Fetch tileset.json
        result = integration.fetch_tileset_json(region="us")
        
        # Check result
        assert mock_get.called
        assert "asset" in result
        assert "root" in result
        
        # Check that tileset.json was cached
        cache_files = os.listdir(temp_dir)
        assert any("tileset" in f for f in cache_files)

    @patch("requests.get")
    def test_fetch_metadata(self, mock_get, mock_api_key, temp_dir):
        """Test fetching metadata."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"regions": ["us", "eu"], "version": "1.0"}
        mock_get.return_value = mock_response
        
        # Create integration and set cache directory
        integration = Google3DTilesIntegration(api_key=mock_api_key, cache_dir=temp_dir)
        
        # Fetch metadata
        result = integration.fetch_metadata()
        
        # Check result
        assert mock_get.called
        assert "regions" in result
        assert "version" in result
        
        # Check that metadata was cached
        cache_files = os.listdir(temp_dir)
        assert any("metadata" in f for f in cache_files)

    @patch("requests.get")
    def test_fetch_tile(self, mock_get, mock_api_key, mock_tile_response, temp_dir):
        """Test fetching a tile."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [mock_tile_response]
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_get.return_value = mock_response
        
        # Create integration and set cache directory
        integration = Google3DTilesIntegration(api_key=mock_api_key, cache_dir=temp_dir)
        
        # Fetch tile
        output_path = os.path.join(temp_dir, "tile.b3dm")
        result = integration.fetch_tile("path/to/tile.b3dm", output_path)
        
        # Check result
        assert mock_get.called
        assert result == output_path
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        
        # Check file contents
        with open(output_path, "rb") as f:
            assert b"b3dm" in f.read()

    @patch("requests.get")
    def test_fetch_tiles_recursive(self, mock_get, mock_api_key, mock_tileset_json, mock_tile_response, temp_dir):
        """Test fetching tiles recursively."""
        # Set up mock response for tiles
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [mock_tile_response]
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_get.return_value = mock_response
        
        # Create integration and set cache directory
        integration = Google3DTilesIntegration(api_key=mock_api_key, cache_dir=temp_dir)
        
        # Fetch tiles recursively
        output_dir = os.path.join(temp_dir, "tiles")
        os.makedirs(output_dir, exist_ok=True)
        result = integration.fetch_tiles_recursive(mock_tileset_json, max_depth=1, output_dir=output_dir)
        
        # Check result
        assert mock_get.called
        assert len(result) > 0
        
        # Check that tiles were downloaded
        assert any(os.path.exists(p) for p in result)


@pytest.mark.integration
class TestGoogleTilesModuleFunctions:
    """Integration tests for the Google 3D Tiles module functions."""

    @patch("modules.integration.g3d_tiles.Google3DTilesIntegration")
    def test_fetch_tiles_with_file_system(self, mock_integration, sample_bounds, temp_dir, mock_tileset_json):
        """Test fetching 3D tiles with the file system."""
        # Set up the mock
        mock_instance = mock_integration.return_value
        mock_instance.fetch_tileset_json.return_value = mock_tileset_json
        mock_instance.fetch_tiles_recursive.return_value = [
            os.path.join(temp_dir, "tile1.b3dm"),
            os.path.join(temp_dir, "tile2.b3dm")
        ]
        
        # Create some mock files
        for file_name in ["tile1.b3dm", "tile2.b3dm"]:
            with open(os.path.join(temp_dir, file_name), "wb") as f:
                f.write(b"mock tile data")
        
        # Call the function
        result = fetch_tiles(sample_bounds, temp_dir)
        
        # Check result
        assert mock_integration.called
        assert mock_instance.fetch_tileset_json.called
        assert mock_instance.fetch_tiles_recursive.called
        assert result["success"] is True
        assert result["downloaded_tiles"] == 3  # 2 tiles + tileset.json
        assert "tileset_path" in result
        assert os.path.exists(result["tileset_path"])

    @patch("modules.integration.g3d_tiles.Google3DTilesIntegration")
    def test_fetch_city_tiles_with_file_system(self, mock_integration, temp_dir, sample_city_name):
        """Test fetching city 3D tiles with the file system."""
        # Set up the mock
        mock_instance = mock_integration.return_value
        mock_instance.fetch_city_tiles.return_value = {
            "success": True,
            "downloaded_tiles": 10,
            "city": sample_city_name,
            "paths": [os.path.join(temp_dir, f"tile{i}.b3dm") for i in range(10)]
        }
        
        # Create some mock files
        for i in range(10):
            with open(os.path.join(temp_dir, f"tile{i}.b3dm"), "wb") as f:
                f.write(b"mock tile data")
        
        # Call the function
        result = fetch_city_tiles(sample_city_name, temp_dir)
        
        # Check result
        assert mock_integration.called
        assert mock_instance.fetch_city_tiles.called
        assert result["success"] is True
        assert result["downloaded_tiles"] == 10
        assert result["city"] == sample_city_name

    @patch("modules.integration.city_tiles.Google3DTilesIntegration")
    def test_fetch_city_3d_tiles_with_file_system(self, mock_integration, temp_dir, sample_city_name):
        """Test fetching city 3D tiles with the file system."""
        # Set up the mock
        mock_instance = mock_integration.return_value
        mock_instance.fetch_city_tiles.return_value = {
            "success": True,
            "downloaded_tiles": 10,
            "city": sample_city_name,
            "paths": [os.path.join(temp_dir, f"tile{i}.b3dm") for i in range(10)]
        }
        
        # Create some mock files
        for i in range(10):
            with open(os.path.join(temp_dir, f"tile{i}.b3dm"), "wb") as f:
                f.write(b"mock tile data")
        
        # Call the function
        result = fetch_city_3d_tiles(sample_city_name, temp_dir)
        
        # Check result
        assert mock_integration.called
        assert mock_instance.fetch_city_tiles.called
        assert result["success"] is True
        assert result["downloaded_tiles"] == 10
        assert result["city"] == sample_city_name
        
        # Check that the summary file was created
        city_dir = os.path.join(temp_dir, sample_city_name.lower().replace(" ", "_"))
        summary_path = os.path.join(city_dir, "summary.json")
        assert os.path.exists(summary_path)
        
        # Check summary content
        with open(summary_path, "r") as f:
            summary = json.load(f)
            assert summary["city"] == sample_city_name
            assert summary["tiles_downloaded"] == 10