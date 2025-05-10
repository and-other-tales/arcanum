#!/usr/bin/env python3
"""
Integration tests for the Arcanum OSM module.
"""

import os
import sys
import pytest
import json
import shutil
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

# Try to import required dependencies
try:
    import geopandas as gpd
    import osmnx as ox
    import pyproj
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

# Import the modules to test
try:
    from modules.osm.bbox_downloader import download_osm_data
    from modules.osm.grid_downloader import (
        download_osm_grid,
        download_osm_for_cell,
        merge_grid_data
    )
    from modules.osm.config import configure_osmnx, configure_for_grid_downloads
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# Skip all tests if dependencies or modules are missing
if not MODULES_AVAILABLE or not HAS_DEPENDENCIES:
    pytestmark = pytest.mark.skip(reason="OSM module or dependencies not available")


@pytest.fixture
def test_osm_buildings(test_data_dir):
    """Return the path to the test OSM buildings data."""
    return os.path.join(test_data_dir, "osm", "test_buildings.json")


@pytest.fixture
def mock_osm_response():
    """Create a mock OSM response GeoDataFrame."""
    if not HAS_DEPENDENCIES:
        pytest.skip("GeoDataFrame dependencies not available")
    
    # Create a GeoDataFrame from the test data
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        import pandas as pd
        
        # Create simple buildings data
        buildings_data = {
            'geometry': [
                Polygon([(0, 0), (10, 0), (10, 15), (0, 15), (0, 0)]),
                Polygon([(15, 5), (25, 5), (25, 20), (15, 20), (15, 5)])
            ],
            'building': ['residential', 'commercial'],
            'height': [25.0, 40.0],
            'levels': [3, 8]
        }
        
        return gpd.GeoDataFrame(buildings_data, crs="EPSG:4326")
    except Exception:
        pytest.skip("Failed to create mock GeoDataFrame")


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="OSMnx and GeoDataFrames required")
class TestOsmIntegration:
    """Integration tests for the OSM module."""

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="OSMnx required")
    @patch('modules.osm.bbox_downloader.download_buildings_and_roads')
    def test_download_osm_data_file_creation(self, mock_download, temp_dir, mock_osm_response):
        """Test that download_osm_data creates the expected output files."""
        # Mock the download response
        mock_roads = gpd.GeoDataFrame({'geometry': mock_osm_response['geometry']})
        mock_download.return_value = (mock_osm_response, mock_roads)
        
        # Test bounds
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        # Call the function
        result = download_osm_data(bounds, temp_dir)
        
        # Check if the function returned success
        assert result["success"] is True
        
        # Check if the expected files were created
        vector_dir = os.path.join(temp_dir, "vector")
        assert os.path.exists(vector_dir)
        
        # Verify the buildings file was created
        buildings_path = os.path.join(vector_dir, "osm_arcanum.gpkg")
        assert os.path.exists(buildings_path)
        assert result["buildings_path"] == buildings_path
        
        # Verify the roads file was created
        roads_path = os.path.join(vector_dir, "osm_roads.gpkg")
        assert os.path.exists(roads_path)
        assert result["roads_path"] == roads_path
        
        # Verify file contents by trying to read them
        try:
            buildings_gdf = gpd.read_file(buildings_path)
            roads_gdf = gpd.read_file(roads_path)
            assert len(buildings_gdf) == len(mock_osm_response)
            assert len(roads_gdf) == len(mock_roads)
        except Exception as e:
            pytest.fail(f"Failed to read GeoPackage files: {str(e)}")

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="OSMnx required")
    @patch('modules.osm.grid_downloader.download_osm_for_cell')
    def test_download_osm_grid_integration(self, mock_download_cell, temp_dir):
        """Test that the grid downloader creates proper directory structure and files."""
        # Mock cell download results
        mock_download_cell.side_effect = [
            {
                "cell_name": "cell_0_0",
                "success": True,
                "buildings_path": os.path.join(temp_dir, "vector/grid/cell_0_0/buildings.gpkg"),
                "roads_path": os.path.join(temp_dir, "vector/grid/cell_0_0/roads.gpkg")
            },
            {
                "cell_name": "cell_0_1",
                "success": True,
                "buildings_path": os.path.join(temp_dir, "vector/grid/cell_0_1/buildings.gpkg"),
                "roads_path": os.path.join(temp_dir, "vector/grid/cell_0_1/roads.gpkg")
            }
        ]
        
        # Test bounds
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        # Create test directory structure
        os.makedirs(os.path.join(temp_dir, "vector", "grid", "cell_0_0"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "vector", "grid", "cell_0_1"), exist_ok=True)
        
        # Call the function with small cell size to ensure multiple cells
        result = download_osm_grid(bounds, temp_dir, cell_size_meters=100)
        
        # Verify results
        assert result["success_count"] > 0
        assert len(result["cell_results"]) > 0
        
        # Check for metadata file
        metadata_path = os.path.join(temp_dir, "vector", "grid_download_metadata.json")
        assert os.path.exists(metadata_path)
        
        # Load and verify metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["bounds"] == bounds
        assert metadata["cell_size_meters"] == 100
        assert "duration_seconds" in metadata
        assert "completed_at" in metadata

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="GeoDataFrame dependencies required")
    def test_merge_grid_data_integration(self, temp_dir, mock_osm_response):
        """Test merging grid cell data with actual file operations."""
        # Create mock grid structure
        grid_dir = os.path.join(temp_dir, "vector", "grid")
        os.makedirs(os.path.join(grid_dir, "cell_0_0"), exist_ok=True)
        os.makedirs(os.path.join(grid_dir, "cell_1_0"), exist_ok=True)
        
        # Save sample GeoDataFrames to the grid structure
        buildings_path_1 = os.path.join(grid_dir, "cell_0_0", "buildings.gpkg")
        buildings_path_2 = os.path.join(grid_dir, "cell_1_0", "buildings.gpkg")
        roads_path_1 = os.path.join(grid_dir, "cell_0_0", "roads.gpkg")
        roads_path_2 = os.path.join(grid_dir, "cell_1_0", "roads.gpkg")
        
        # Create a modified copy for the second cell
        mock_osm_response_2 = mock_osm_response.copy()
        
        # Save to GeoPackage files
        mock_osm_response.to_file(buildings_path_1, layer='buildings', driver='GPKG')
        mock_osm_response_2.to_file(buildings_path_2, layer='buildings', driver='GPKG')
        
        # Create simple road GeoDataFrames and save them
        roads_gdf = gpd.GeoDataFrame({'geometry': mock_osm_response['geometry']})
        roads_gdf.to_file(roads_path_1, layer='roads', driver='GPKG')
        roads_gdf.to_file(roads_path_2, layer='roads', driver='GPKG')
        
        # Call the merge function
        result = merge_grid_data(temp_dir)
        
        # Verify function succeeded
        assert result["success"] is True
        
        # Check merged files exist
        merged_buildings_path = os.path.join(temp_dir, "vector", "osm_all_buildings.gpkg")
        merged_roads_path = os.path.join(temp_dir, "vector", "osm_all_roads.gpkg")
        assert os.path.exists(merged_buildings_path)
        assert os.path.exists(merged_roads_path)
        
        # Check symlinks to standard names exist
        standard_buildings_path = os.path.join(temp_dir, "vector", "osm_arcanum.gpkg")
        standard_roads_path = os.path.join(temp_dir, "vector", "osm_roads.gpkg")
        assert os.path.exists(standard_buildings_path)
        assert os.path.exists(standard_roads_path)
        
        # Verify merged file contents
        try:
            merged_buildings = gpd.read_file(merged_buildings_path)
            merged_roads = gpd.read_file(merged_roads_path)
            
            # Should have data from both cells
            assert len(merged_buildings) == len(mock_osm_response) * 2
            assert len(merged_roads) == len(roads_gdf) * 2
        except Exception as e:
            pytest.fail(f"Failed to read merged GeoPackage files: {str(e)}")

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="OSMnx required")
    @patch('osmnx.features_from_bbox')
    @patch('osmnx.graph_from_bbox')
    @patch('osmnx.graph_to_gdfs')
    def test_end_to_end_bbox_download(self, mock_graph_to_gdfs, mock_graph_from_bbox, 
                                      mock_features_from_bbox, temp_dir, mock_osm_response):
        """Test the end-to-end OSM bounding box download process."""
        # Mock OSMnx responses
        mock_graph = MagicMock()
        mock_graph_from_bbox.return_value = mock_graph
        
        mock_features_from_bbox.return_value = mock_osm_response
        
        mock_roads = gpd.GeoDataFrame({'geometry': mock_osm_response['geometry']})
        mock_graph_to_gdfs.return_value = mock_roads
        
        # Test bounds
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        # Call the function
        with patch('modules.osm.bbox_downloader.get_osmnx_version', return_value="2.0.0"):
            result = download_osm_data(bounds, temp_dir)
        
        # Verify success
        assert result["success"] is True
        assert "buildings_path" in result
        assert "roads_path" in result
        
        # Verify OSMnx functions were called with correct parameters
        mock_graph_from_bbox.assert_called_once()
        mock_features_from_bbox.assert_called_once()
        
        # Verify files were created
        assert os.path.exists(result["buildings_path"])
        assert os.path.exists(result["roads_path"])


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="OSMnx and GeoDataFrames required")
class TestOsmFileFormats:
    """Tests for OSM file format compatibility."""

    def test_read_mock_buildings_json(self, test_osm_buildings, temp_dir):
        """Test reading the mock buildings JSON file and converting to GeoDataFrame."""
        # Load the test data
        with open(test_osm_buildings) as f:
            data = json.load(f)
        
        # Convert to GeoDataFrame
        try:
            buildings_gdf = gpd.GeoDataFrame.from_features(data["features"])
            
            # Verify the data was loaded correctly
            assert len(buildings_gdf) == 2
            assert "height" in buildings_gdf.columns
            assert "material" in buildings_gdf.columns
            assert "building" in buildings_gdf.columns
            
            # Test saving to GeoPackage format
            output_path = os.path.join(temp_dir, "test_buildings.gpkg")
            buildings_gdf.to_file(output_path, layer='buildings', driver='GPKG')
            
            # Verify the file was created
            assert os.path.exists(output_path)
            
            # Read it back and verify
            read_gdf = gpd.read_file(output_path)
            assert len(read_gdf) == len(buildings_gdf)
            
        except Exception as e:
            pytest.fail(f"Failed to process GeoJSON: {str(e)}")

    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="OSMnx required")
    def test_osmnx_version_compatibility(self):
        """Test compatibility with the installed OSMnx version."""
        from modules.osm.config import get_osmnx_version
        
        # Get the installed version
        version = get_osmnx_version()
        
        # Make sure we can get the version
        assert version is not None
        
        # Configure OSMnx based on the version
        from modules.osm.config import configure_osmnx
        try:
            settings = configure_osmnx()
            assert "max_query_area_size" in settings
            assert "requests_timeout" in settings
            assert "overpass_url" in settings
        except Exception as e:
            pytest.fail(f"Failed to configure OSMnx: {str(e)}")


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="OSMnx and GeoDataFrames required")
class TestOsmErrorHandling:
    """Tests for OSM module error handling."""

    @patch('modules.osm.bbox_downloader.download_buildings_and_roads')
    def test_download_osm_data_with_errors(self, mock_download, temp_dir):
        """Test error handling in the download_osm_data function."""
        # Mock an error in the download function
        mock_download.side_effect = Exception("Test error")
        
        # Test bounds
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        # Call the function
        result = download_osm_data(bounds, temp_dir)
        
        # Function should still "succeed" with placeholders
        assert result["success"] is True
        assert "error" in result
        assert "Test error" in result["error"]
        
        # Check for placeholder files
        assert "buildings_path" in result
        assert "roads_path" in result
        assert os.path.exists(result["buildings_path"])
        assert os.path.exists(result["roads_path"])
        
        # Verify placeholder content
        with open(result["buildings_path"]) as f:
            content = f.read()
            assert "Placeholder for OSM buildings data" in content

    @patch('modules.osm.grid_downloader.download_osm_for_cell')
    def test_download_osm_grid_with_cell_errors(self, mock_download_cell, temp_dir):
        """Test error handling in the download_osm_grid function when some cells fail."""
        # Mock success and failure in different cells
        mock_download_cell.side_effect = [
            {"success": True, "cell_name": "cell_0_0"},
            {"success": False, "error": "Test error", "cell_name": "cell_0_1"},
            {"success": True, "cell_name": "cell_0_2"}
        ]
        
        # Test bounds
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        # Call the function with small cell size to ensure multiple cells
        with patch('modules.osm.grid_downloader.get_grid_cells', return_value=[
            {"grid_position": (0, 0)}, 
            {"grid_position": (0, 1)}, 
            {"grid_position": (0, 2)}
        ]):
            result = download_osm_grid(bounds, temp_dir, cell_size_meters=100)
        
        # Verify that the function continued despite the error
        assert result["success_count"] == 2
        assert result["fail_count"] == 1
        assert len(result["cell_results"]) == 3
        
        # Verify metadata file was created
        assert os.path.exists(os.path.join(temp_dir, "vector", "grid_download_metadata.json"))

    def test_osm_api_errors(self, temp_dir):
        """Test handling actual API errors with OSMnx."""
        if not HAS_DEPENDENCIES:
            pytest.skip("OSMnx required for this test")
        
        # Invalid bounds that would cause an API error
        invalid_bounds = {
            "north": 1000,  # Invalid latitude
            "south": 990,
            "east": 1000,   # Invalid longitude
            "west": 990
        }
        
        # Call the function (should fail gracefully)
        try:
            result = download_osm_data(invalid_bounds, temp_dir)
            
            # Should return a failure status with error info
            assert result["success"] is False
            assert "error" in result
            
        except Exception as e:
            pytest.fail(f"download_osm_data failed to handle API error gracefully: {str(e)}")


@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="GeoPandas required")
class TestMapProjections:
    """Tests for handling different map projections and coordinate systems."""

    @patch('modules.osm.bbox_downloader.download_buildings_and_roads')
    def test_coordinate_transformation(self, mock_download, temp_dir, mock_osm_response):
        """Test coordinate transformation from other systems to WGS84."""
        from modules.osm.bbox_downloader import download_osm_data
        import pyproj
        
        # Mock the download response
        mock_roads = gpd.GeoDataFrame({'geometry': mock_osm_response['geometry']})
        mock_download.return_value = (mock_osm_response, mock_roads)
        
        # Test bounds in a different coordinate system (Web Mercator/EPSG:3857)
        web_mercator_bounds = {
            "north": 6701387.0,
            "south": 6700108.0,
            "east": -13569.0,
            "west": -14473.0
        }
        
        # Call the function with the non-WGS84 bounds
        result = download_osm_data(web_mercator_bounds, temp_dir, coordinate_system="EPSG:3857")
        
        # Check if the function returned success
        assert result["success"] is True
        
        # Verify that OSMnx was called with transformed coordinates
        north, south, east, west = mock_download.call_args[0]
        
        # The transformed coordinates should be in WGS84 range
        assert -90 <= north <= 90
        assert -90 <= south <= 90
        assert -180 <= east <= 180
        assert -180 <= west <= 180