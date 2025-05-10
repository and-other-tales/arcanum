#!/usr/bin/env python3
"""
Unit tests for the Arcanum OSM module.
"""

import os
import sys
import pytest
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Try to import required dependencies
try:
    import osmnx
    import geopandas
    import pyproj
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

# Import the module to test
try:
    from modules.osm.bbox_downloader import download_osm_data, download_buildings_and_roads
    from modules.osm.grid_downloader import (
        download_osm_grid,
        download_osm_for_cell,
        get_grid_cells,
        lat_lon_to_meters,
        merge_grid_data
    )
    from modules.osm.config import configure_osmnx, get_osmnx_version, configure_for_grid_downloads
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# Skip all tests if dependencies or modules are missing
if not MODULES_AVAILABLE or not HAS_DEPENDENCIES:
    pytestmark = pytest.mark.skip(reason="OSM module or dependencies not available")


class TestOsmConfig:
    """Test the OSM configuration module."""

    @patch('modules.osm.config.logger')
    def test_get_osmnx_version_success(self, mock_logger):
        """Test getting the OSMnx version when it's installed."""
        with patch.dict('sys.modules', {'osmnx': MagicMock(__version__='1.2.3')}):
            version = get_osmnx_version()
            assert version == '1.2.3'
    
    @patch('modules.osm.config.logger')
    def test_get_osmnx_version_not_installed(self, mock_logger):
        """Test getting the OSMnx version when it's not installed."""
        with patch.dict('sys.modules', {}):
            with patch('modules.osm.config.osmnx', None):
                with patch('builtins.__import__', side_effect=ImportError):
                    version = get_osmnx_version()
                    assert version is None
                    mock_logger.error.assert_called_once()
    
    @patch('modules.osm.config.logger')
    def test_get_osmnx_version_no_attribute(self, mock_logger):
        """Test getting the OSMnx version when __version__ attribute is missing."""
        mock_osmnx = MagicMock()
        delattr(type(mock_osmnx), '__version__')
        
        with patch.dict('sys.modules', {'osmnx': mock_osmnx}):
            with patch('modules.osm.config.osmnx', mock_osmnx):
                version = get_osmnx_version()
                assert version == 'unknown'

    @patch('modules.osm.config.logger')
    def test_configure_osmnx_success(self, mock_logger):
        """Test configuring OSMnx with default parameters."""
        mock_osmnx = MagicMock()
        mock_osmnx.settings = MagicMock(
            max_query_area_size=1000000,
            requests_timeout=60,
            overpass_memory=None,
            overpass_url="https://overpass-api.de/api",
            log_level=10
        )
        
        with patch.dict('sys.modules', {'osmnx': mock_osmnx}):
            with patch('modules.osm.config.osmnx', mock_osmnx):
                result = configure_osmnx()
                
                # Verify OSMnx settings were updated
                assert mock_osmnx.settings.max_query_area_size == 100000000000
                assert mock_osmnx.settings.requests_timeout == 300
                assert mock_osmnx.settings.overpass_url == "https://overpass-api.de/api/interpreter"
                
                # Verify return value
                assert result["max_query_area_size"] == 100000000000
                assert result["requests_timeout"] == 300
                assert result["overpass_url"] == "https://overpass-api.de/api/interpreter"
                assert "log_level" in result
    
    @patch('modules.osm.config.logger')
    def test_configure_osmnx_custom_params(self, mock_logger):
        """Test configuring OSMnx with custom parameters."""
        mock_osmnx = MagicMock()
        mock_osmnx.settings = MagicMock(
            max_query_area_size=1000000,
            requests_timeout=60,
            overpass_memory=None,
            overpass_url="https://overpass-api.de/api",
            log_level=10
        )
        
        with patch.dict('sys.modules', {'osmnx': mock_osmnx}):
            with patch('modules.osm.config.osmnx', mock_osmnx):
                result = configure_osmnx(
                    max_query_area_size=50000000,
                    timeout=120,
                    memory=1073741824,
                    overpass_endpoint="https://example.com/overpass"
                )
                
                # Verify OSMnx settings were updated with custom values
                assert mock_osmnx.settings.max_query_area_size == 50000000
                assert mock_osmnx.settings.requests_timeout == 120
                assert mock_osmnx.settings.overpass_memory == 1073741824
                assert mock_osmnx.settings.overpass_url == "https://example.com/overpass"
                
                # Verify return value
                assert result["max_query_area_size"] == 50000000
                assert result["requests_timeout"] == 120
                assert result["overpass_memory"] == 1073741824
                assert result["overpass_url"] == "https://example.com/overpass"
    
    @patch('modules.osm.config.logger')
    def test_configure_osmnx_not_installed(self, mock_logger):
        """Test configuring OSMnx when it's not installed."""
        with patch.dict('sys.modules', {}):
            with patch('builtins.__import__', side_effect=ImportError):
                result = configure_osmnx()
                assert "error" in result
                assert result["error"] == "OSMnx not installed"
                mock_logger.error.assert_called_once()
    
    @patch('modules.osm.config.configure_osmnx')
    def test_configure_for_grid_downloads(self, mock_configure_osmnx):
        """Test configuring OSMnx for grid downloads."""
        mock_configure_osmnx.return_value = {"success": True}
        
        result = configure_for_grid_downloads(cell_size_meters=1000)
        
        # Verify configure_osmnx was called with correct parameters
        mock_configure_osmnx.assert_called_once_with(
            max_query_area_size=250000000,
            timeout=180,
            memory=1073741824,
            overpass_endpoint="https://overpass-api.de/api/interpreter"
        )
        
        # Verify return value was passed through
        assert result == {"success": True}


class TestBboxDownloader:
    """Test the bounding box downloader module."""

    @patch('modules.osm.bbox_downloader.download_buildings_and_roads')
    @patch('modules.osm.bbox_downloader.configure_osmnx')
    @patch('modules.osm.bbox_downloader.logger')
    def test_download_osm_data_success(self, mock_logger, mock_configure_osmnx, mock_download):
        """Test downloading OSM data successfully."""
        # Mock the return values
        mock_buildings = MagicMock()
        mock_buildings.empty = False
        mock_roads = MagicMock()
        mock_roads.empty = False
        mock_download.return_value = (mock_buildings, mock_roads)
        
        # Test with default coordinate system (EPSG:4326)
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        with patch('os.makedirs') as mock_makedirs:
            with patch('geopandas.GeoDataFrame.to_file') as mock_to_file:
                result = download_osm_data(bounds, "/tmp/test_output")
                
                # Verify correct methods were called
                mock_configure_osmnx.assert_called_once()
                mock_download.assert_called_once_with(
                    bounds["north"], bounds["south"], bounds["east"], bounds["west"]
                )
                assert mock_to_file.call_count == 2  # One for buildings, one for roads
                
                # Verify the result
                assert result["success"] is True
                assert "buildings_path" in result
                assert "roads_path" in result
                assert "message" in result
    
    @patch('modules.osm.bbox_downloader.download_buildings_and_roads')
    @patch('modules.osm.bbox_downloader.configure_osmnx')
    @patch('modules.osm.bbox_downloader.logger')
    def test_download_osm_data_with_transformation(self, mock_logger, mock_configure_osmnx, mock_download):
        """Test downloading OSM data with coordinate transformation."""
        # Mock the return values
        mock_buildings = MagicMock()
        mock_buildings.empty = False
        mock_roads = MagicMock()
        mock_roads.empty = False
        mock_download.return_value = (mock_buildings, mock_roads)
        
        # Test with non-default coordinate system
        bounds = {
            "north": 5720000,  # Example in EPSG:3857
            "south": 5710000,
            "east": -10000,
            "west": -15000
        }
        
        with patch('os.makedirs') as mock_makedirs:
            with patch('geopandas.GeoDataFrame.to_file') as mock_to_file:
                with patch('pyproj.Transformer.from_crs') as mock_transformer:
                    # Mock the transformer
                    mock_transformer_instance = MagicMock()
                    mock_transformer_instance.transform.side_effect = [
                        (51.51, -0.11),  # north, east
                        (51.50, -0.13)   # south, west
                    ]
                    mock_transformer.return_value = mock_transformer_instance
                    
                    result = download_osm_data(bounds, "/tmp/test_output", coordinate_system="EPSG:3857")
                    
                    # Verify transformer was created and used correctly
                    mock_transformer.assert_called_once_with(
                        "EPSG:3857",
                        "EPSG:4326",
                        always_xy=True
                    )
                    assert mock_transformer_instance.transform.call_count == 2
                    
                    # Verify the download was called with transformed coordinates
                    mock_download.assert_called_once_with(51.51, 51.50, -0.11, -0.13)
                    
                    # Verify the result
                    assert result["success"] is True
    
    @patch('modules.osm.bbox_downloader.download_buildings_and_roads')
    @patch('modules.osm.bbox_downloader.configure_osmnx')
    @patch('modules.osm.bbox_downloader.logger')
    def test_download_osm_data_empty_result(self, mock_logger, mock_configure_osmnx, mock_download):
        """Test downloading OSM data with empty results."""
        # Mock empty return values
        mock_buildings = MagicMock()
        mock_buildings.empty = True
        mock_roads = MagicMock()
        mock_roads.empty = True
        mock_download.return_value = (mock_buildings, mock_roads)
        
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        with patch('os.makedirs') as mock_makedirs:
            result = download_osm_data(bounds, "/tmp/test_output")
            
            # Verify the result
            assert result["success"] is True
            assert result["buildings_path"] is None
            assert result["roads_path"] is None
    
    @patch('modules.osm.bbox_downloader.download_buildings_and_roads')
    @patch('modules.osm.bbox_downloader.configure_osmnx')
    @patch('modules.osm.bbox_downloader.logger')
    def test_download_osm_data_download_error(self, mock_logger, mock_configure_osmnx, mock_download):
        """Test handling errors during OSM data download."""
        # Mock an error during download
        mock_download.side_effect = Exception("Test error")
        
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        with patch('os.makedirs') as mock_makedirs:
            with patch('builtins.open', create=True) as mock_open:
                result = download_osm_data(bounds, "/tmp/test_output")
                
                # Verify placeholder files were created
                assert mock_open.call_count == 2
                
                # Verify the result contains error info but still has placeholder paths
                assert result["success"] is True
                assert "error" in result
                assert "buildings_path" in result
                assert "roads_path" in result
                assert "placeholder" in result["message"]
    
    @patch('modules.osm.bbox_downloader.logger')
    def test_download_osm_data_critical_error(self, mock_logger):
        """Test handling critical errors that prevent any processing."""
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        # Create a situation where os.makedirs raises an error
        with patch('os.makedirs', side_effect=PermissionError("Permission denied")):
            result = download_osm_data(bounds, "/tmp/test_output")
            
            # Verify the result shows failure
            assert result["success"] is False
            assert "error" in result
            assert "Permission denied" in result["error"]
            assert "message" in result
            assert "Failed to download" in result["message"]
    
    @patch('modules.osm.bbox_downloader.get_osmnx_version')
    @patch('modules.osm.bbox_downloader.logger')
    def test_download_buildings_and_roads_v1(self, mock_logger, mock_get_version):
        """Test downloading buildings and roads with OSMnx v1.x."""
        # Mock OSMnx version 1.x
        mock_get_version.return_value = "1.2.3"
        
        # Mock OSMnx functions
        with patch('osmnx.graph_from_bbox') as mock_graph_from_bbox:
            with patch('osmnx.features_from_bbox') as mock_features_from_bbox:
                with patch('osmnx.graph_to_gdfs') as mock_graph_to_gdfs:
                    # Set up mock returns
                    mock_graph = MagicMock()
                    mock_graph_from_bbox.return_value = mock_graph
                    
                    mock_buildings = MagicMock()
                    mock_features_from_bbox.return_value = mock_buildings
                    
                    mock_roads = MagicMock()
                    mock_graph_to_gdfs.return_value = mock_roads
                    
                    # Call the function with test coordinates
                    result = download_buildings_and_roads(51.51, 51.50, -0.11, -0.13)
                    
                    # Verify OSMnx v1.x API was used with positional arguments
                    mock_graph_from_bbox.assert_called_once_with(
                        51.51, 51.50, -0.11, -0.13, network_type='all'
                    )
                    
                    mock_features_from_bbox.assert_called_once_with(
                        51.51, 51.50, -0.11, -0.13, tags={'building': True}
                    )
                    
                    # Verify the result
                    assert result[0] == mock_buildings
                    assert result[1] == mock_roads
    
    @patch('modules.osm.bbox_downloader.get_osmnx_version')
    @patch('modules.osm.bbox_downloader.logger')
    def test_download_buildings_and_roads_v2(self, mock_logger, mock_get_version):
        """Test downloading buildings and roads with OSMnx v2.x."""
        # Mock OSMnx version 2.x
        mock_get_version.return_value = "2.0.0"
        
        # Mock OSMnx functions
        with patch('osmnx.graph_from_bbox') as mock_graph_from_bbox:
            with patch('osmnx.features_from_bbox') as mock_features_from_bbox:
                with patch('osmnx.graph_to_gdfs') as mock_graph_to_gdfs:
                    # Set up mock returns
                    mock_graph = MagicMock()
                    mock_graph_from_bbox.return_value = mock_graph
                    
                    mock_buildings = MagicMock()
                    mock_features_from_bbox.return_value = mock_buildings
                    
                    mock_roads = MagicMock()
                    mock_graph_to_gdfs.return_value = mock_roads
                    
                    # Call the function with test coordinates
                    result = download_buildings_and_roads(51.51, 51.50, -0.11, -0.13)
                    
                    # Verify OSMnx v2.x API was used with bbox dictionary
                    mock_graph_from_bbox.assert_called_once_with(
                        bbox=(51.51, 51.50, -0.11, -0.13),
                        network_type='all'
                    )
                    
                    mock_features_from_bbox.assert_called_once_with(
                        bbox=(51.51, 51.50, -0.11, -0.13),
                        tags={'building': True}
                    )
                    
                    # Verify the result
                    assert result[0] == mock_buildings
                    assert result[1] == mock_roads
    
    @patch('modules.osm.bbox_downloader.get_osmnx_version')
    @patch('modules.osm.bbox_downloader.logger')
    def test_download_buildings_and_roads_feature_error(self, mock_logger, mock_get_version):
        """Test handling errors in downloading buildings."""
        # Mock OSMnx version 2.x
        mock_get_version.return_value = "2.0.0"
        
        # Mock OSMnx functions
        with patch('osmnx.graph_from_bbox') as mock_graph_from_bbox:
            with patch('osmnx.features_from_bbox') as mock_features_from_bbox:
                with patch('osmnx.graph_to_gdfs') as mock_graph_to_gdfs:
                    with patch('geopandas.GeoDataFrame') as mock_geodataframe:
                        # Set up mock returns and errors
                        mock_graph = MagicMock()
                        mock_graph_from_bbox.return_value = mock_graph
                        
                        mock_features_from_bbox.side_effect = Exception("Test error")
                        
                        empty_gdf = MagicMock()
                        mock_geodataframe.return_value = empty_gdf
                        
                        mock_roads = MagicMock()
                        mock_graph_to_gdfs.return_value = mock_roads
                        
                        # Call the function
                        result = download_buildings_and_roads(51.51, 51.50, -0.11, -0.13)
                        
                        # Verify error handling with empty GeoDataFrame for buildings
                        assert result[0] == empty_gdf
                        assert result[1] == mock_roads
                        
                        # Verify warning was logged
                        mock_logger.warning.assert_called_once()


class TestGridDownloader:
    """Test the grid downloader module."""

    def test_lat_lon_to_meters(self):
        """Test conversion of lat/lon coordinates to meters."""
        # Test a known distance (approximately 111km per degree at equator)
        distance = lat_lon_to_meters(0, 0, 1, 0)  # 1 degree latitude difference
        assert abs(distance - 111000) < 1000  # Within 1km of expected value
        
        # Test a zero distance
        distance = lat_lon_to_meters(51.5, -0.12, 51.5, -0.12)
        assert distance == 0
        
        # Test a diagonal distance
        distance = lat_lon_to_meters(51.5, -0.12, 51.6, -0.10)
        assert distance > 0

    def test_get_grid_cells(self):
        """Test dividing a bounding box into grid cells."""
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        # Test with a large cell size (should get a single cell)
        cells = get_grid_cells(bounds, cell_size_meters=5000)
        assert len(cells) == 1
        assert cells[0]["north"] == bounds["north"]
        assert cells[0]["south"] == bounds["south"]
        assert cells[0]["east"] == bounds["east"]
        assert cells[0]["west"] == bounds["west"]
        
        # Test with a smaller cell size (should get multiple cells)
        cells = get_grid_cells(bounds, cell_size_meters=100)
        assert len(cells) > 1
        
        # Check properties of the first cell
        assert cells[0]["north"] == bounds["north"]
        assert cells[0]["west"] == bounds["west"]
        assert cells[0]["grid_position"] == (0, 0)
        
        # Verify overall coverage
        # The last cell's south and east should match the bounds
        last_cell = cells[-1]
        # This tolerance accounts for floating point precision
        assert abs(last_cell["south"] - bounds["south"]) < 0.0001
        assert abs(last_cell["east"] - bounds["east"]) < 0.0001

    @patch('modules.osm.grid_downloader.configure_for_grid_downloads')
    @patch('modules.osm.grid_downloader.download_buildings_and_roads')
    @patch('modules.osm.grid_downloader.logger')
    def test_download_osm_for_cell(self, mock_logger, mock_download, mock_configure):
        """Test downloading OSM data for a single grid cell."""
        # Mock return values
        mock_buildings = MagicMock()
        mock_buildings.empty = False
        mock_roads = MagicMock()
        mock_roads.empty = False
        mock_download.return_value = (mock_buildings, mock_roads)
        
        cell_bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13,
            "grid_position": (1, 2)
        }
        
        with patch('os.makedirs') as mock_makedirs:
            with patch('geopandas.GeoDataFrame.to_file') as mock_to_file:
                with patch('builtins.open', create=True) as mock_open:
                    with patch('json.dump') as mock_json_dump:
                        result = download_osm_for_cell(cell_bounds, "/tmp/test_output")
                        
                        # Verify configuration and download were called
                        mock_configure.assert_called_once()
                        mock_download.assert_called_once_with(
                            cell_bounds["north"], 
                            cell_bounds["south"], 
                            cell_bounds["east"], 
                            cell_bounds["west"]
                        )
                        
                        # Verify GeoDataFrames were saved
                        assert mock_to_file.call_count == 2
                        
                        # Verify metadata was saved
                        assert mock_json_dump.call_count == 1
                        
                        # Verify the result
                        assert result["success"] is True
                        assert result["cell_name"] == "cell_1_2"
                        assert "buildings_path" in result
                        assert "roads_path" in result
                        assert "timestamp" in result

    @patch('modules.osm.grid_downloader.get_grid_cells')
    @patch('modules.osm.grid_downloader.download_osm_for_cell')
    @patch('modules.osm.grid_downloader.logger')
    def test_download_osm_grid(self, mock_logger, mock_download_cell, mock_get_grid_cells):
        """Test downloading OSM data using a grid approach."""
        # Mock grid cells
        mock_cells = [
            {
                "north": 51.51,
                "south": 51.505,
                "east": -0.12,
                "west": -0.13,
                "grid_position": (0, 0)
            },
            {
                "north": 51.505,
                "south": 51.50,
                "east": -0.12,
                "west": -0.13,
                "grid_position": (1, 0)
            }
        ]
        mock_get_grid_cells.return_value = mock_cells
        
        # Mock cell download results
        mock_download_cell.side_effect = [
            {
                "cell_name": "cell_0_0",
                "success": True,
                "buildings_path": "/tmp/test_output/grid/cell_0_0/buildings.gpkg",
                "roads_path": "/tmp/test_output/grid/cell_0_0/roads.gpkg"
            },
            {
                "cell_name": "cell_1_0",
                "success": False,
                "error": "Test error"
            }
        ]
        
        bounds = {
            "north": 51.51,
            "south": 51.50,
            "east": -0.11,
            "west": -0.13
        }
        
        with patch('os.makedirs') as mock_makedirs:
            with patch('builtins.open', create=True) as mock_open:
                with patch('json.dump') as mock_json_dump:
                    with patch('time.sleep') as mock_sleep:
                        result = download_osm_grid(
                            bounds, 
                            "/tmp/test_output", 
                            cell_size_meters=200
                        )
                        
                        # Verify grid cells were generated with the correct cell size
                        mock_get_grid_cells.assert_called_once_with(bounds, 200)
                        
                        # Verify each cell was processed
                        assert mock_download_cell.call_count == 2
                        
                        # Verify sleep was called between cells (rate limiting)
                        assert mock_sleep.call_count == 1
                        
                        # Verify metadata was saved
                        assert mock_json_dump.call_count == 1
                        
                        # Verify the result
                        assert result["cell_count"] == 2
                        assert result["success_count"] == 1
                        assert result["fail_count"] == 1
                        assert len(result["cell_results"]) == 2
                        assert "duration_seconds" in result
                        assert "completed_at" in result

    @patch('modules.osm.grid_downloader.logger')
    def test_merge_grid_data_success(self, mock_logger):
        """Test merging grid cell data successfully."""
        with patch('geopandas.read_file') as mock_read_file:
            with patch('geopandas.pd.concat') as mock_concat:
                with patch('geopandas.GeoDataFrame.to_file') as mock_to_file:
                    with patch('pathlib.Path.glob') as mock_glob:
                        with patch('os.path.exists') as mock_exists:
                            with patch('os.remove') as mock_remove:
                                with patch('os.symlink') as mock_symlink:
                                    # Mock file discovery
                                    mock_glob.side_effect = [
                                        [Path("/tmp/test_output/vector/grid/cell_0_0/buildings.gpkg"),
                                         Path("/tmp/test_output/vector/grid/cell_1_0/buildings.gpkg")],
                                        [Path("/tmp/test_output/vector/grid/cell_0_0/roads.gpkg"),
                                         Path("/tmp/test_output/vector/grid/cell_1_0/roads.gpkg")]
                                    ]
                                    
                                    # Mock successful GeoDataFrame operations
                                    mock_buildings_gdf = MagicMock()
                                    mock_roads_gdf = MagicMock()
                                    mock_read_file.side_effect = [mock_buildings_gdf, mock_buildings_gdf,
                                                                 mock_roads_gdf, mock_roads_gdf]
                                    
                                    # Mock merged GeoDataFrames
                                    mock_merged_buildings = MagicMock()
                                    mock_merged_roads = MagicMock()
                                    mock_concat.side_effect = [mock_merged_buildings, mock_merged_roads]
                                    
                                    # Mock file existence checks
                                    mock_exists.return_value = True
                                    
                                    # Call the function
                                    result = merge_grid_data("/tmp/test_output")
                                    
                                    # Verify GeoDataFrames were read
                                    assert mock_read_file.call_count == 4
                                    
                                    # Verify GeoDataFrames were concatenated
                                    assert mock_concat.call_count == 2
                                    
                                    # Verify merged files were saved
                                    assert mock_to_file.call_count == 2
                                    
                                    # Verify symlinks were created
                                    assert mock_remove.call_count == 2
                                    assert mock_symlink.call_count == 2
                                    
                                    # Verify the result
                                    assert result["success"] is True
                                    assert "buildings_path" in result
                                    assert "roads_path" in result
                                    assert "standard_buildings_path" in result
                                    assert "standard_roads_path" in result

    @patch('modules.osm.grid_downloader.logger')
    def test_merge_grid_data_no_files(self, mock_logger):
        """Test merging grid data when no files are found."""
        with patch('pathlib.Path.glob') as mock_glob:
            # Mock empty file lists
            mock_glob.side_effect = [[], []]
            
            # Call the function
            result = merge_grid_data("/tmp/test_output")
            
            # Verify the result
            assert result["success"] is True
            assert result["buildings_path"] is None
            assert result["roads_path"] is None

    @patch('modules.osm.grid_downloader.logger')
    def test_merge_grid_data_error(self, mock_logger):
        """Test handling errors during grid data merging."""
        with patch('pathlib.Path.glob') as mock_glob:
            # Force an ImportError
            mock_glob.side_effect = ImportError("Test import error")
            
            # Call the function
            result = merge_grid_data("/tmp/test_output")
            
            # Verify the result
            assert result["success"] is False
            assert "error" in result
            assert "Test import error" in result["error"]