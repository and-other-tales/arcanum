#!/usr/bin/env python3
"""
Unit tests for the Arcanum integration modules.
"""

import os
import sys
import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock, call

# Import the modules to test
try:
    from modules.integration.g3d_tiles import fetch_tiles, fetch_city_tiles
    from modules.integration.google_3d_tiles_integration import Google3DTilesIntegration
    from modules.integration.coverage import verify_coverage, CoverageVerifier
    from modules.integration.textures import project_textures, TextureProjector
    from modules.integration.spatial_bounds import (
        bounds_to_polygon, polygon_to_bounds, city_polygon, 
        create_grid_from_bounds, calculate_bounds_area_km2
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Skip tests if modules not available
    pytestmark = pytest.mark.skip(reason="Integration modules not available")


class TestGoogle3DTiles:
    """Test the Google 3D Tiles integration functions."""

    @patch("modules.integration.g3d_tiles.Google3DTilesIntegration")
    def test_fetch_tiles(self, mock_integration, sample_bounds, temp_dir):
        """Test fetching 3D tiles for a geographic area."""
        # Set up the mock
        mock_instance = mock_integration.return_value
        mock_instance.fetch_tileset_json.return_value = {"asset": {}}
        mock_instance.fetch_tiles_recursive.return_value = ["tile1.b3dm", "tile2.b3dm"]
        
        # Call the function
        result = fetch_tiles(sample_bounds, temp_dir)
        
        # Check the result
        assert mock_integration.called
        assert mock_instance.fetch_tileset_json.called
        assert mock_instance.fetch_tiles_recursive.called
        assert result["success"] is True
        assert result["downloaded_tiles"] == 3  # 2 tiles + tileset.json
    
    @patch("modules.integration.g3d_tiles.Google3DTilesIntegration")
    def test_fetch_tiles_error(self, mock_integration, sample_bounds, temp_dir):
        """Test fetching 3D tiles with an error."""
        # Set up the mock to raise an exception
        mock_instance = mock_integration.return_value
        mock_instance.fetch_tileset_json.side_effect = Exception("Test error")
        
        # Call the function
        result = fetch_tiles(sample_bounds, temp_dir)
        
        # Check the result
        assert mock_integration.called
        assert mock_instance.fetch_tileset_json.called
        assert result["success"] is False
        assert "error" in result
    
    @patch("modules.integration.city_tiles.Google3DTilesIntegration")
    def test_fetch_city_3d_tiles(self, mock_integration, temp_dir, sample_city_name):
        """Test fetching 3D tiles for a city."""
        # Set up the mock
        mock_instance = mock_integration.return_value
        mock_instance.fetch_city_tiles.return_value = {
            "success": True,
            "downloaded_tiles": 10,
            "city": sample_city_name
        }
        
        # Import the function
        from modules.integration.city_tiles import fetch_city_3d_tiles
        
        # Call the function
        result = fetch_city_3d_tiles(sample_city_name, temp_dir)
        
        # Check the result
        assert mock_integration.called
        assert mock_instance.fetch_city_tiles.called
        assert result["success"] is True
        assert result["downloaded_tiles"] == 10
        assert result["city"] == sample_city_name
    
    @patch("modules.integration.city_tiles.Google3DTilesIntegration")
    def test_fetch_city_3d_tiles_error(self, mock_integration, temp_dir, sample_city_name):
        """Test fetching 3D tiles for a city with an error."""
        # Set up the mock to raise an exception
        mock_instance = mock_integration.return_value
        mock_instance.fetch_city_tiles.side_effect = Exception("Test error")
        
        # Import the function
        from modules.integration.city_tiles import fetch_city_3d_tiles
        
        # Call the function
        result = fetch_city_3d_tiles(sample_city_name, temp_dir)
        
        # Check the result
        assert mock_integration.called
        assert mock_instance.fetch_city_tiles.called
        assert result["success"] is False
        assert "error" in result


class TestCoverageVerification:
    """Test the coverage verification functions."""

    @patch("modules.integration.coverage._verify_street_view_coverage")
    @patch("modules.integration.coverage._verify_3d_tiles_coverage")
    def test_verify_coverage_street_view(self, mock_3d, mock_sv, temp_dir):
        """Test verifying Street View coverage."""
        # Set up the mocks
        mock_sv.return_value = {"success": True, "coverage_percent": 75}
        
        # Call the function
        result = verify_coverage(mode="street_view", input_dir=temp_dir)
        
        # Check the result
        assert mock_sv.called
        assert not mock_3d.called
        assert result["success"] is True
        assert result["coverage_percent"] == 75
    
    @patch("modules.integration.coverage._verify_street_view_coverage")
    @patch("modules.integration.coverage._verify_3d_tiles_coverage")
    def test_verify_coverage_3d_tiles(self, mock_3d, mock_sv, temp_dir, sample_city_name):
        """Test verifying 3D tiles coverage."""
        # Set up the mocks
        mock_3d.return_value = {"success": True, "coverage_percent": 85}
        
        # Call the function
        result = verify_coverage(mode="3d_tiles", input_dir=temp_dir, city_name=sample_city_name)
        
        # Check the result
        assert mock_3d.called
        assert not mock_sv.called
        assert result["success"] is True
        assert result["coverage_percent"] == 85
    
    @patch("modules.integration.coverage._verify_street_view_coverage")
    @patch("modules.integration.coverage._verify_3d_tiles_coverage")
    def test_verify_coverage_both(self, mock_3d, mock_sv, temp_dir, sample_city_name):
        """Test verifying both Street View and 3D tiles coverage."""
        # Set up the mocks
        mock_sv.return_value = {"success": True, "coverage_percent": 75}
        mock_3d.return_value = {"success": True, "coverage_percent": 85}
        
        # Call the function
        result = verify_coverage(
            mode="both", 
            input_dir=temp_dir, 
            city_name=sample_city_name
        )
        
        # Check the result
        assert mock_sv.called
        assert mock_3d.called
        assert result["success"] is True
        assert "street_view" in result
        assert "3d_tiles" in result
    
    def test_coverage_verifier_init(self, sample_bounds):
        """Test initializing the CoverageVerifier."""
        verifier = CoverageVerifier(bounds=sample_bounds)
        assert verifier.bounds == sample_bounds
    
    @patch("modules.integration.coverage._verify_street_view_coverage")
    def test_coverage_verifier_verify_street_view(self, mock_verify, temp_dir, sample_bounds):
        """Test verifying Street View coverage with CoverageVerifier."""
        # Set up the mock
        mock_verify.return_value = {"success": True, "coverage_percent": 75}
        
        # Create verifier and call method
        verifier = CoverageVerifier(bounds=sample_bounds)
        result = verifier.verify_street_view_coverage(temp_dir)
        
        # Check result
        assert mock_verify.called
        assert result["success"] is True
        assert result["coverage_percent"] == 75


class TestTextureProjection:
    """Test the texture projection functions."""

    def test_project_textures(self, temp_dir):
        """Test projecting textures."""
        # Create input directory with a test image
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        with open(os.path.join(input_dir, "test.jpg"), "w") as f:
            f.write("test")
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "output")
        
        # Call the function
        result = project_textures(input_dir, output_dir)
        
        # Check result
        assert result["success"] is True
        assert result["mode"] == "facades"
        assert result["quality"] == "high"
        assert result["output_dir"] == output_dir
        assert os.path.isdir(output_dir)
    
    def test_texture_projector_init(self):
        """Test initializing the TextureProjector."""
        projector = TextureProjector(quality="medium", use_ai=False)
        assert projector.quality == "medium"
        assert projector.use_ai is False
        assert projector.cache_dir is None
    
    def test_texture_projector_project_facades(self, temp_dir):
        """Test projecting facades with TextureProjector."""
        # Create input directory with a test image
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        with open(os.path.join(input_dir, "test.jpg"), "w") as f:
            f.write("test")
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "output")
        
        # Create projector and call method
        projector = TextureProjector(quality="high")
        result = projector.project_facades(input_dir, output_dir)
        
        # Check result
        assert result["success"] is True
        assert result["mode"] == "facades"
        assert result["quality"] == "high"
        assert result["output_dir"] == output_dir
        assert os.path.isdir(output_dir)


class TestSpatialBounds:
    """Test the spatial bounds functions."""

    def test_bounds_to_polygon(self, sample_bounds):
        """Test converting bounds to polygon."""
        polygon = bounds_to_polygon(sample_bounds)
        assert polygon.is_valid
        assert polygon.area > 0
    
    def test_polygon_to_bounds(self, sample_bounds):
        """Test converting polygon to bounds."""
        polygon = bounds_to_polygon(sample_bounds)
        bounds = polygon_to_bounds(polygon)
        
        # Check that bounds are close to original
        assert abs(bounds["north"] - sample_bounds["north"]) < 1e-6
        assert abs(bounds["south"] - sample_bounds["south"]) < 1e-6
        assert abs(bounds["east"] - sample_bounds["east"]) < 1e-6
        assert abs(bounds["west"] - sample_bounds["west"]) < 1e-6
    
    def test_city_polygon(self, sample_city_name):
        """Test getting a city polygon."""
        polygon = city_polygon(sample_city_name)
        assert polygon is not None
        assert polygon.is_valid
        assert polygon.area > 0
    
    def test_create_grid_from_bounds(self, sample_bounds):
        """Test creating a grid from bounds."""
        # Create a grid with 1km cells
        grid = create_grid_from_bounds(sample_bounds, 1000)
        assert len(grid) > 0
        
        # Check that all grid cells are valid bounds
        for cell in grid:
            assert "north" in cell
            assert "south" in cell
            assert "east" in cell
            assert "west" in cell
            assert cell["north"] > cell["south"]
            assert cell["east"] > cell["west"]
    
    def test_calculate_bounds_area_km2(self, sample_bounds):
        """Test calculating bounds area in square kilometers."""
        area = calculate_bounds_area_km2(sample_bounds)
        assert area > 0
        # London area should be approximately 1500-1600 km²
        # Our test bounds are much smaller
        assert area < 10