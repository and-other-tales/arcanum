#!/usr/bin/env python3
"""
Unit tests for the OSM Building Processor module.
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
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Check for TextureAtlasManager dependency
try:
    from modules.texture_atlas_manager import TextureAtlasManager
    HAS_TEXTURE_MANAGER = True
except ImportError:
    HAS_TEXTURE_MANAGER = False

# Import the module to test
try:
    from modules.osm.building_processor import OSMBuildingProcessor
    BUILDING_PROCESSOR_AVAILABLE = True
except ImportError:
    BUILDING_PROCESSOR_AVAILABLE = False

# Skip all tests if module or dependencies are not available
if not BUILDING_PROCESSOR_AVAILABLE or not HAS_TEXTURE_MANAGER:
    pytestmark = pytest.mark.skip(reason="OSM building processor module or dependencies not available")


@pytest.fixture
def mock_texture_atlas_manager():
    """Create a mock TextureAtlasManager."""
    mock_manager = MagicMock()
    
    # Mock the assign_texture_to_building method
    mock_manager.assign_texture_to_building.return_value = {
        "success": True,
        "atlas_path": "/test/atlases/atlas_0.jpg",
        "uv_mapping": {
            "uv_sets": {
                "default": [
                    [0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2],
                    [0.3, 0.1], [0.4, 0.1], [0.4, 0.2], [0.3, 0.2]
                ]
            }
        }
    }
    
    return mock_manager


@pytest.fixture
def sample_building_json(test_data_dir, temp_dir):
    """Create a sample buildings JSON file for testing."""
    # Path for the test file
    test_file = os.path.join(temp_dir, "test_buildings.json")
    
    # Sample building data
    building_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "test_building_1",
                "properties": {
                    "height": 25.0,
                    "material": "brick",
                    "style": "victorian",
                    "building": "residential",
                    "levels": 3
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0, 0],
                            [10, 0],
                            [10, 15],
                            [0, 15],
                            [0, 0]
                        ]
                    ]
                }
            },
            {
                "type": "Feature",
                "id": "test_building_2",
                "properties": {
                    "height": 40.0,
                    "material": "glass",
                    "style": "modern",
                    "building": "commercial",
                    "levels": 8
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [15, 5],
                            [25, 5],
                            [25, 20],
                            [15, 20],
                            [15, 5]
                        ]
                    ]
                }
            }
        ]
    }
    
    # Write the test file
    with open(test_file, 'w') as f:
        json.dump(building_data, f)
    
    return test_file


@pytest.fixture
def sample_textures(temp_dir):
    """Create sample textures for testing."""
    if not HAS_PIL:
        return []
        
    # Create texture directory
    texture_dir = os.path.join(temp_dir, "textures")
    os.makedirs(texture_dir, exist_ok=True)
    
    # Create a few sample textures
    textures = ["default.jpg", "residential.jpg", "commercial.jpg", "brick.jpg", "glass.jpg"]
    created_paths = []
    
    for texture_name in textures:
        texture_path = os.path.join(texture_dir, texture_name)
        
        # Create a simple colored image
        img = Image.new("RGB", (256, 256), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw the texture name for identification
        draw.text((20, 20), texture_name, fill=(0, 0, 0))
        
        # Save the image
        img.save(texture_path, "JPEG", quality=90)
        created_paths.append(texture_path)
    
    return created_paths


class TestOSMBuildingProcessor:
    """Test the OSM building processor."""

    def test_init(self, temp_dir, mock_texture_atlas_manager):
        """Test initialization with default parameters."""
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager
        )
        
        # Verify directories were created
        assert os.path.isdir(os.path.join(temp_dir, "models"))
        assert os.path.isdir(os.path.join(temp_dir, "metadata"))
        assert os.path.isdir(os.path.join(temp_dir, "textures"))
        
        # Verify properties were set correctly
        assert processor.texture_atlas_manager == mock_texture_atlas_manager
        assert processor.default_height == 10.0
        assert processor.output_dir == Path(temp_dir)

    def test_init_with_params(self, temp_dir, mock_texture_atlas_manager):
        """Test initialization with custom parameters."""
        texture_dir = os.path.join(temp_dir, "custom_textures")
        os.makedirs(texture_dir, exist_ok=True)
        
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager,
            texture_dir=texture_dir,
            default_height=15.0
        )
        
        # Verify properties were set correctly
        assert processor.texture_dir == texture_dir
        assert processor.default_height == 15.0
        
        # Verify buildings_metadata was initialized
        assert processor.buildings_metadata == {}

    def test_get_building_id(self, temp_dir, mock_texture_atlas_manager):
        """Test the _get_building_id method."""
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager
        )
        
        # Test with ID in the building object
        building_with_id = {"id": "test_id_1"}
        assert processor._get_building_id(building_with_id) == "test_id_1"
        
        # Test with ID in properties
        building_with_prop_id = {"properties": {"id": "test_id_2"}}
        assert processor._get_building_id(building_with_prop_id) == "test_id_2"
        
        # Test with geometry hash
        geometry = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
        building_with_geometry = {"geometry": geometry}
        
        # The result should be a hash string
        result = processor._get_building_id(building_with_geometry)
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Test with no identifiable information
        # Should return a random ID with "building_" prefix
        building_no_id = {}
        result = processor._get_building_id(building_no_id)
        assert result.startswith("building_")
        assert len(result) > 9  # building_ + some digits

    def test_calculate_building_dimensions(self, temp_dir, mock_texture_atlas_manager):
        """Test the _calculate_building_dimensions method."""
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager
        )
        
        # Valid polygon geometry
        valid_geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [0, 0],
                    [10, 0],
                    [10, 10],
                    [0, 10],
                    [0, 0]
                ]
            ]
        }
        
        dimensions = processor._calculate_building_dimensions(valid_geometry)
        
        # Check for required keys
        assert "width" in dimensions
        assert "depth" in dimensions
        assert "center" in dimensions
        
        # Width and depth should be non-zero
        assert dimensions["width"] > 0
        assert dimensions["depth"] > 0
        
        # Center should be at [5, 5]
        assert dimensions["center"][0] == 5
        assert dimensions["center"][1] == 5
        
        # Invalid geometry should return default values
        invalid_geometry = {"type": "Point", "coordinates": [0, 0]}
        dimensions = processor._calculate_building_dimensions(invalid_geometry)
        
        assert dimensions["width"] == 10.0
        assert dimensions["depth"] == 10.0
        assert dimensions["center"] == [0, 0]

    @patch('modules.osm.building_processor.logger')
    def test_process_osm_buildings(self, mock_logger, temp_dir, mock_texture_atlas_manager, sample_building_json):
        """Test processing OSM buildings from a JSON file."""
        with patch.object(OSMBuildingProcessor, 'process_building') as mock_process_building:
            # Configure the mock to return success for the first building and failure for the second
            mock_process_building.side_effect = [
                {"success": True, "building_id": "test_building_1"},
                {"success": False, "building_id": "test_building_2", "error": "Test error"}
            ]
            
            # Create processor instance
            processor = OSMBuildingProcessor(
                output_dir=temp_dir,
                texture_atlas_manager=mock_texture_atlas_manager
            )
            
            # Process the sample buildings
            results = processor.process_osm_buildings(sample_building_json)
            
            # Verify the results
            assert results["total"] == 2
            assert results["success_count"] == 1
            assert results["failed_count"] == 1
            assert results["failed_buildings"] == ["test_building_2"]
            
            # Verify process_building was called for each building
            assert mock_process_building.call_count == 2
            
            # Verify buildings_metadata was exported
            metadata_path = os.path.join(temp_dir, "metadata", "buildings_metadata.json")
            assert os.path.exists(metadata_path)

    @patch('modules.osm.building_processor.logger')
    def test_process_building(self, mock_logger, temp_dir, mock_texture_atlas_manager):
        """Test processing a single building."""
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager
        )
        
        # Mock the methods called by process_building
        with patch.object(OSMBuildingProcessor, '_generate_building_model') as mock_generate_model:
            with patch.object(OSMBuildingProcessor, '_assign_building_texture') as mock_assign_texture:
                with patch.object(OSMBuildingProcessor, '_apply_uv_mapping_to_model') as mock_apply_uv:
                    # Configure mocks to return success
                    mock_generate_model.return_value = {
                        "success": True,
                        "model_path": "/test/models/test_building.obj",
                        "vertices": 8,
                        "faces": 12
                    }
                    
                    mock_assign_texture.return_value = {
                        "success": True,
                        "atlas_path": "/test/atlases/atlas_0.jpg",
                        "uv_mapping": {"uv_sets": {"default": [[0.1, 0.1], [0.2, 0.2]]}}
                    }
                    
                    mock_apply_uv.return_value = {
                        "success": True,
                        "uv_mapping": {"uv_sets": {"default": [[0.1, 0.1], [0.2, 0.2]]}}
                    }
                    
                    # Test building data
                    building_id = "test_building_1"
                    building_data = {
                        "properties": {
                            "height": 25.0,
                            "building": "residential",
                            "building:levels": 3,
                            "building:material": "brick"
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [0, 0],
                                    [10, 0],
                                    [10, 15],
                                    [0, 15],
                                    [0, 0]
                                ]
                            ]
                        }
                    }
                    
                    # Process the building
                    result = processor.process_building(building_id, building_data)
                    
                    # Verify the result
                    assert result["success"] is True
                    assert result["building_id"] == building_id
                    assert "model_path" in result
                    assert "texture_path" in result
                    assert "uv_mapping" in result
                    
                    # Verify methods were called with correct parameters
                    mock_generate_model.assert_called_once()
                    mock_assign_texture.assert_called_once()
                    mock_apply_uv.assert_called_once()
                    
                    # Check building metadata was stored
                    assert building_id in processor.buildings_metadata
                    assert processor.buildings_metadata[building_id]["height"] == 25.0
                    assert processor.buildings_metadata[building_id]["type"] == "residential"
                    assert processor.buildings_metadata[building_id]["material"] == "brick"

    @patch('modules.osm.building_processor.logger')
    def test_process_building_no_geometry(self, mock_logger, temp_dir, mock_texture_atlas_manager):
        """Test processing a building with no valid geometry."""
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager
        )
        
        # Building data with no geometry
        building_id = "test_building_invalid"
        building_data = {
            "properties": {
                "height": 25.0,
                "building": "residential"
            }
            # No geometry field
        }
        
        # Process the building
        result = processor.process_building(building_id, building_data)
        
        # Verify the result
        assert result["success"] is False
        assert "error" in result
        assert "No valid geometry" in result["error"]
        
        # Building with empty geometry
        building_data_empty_geom = {
            "properties": {},
            "geometry": {}  # Empty geometry
        }
        
        result = processor.process_building(building_id, building_data_empty_geom)
        assert result["success"] is False

    @patch('modules.osm.building_processor.logger')
    def test_generate_building_model(self, mock_logger, temp_dir, mock_texture_atlas_manager):
        """Test generating a 3D model for a building."""
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager
        )
        
        # Test data
        building_id = "test_building_model"
        geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [0, 0],
                    [10, 0],
                    [10, 10],
                    [0, 10],
                    [0, 0]
                ]
            ]
        }
        metadata = {
            "height": 20.0,
            "width": 10.0,
            "depth": 10.0
        }
        
        # Generate model
        result = processor._generate_building_model(building_id, geometry, metadata)
        
        # Verify the result
        assert result["success"] is True
        assert "model_path" in result
        assert "vertices" in result
        assert "faces" in result
        
        # Check if the model file was created
        model_path = result["model_path"]
        assert os.path.exists(model_path)
        
        # Verify model content
        with open(model_path, 'r') as f:
            content = f.read()
            
            # Check for header
            assert f"# Building {building_id}" in content
            assert f"# Height: {metadata['height']}m" in content
            
            # Check for vertices, texture coordinates, and faces
            assert "v " in content
            assert "vt " in content
            assert "vn " in content
            assert "f " in content
            
            # Should have 8 vertices for a box
            v_count = content.count("\nv ")
            assert v_count == 8
            
            # Should have at least 4 texture coordinates
            vt_count = content.count("\nvt ")
            assert vt_count >= 4
            
            # Should have 6 normal vectors for a box
            vn_count = content.count("\nvn ")
            assert vn_count == 6
            
            # Should have 12 triangles (2 per face * 6 faces)
            f_count = content.count("\nf ")
            assert f_count == 12

    @patch('modules.osm.building_processor.logger')
    def test_assign_building_texture(self, mock_logger, temp_dir, mock_texture_atlas_manager, sample_textures):
        """Test assigning a texture to a building."""
        # Skip if PIL is not available
        if not HAS_PIL:
            pytest.skip("PIL not available for texture tests")
            
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager,
            texture_dir=os.path.join(temp_dir, "textures")
        )
        
        # Test data
        building_id = "test_building_1"
        metadata = {
            "type": "residential",
            "style": "victorian",
            "material": "brick"
        }
        
        # Assign texture
        result = processor._assign_building_texture(building_id, metadata)
        
        # Verify the result
        assert result["success"] is True
        assert "atlas_path" in result
        assert "uv_mapping" in result
        
        # Verify texture atlas manager was called
        mock_texture_atlas_manager.assign_texture_to_building.assert_called_once()
        call_args = mock_texture_atlas_manager.assign_texture_to_building.call_args[0]
        assert call_args[0] == building_id
        
        # Should find a texture matching the building type or material
        texture_path = call_args[1]
        assert os.path.exists(texture_path)
        
        # Try with a different building type
        metadata = {
            "type": "commercial",
            "style": "modern",
            "material": "glass"
        }
        
        result = processor._assign_building_texture("test_building_2", metadata)
        assert result["success"] is True

    @patch('modules.osm.building_processor.logger')
    def test_create_default_texture(self, mock_logger, temp_dir, mock_texture_atlas_manager):
        """Test creating a default texture."""
        # Skip if PIL is not available
        if not HAS_PIL:
            pytest.skip("PIL not available for texture tests")
            
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager
        )
        
        # Create a default texture
        output_path = os.path.join(temp_dir, "default_test.jpg")
        result = processor._create_default_texture(output_path)
        
        # Verify the result
        assert result is True
        assert os.path.exists(output_path)
        
        # Check the image
        try:
            img = Image.open(output_path)
            assert img.size == (1024, 1024)
            assert img.mode == "RGB"
        except Exception as e:
            pytest.fail(f"Failed to open created texture: {str(e)}")

    @patch('modules.osm.building_processor.logger')
    def test_apply_uv_mapping_to_model(self, mock_logger, temp_dir, mock_texture_atlas_manager):
        """Test applying UV mapping to a model."""
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager
        )
        
        # Create a test model file
        model_path = os.path.join(temp_dir, "test_model.obj")
        with open(model_path, 'w') as f:
            f.write("# Test Model\n")
            f.write("v 0 0 0\n")
            f.write("v 1 0 0\n")
            f.write("v 1 1 0\n")
            f.write("v 0 1 0\n")
            f.write("\n")
            f.write("vt 0 0\n")
            f.write("vt 1 0\n")
            f.write("vt 1 1\n")
            f.write("vt 0 1\n")
            f.write("\n")
            f.write("vn 0 0 1\n")
            f.write("\n")
            f.write("f 1/1/1 2/2/1 3/3/1 4/4/1\n")
        
        # Test data
        building_id = "test_building_uv"
        texture_result = {
            "success": True,
            "atlas_path": "/test/atlases/atlas_0.jpg",
            "uv_mapping": {
                "uv_sets": {
                    "default": [
                        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
                    ]
                }
            }
        }
        
        # Apply UV mapping
        result = processor._apply_uv_mapping_to_model(building_id, model_path, texture_result)
        
        # Verify the result
        assert result["success"] is True
        assert "uv_mapping" in result
        
        # Check if the model was updated
        with open(model_path, 'r') as f:
            content = f.read()
            
            # Should contain the new UV coordinates
            assert "vt 0.1 0.1" in content
            assert "vt 0.9 0.1" in content
            assert "vt 0.9 0.9" in content
            assert "vt 0.1 0.9" in content
            
            # Should not contain the old UV coordinates
            assert "vt 0 0" not in content
            assert "vt 1 0" not in content
            
            # The rest of the file should be unchanged
            assert "v 0 0 0" in content
            assert "vn 0 0 1" in content
            assert "f 1/1/1 2/2/1 3/3/1 4/4/1" in content

    @patch('modules.osm.building_processor.logger')
    def test_apply_uv_mapping_no_uvs(self, mock_logger, temp_dir, mock_texture_atlas_manager):
        """Test applying UV mapping when no UV coordinates are available."""
        processor = OSMBuildingProcessor(
            output_dir=temp_dir,
            texture_atlas_manager=mock_texture_atlas_manager
        )
        
        # Create a test model file
        model_path = os.path.join(temp_dir, "test_model.obj")
        with open(model_path, 'w') as f:
            f.write("# Test Model\n")
            f.write("v 0 0 0\n")
            f.write("v 1 0 0\n")
            f.write("v 1 1 0\n")
            f.write("v 0 1 0\n")
            f.write("\n")
            f.write("vt 0 0\n")
            f.write("vt 1 0\n")
            f.write("vt 1 1\n")
            f.write("vt 0 1\n")
        
        # Test with empty UV mapping
        building_id = "test_building_no_uv"
        texture_result = {
            "success": True,
            "atlas_path": "/test/atlases/atlas_0.jpg",
            "uv_mapping": {
                "uv_sets": {
                    # Empty default UV set
                    "default": []
                }
            }
        }
        
        # Apply UV mapping
        result = processor._apply_uv_mapping_to_model(building_id, model_path, texture_result)
        
        # Verify the result
        assert result["success"] is False
        assert "error" in result
        assert "No UV mapping available" in result["error"]
        
        # Test with no UV mapping data
        texture_result = {
            "success": True,
            "atlas_path": "/test/atlases/atlas_0.jpg"
            # No uv_mapping key
        }
        
        result = processor._apply_uv_mapping_to_model(building_id, model_path, texture_result)
        assert result["success"] is False
        
        # Test with invalid model file
        texture_result = {
            "success": True,
            "atlas_path": "/test/atlases/atlas_0.jpg",
            "uv_mapping": {
                "uv_sets": {
                    "default": [[0.1, 0.1], [0.9, 0.9]]
                }
            }
        }
        
        # Create a model file with no texture coordinates
        no_vt_model_path = os.path.join(temp_dir, "no_vt_model.obj")
        with open(no_vt_model_path, 'w') as f:
            f.write("# Model with no texture coordinates\n")
            f.write("v 0 0 0\n")
            f.write("v 1 0 0\n")
            f.write("v 1 1 0\n")
            f.write("v 0 1 0\n")
            f.write("\n")
            f.write("vn 0 0 1\n")
            f.write("\n")
            f.write("f 1//1 2//1 3//1 4//1\n")
        
        result = processor._apply_uv_mapping_to_model(building_id, no_vt_model_path, texture_result)
        assert result["success"] is False
        assert "No texture coordinates found" in result["error"]