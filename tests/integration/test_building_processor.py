#!/usr/bin/env python3
"""
Integration tests for the OSM Building Processor module.
"""

import os
import sys
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

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
    pytestmark = pytest.mark.skip(reason="OSM Building Processor module or dependencies not available")


@pytest.fixture
def sample_buildings_json(test_data_dir):
    """Return path to sample OSM buildings data."""
    return os.path.join(test_data_dir, "osm", "test_buildings.json")


@pytest.fixture
def texture_dir(temp_dir):
    """Create a temporary directory with test textures."""
    if not HAS_PIL:
        pytest.skip("PIL not available for texture tests")
    
    texture_dir = os.path.join(temp_dir, "textures")
    os.makedirs(texture_dir, exist_ok=True)
    
    # Create various test textures
    texture_data = [
        ("brick.jpg", (180, 100, 80)),
        ("glass.jpg", (200, 220, 255)),
        ("residential.jpg", (160, 140, 120)),
        ("commercial.jpg", (120, 140, 180)),
        ("default.jpg", (200, 200, 200))
    ]
    
    for name, color in texture_data:
        img = Image.new("RGB", (256, 256), color)
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), name, fill=(0, 0, 0))
        img.save(os.path.join(texture_dir, name), "JPEG", quality=90)
    
    return texture_dir


@pytest.mark.skipif(not HAS_PIL, reason="PIL required for texture tests")
class TestBuildingProcessorIntegration:
    """Integration tests for the Building Processor."""

    def test_end_to_end_building_processing(self, temp_dir, sample_buildings_json, texture_dir):
        """Test end-to-end building processing pipeline."""
        # Create output directory
        output_dir = os.path.join(temp_dir, "building_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize texture atlas manager
        atlas_dir = os.path.join(output_dir, "atlases")
        os.makedirs(atlas_dir, exist_ok=True)
        
        texture_atlas_manager = TextureAtlasManager(atlas_dir)
        
        # Copy sample textures to texture directory if needed
        if not os.path.exists(os.path.join(output_dir, "textures")):
            shutil.copytree(texture_dir, os.path.join(output_dir, "textures"))
        
        # Initialize building processor
        processor = OSMBuildingProcessor(
            output_dir=output_dir,
            texture_atlas_manager=texture_atlas_manager,
            texture_dir=texture_dir
        )
        
        # Process all buildings
        results = processor.process_osm_buildings(sample_buildings_json)
        
        # Verify results
        assert results["total"] >= 1
        assert results["success_count"] >= 1
        
        # Check if expected files were created
        metadata_file = os.path.join(output_dir, "metadata", "buildings_metadata.json")
        assert os.path.exists(metadata_file)
        
        # Load and validate metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Should have entries for all processed buildings
        assert len(metadata) >= results["success_count"]
        
        # Each building should have a model file
        models_dir = os.path.join(output_dir, "models")
        for building_id in metadata:
            model_path = os.path.join(models_dir, f"{building_id}.obj")
            assert os.path.exists(model_path)
            
            # Verify model file content
            with open(model_path, 'r') as f:
                content = f.read()
                assert "v " in content  # Has vertices
                assert "vt " in content  # Has texture coordinates
                assert "vn " in content  # Has normals
                assert "f " in content  # Has faces
        
        # Should have created at least one texture atlas
        atlas_files = os.listdir(atlas_dir)
        assert len(atlas_files) > 0
        
        # Check image files to ensure they're valid
        for atlas_file in atlas_files:
            if atlas_file.endswith(('.jpg', '.png')):
                try:
                    img = Image.open(os.path.join(atlas_dir, atlas_file))
                    assert img.size[0] > 0 and img.size[1] > 0
                except Exception as e:
                    pytest.fail(f"Failed to open atlas image {atlas_file}: {str(e)}")

    def test_process_specific_building(self, temp_dir, sample_buildings_json, texture_dir):
        """Test processing a specific building."""
        # Load buildings data
        with open(sample_buildings_json, 'r') as f:
            buildings_data = json.load(f)
        
        # Get the first building
        building = buildings_data["features"][0]
        building_id = building["id"]
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "single_building_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize texture atlas manager
        atlas_dir = os.path.join(output_dir, "atlases")
        os.makedirs(atlas_dir, exist_ok=True)
        
        texture_atlas_manager = TextureAtlasManager(atlas_dir)
        
        # Initialize building processor
        processor = OSMBuildingProcessor(
            output_dir=output_dir,
            texture_atlas_manager=texture_atlas_manager,
            texture_dir=texture_dir
        )
        
        # Process the specific building
        result = processor.process_building(building_id, building)
        
        # Verify result
        assert result["success"] is True
        assert result["building_id"] == building_id
        assert "model_path" in result
        assert os.path.exists(result["model_path"])
        
        # Verify model file
        with open(result["model_path"], 'r') as f:
            content = f.read()
            assert f"# Building {building_id}" in content
            
            # Check for building dimensions in the model
            if "height" in building["properties"]:
                assert f"# Height: {float(building['properties']['height'])}m" in content
        
        # Check metadata
        assert building_id in processor.buildings_metadata
        metadata = processor.buildings_metadata[building_id]
        
        # Verify key properties
        assert "height" in metadata
        assert "type" in metadata
        assert "material" in metadata
        assert "width" in metadata
        assert "depth" in metadata

    def test_multiple_processing_runs(self, temp_dir, sample_buildings_json, texture_dir):
        """Test running the processor multiple times with the same output directory."""
        # Create output directory
        output_dir = os.path.join(temp_dir, "multi_run_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize texture atlas manager
        atlas_dir = os.path.join(output_dir, "atlases")
        os.makedirs(atlas_dir, exist_ok=True)
        
        texture_atlas_manager = TextureAtlasManager(atlas_dir)
        
        # Run first processing
        processor1 = OSMBuildingProcessor(
            output_dir=output_dir,
            texture_atlas_manager=texture_atlas_manager,
            texture_dir=texture_dir
        )
        
        results1 = processor1.process_osm_buildings(sample_buildings_json)
        
        # Save initial metadata
        with open(os.path.join(output_dir, "metadata", "buildings_metadata.json"), 'r') as f:
            initial_metadata = json.load(f)
        
        # Count initial model files
        initial_models = len(os.listdir(os.path.join(output_dir, "models")))
        
        # Run second processing (should overwrite or update existing files)
        processor2 = OSMBuildingProcessor(
            output_dir=output_dir,
            texture_atlas_manager=texture_atlas_manager,
            texture_dir=texture_dir
        )
        
        results2 = processor2.process_osm_buildings(sample_buildings_json)
        
        # Verify results
        assert results1["total"] == results2["total"]
        assert results1["success_count"] == results2["success_count"]
        
        # Verify metadata and models were updated
        with open(os.path.join(output_dir, "metadata", "buildings_metadata.json"), 'r') as f:
            updated_metadata = json.load(f)
        
        # Metadata should be identical or slightly different if timestamps are included
        assert len(initial_metadata) == len(updated_metadata)
        
        # Model count should remain the same
        updated_models = len(os.listdir(os.path.join(output_dir, "models")))
        assert initial_models == updated_models


@pytest.mark.skipif(not HAS_PIL, reason="PIL required for texture tests")
class TestBuildingTexturing:
    """Tests for building texturing and UV mapping."""

    def test_material_selection(self, temp_dir, texture_dir):
        """Test texture selection based on building materials."""
        # Initialize processor
        output_dir = os.path.join(temp_dir, "material_test_output")
        os.makedirs(output_dir, exist_ok=True)
        
        texture_atlas_manager = TextureAtlasManager(os.path.join(output_dir, "atlases"))
        
        processor = OSMBuildingProcessor(
            output_dir=output_dir,
            texture_atlas_manager=texture_atlas_manager,
            texture_dir=texture_dir
        )
        
        # Create test buildings with different materials
        building_types = [
            {
                "id": "brick_building",
                "properties": {"material": "brick", "type": "residential"}
            },
            {
                "id": "glass_building",
                "properties": {"material": "glass", "type": "commercial"}
            },
            {
                "id": "no_material_building",
                "properties": {"type": "residential"}
            }
        ]
        
        # Process each building and check texture selection
        for building_data in building_types:
            building_id = building_data["id"]
            
            # Add geometry
            building_data["geometry"] = {
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
            
            # Patch _generate_building_model and _apply_uv_mapping_to_model to avoid file operations
            with patch.object(OSMBuildingProcessor, '_generate_building_model') as mock_generate:
                with patch.object(OSMBuildingProcessor, '_apply_uv_mapping_to_model') as mock_apply_uv:
                    # Configure mocks to return success
                    mock_generate.return_value = {
                        "success": True,
                        "model_path": os.path.join(output_dir, "models", f"{building_id}.obj")
                    }
                    
                    mock_apply_uv.return_value = {"success": True}
                    
                    # Process the building
                    result = processor.process_building(building_id, building_data)
                    
                    assert result["success"] is True
                    
                    # Check texture selection by examining call to texture_atlas_manager
                    calls = texture_atlas_manager.assign_texture_to_building.call_args_list
                    for call_args, _ in calls:
                        if call_args[0] == building_id:
                            texture_path = call_args[1]
                            texture_filename = os.path.basename(texture_path)
                            
                            # Verify texture selection based on building type/material
                            if building_id == "brick_building":
                                assert "brick" in texture_filename.lower()
                            elif building_id == "glass_building":
                                assert "glass" in texture_filename.lower() or "commercial" in texture_filename.lower()
                            elif building_id == "no_material_building":
                                assert "residential" in texture_filename.lower() or "default" in texture_filename.lower()

    def test_default_texture_creation(self, temp_dir):
        """Test generation of default texture when none available."""
        # Initialize processor in a directory with no textures
        output_dir = os.path.join(temp_dir, "no_textures_output")
        os.makedirs(output_dir, exist_ok=True)
        
        texture_atlas_manager = TextureAtlasManager(os.path.join(output_dir, "atlases"))
        
        processor = OSMBuildingProcessor(
            output_dir=output_dir,
            texture_atlas_manager=texture_atlas_manager
        )
        
        # Test building data
        building_id = "test_building_default_texture"
        building_data = {
            "properties": {"type": "residential"},
            "geometry": {
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
        }
        
        # Process building with no existing textures
        with patch.object(OSMBuildingProcessor, '_generate_building_model') as mock_generate:
            with patch.object(OSMBuildingProcessor, '_apply_uv_mapping_to_model') as mock_apply_uv:
                # Configure mocks to return success
                mock_generate.return_value = {
                    "success": True,
                    "model_path": os.path.join(output_dir, "models", f"{building_id}.obj")
                }
                
                mock_apply_uv.return_value = {"success": True}
                
                # Process the building
                result = processor.process_building(building_id, building_data)
                
                assert result["success"] is True
                
                # Verify default texture was created
                default_texture = os.path.join(output_dir, "textures", "default.jpg")
                assert os.path.exists(default_texture)
                
                # Check texture properties
                img = Image.open(default_texture)
                assert img.size == (1024, 1024)
                assert img.mode == "RGB"

    def test_uv_mapping_application(self, temp_dir, texture_dir):
        """Test applying UV mapping to building models."""
        # Initialize processor
        output_dir = os.path.join(temp_dir, "uv_mapping_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create an actual model file
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        building_id = "test_uv_mapping"
        model_path = os.path.join(models_dir, f"{building_id}.obj")
        
        # Create a simple model file
        with open(model_path, 'w') as f:
            f.write(f"# Building {building_id}\n")
            f.write("v -5 0 -5\n")
            f.write("v 5 0 -5\n")
            f.write("v 5 0 5\n")
            f.write("v -5 0 5\n")
            f.write("v -5 10 -5\n")
            f.write("v 5 10 -5\n")
            f.write("v 5 10 5\n")
            f.write("v -5 10 5\n")
            f.write("\n")
            f.write("vt 0 0\n")
            f.write("vt 1 0\n")
            f.write("vt 1 1\n")
            f.write("vt 0 1\n")
            f.write("\n")
            f.write("vn 0 0 -1\n")
            f.write("vn 1 0 0\n")
            f.write("vn 0 0 1\n")
            f.write("vn -1 0 0\n")
            f.write("vn 0 1 0\n")
            f.write("vn 0 -1 0\n")
            f.write("\n")
            f.write("f 1/1/1 2/2/1 6/3/1 5/4/1\n")
            f.write("f 2/1/2 3/2/2 7/3/2 6/4/2\n")
        
        # Create a mock texture result
        texture_result = {
            "success": True,
            "atlas_path": "/test/atlas_0.jpg",
            "uv_mapping": {
                "uv_sets": {
                    "default": [
                        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9],
                        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]
                    ]
                }
            }
        }
        
        # Initialize processor
        texture_atlas_manager = TextureAtlasManager(os.path.join(output_dir, "atlases"))
        processor = OSMBuildingProcessor(
            output_dir=output_dir,
            texture_atlas_manager=texture_atlas_manager,
            texture_dir=texture_dir
        )
        
        # Apply UV mapping
        result = processor._apply_uv_mapping_to_model(building_id, model_path, texture_result)
        
        # Verify result
        assert result["success"] is True
        
        # Check if the model was updated correctly
        with open(model_path, 'r') as f:
            content = f.read()
            
            # Should have the new UV coordinates
            assert "vt 0.1 0.1" in content
            assert "vt 0.9 0.1" in content
            assert "vt 0.9 0.9" in content
            assert "vt 0.1 0.9" in content
            
            # Should not have the old UV coordinates
            assert "vt 0 0" not in content
            assert "vt 1 0" not in content
            
            # Should still have the original vertices and faces
            assert "v -5 0 -5" in content
            assert "f 1/1/1 2/2/1 6/3/1 5/4/1" in content