#!/usr/bin/env python3
"""
Test the complete Arcanum pipeline from OSM data to Unity with tile server integration.
"""

import os
import logging
import json
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Define test directories
TEST_DIR = Path("test_data")
OUTPUT_DIR = TEST_DIR / "integrated_output"
OSM_DIR = TEST_DIR / "osm"
TEXTURES_DIR = TEST_DIR / "textures"
UNITY_DIR = TEST_DIR / "unity"
SERVER_DIR = TEST_DIR / "server"

# Clean and create directories
def setup_directories():
    """Setup test directories."""
    # Clean output dir if it exists
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    # Create directories
    dirs = [
        OUTPUT_DIR,
        OUTPUT_DIR / "atlases",
        OUTPUT_DIR / "building_data",
        OUTPUT_DIR / "unity",
        OUTPUT_DIR / "materials",
        SERVER_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    logger.info("Directories set up")

# Import modules only after directories are set up
def run_pipeline():
    """Run the complete pipeline."""
    # 1. Import modules
    from modules.texture_atlas_manager import TextureAtlasManager
    from modules.osm.building_processor import OSMBuildingProcessor
    from integration_tools.unity_prefab_generator import UnityPrefabGenerator
    from integration_tools.unity_material_pipeline import UnityMaterialPipeline
    from integration_tools.tile_server_integration import TileServerIntegration
    
    # 2. Initialize components
    texture_atlas_manager = TextureAtlasManager(str(OUTPUT_DIR / "atlases"))
    logger.info("1. Initialized TextureAtlasManager")
    
    building_processor = OSMBuildingProcessor(
        str(OUTPUT_DIR / "building_data"),
        texture_atlas_manager,
        str(TEXTURES_DIR)
    )
    logger.info("2. Initialized OSMBuildingProcessor")
    
    # 3. Process OSM data
    building_result = building_processor.process_osm_buildings(str(OSM_DIR / "test_buildings.json"))
    logger.info(f"3. Processed {building_result['success_count']} buildings from OSM data")
    
    # 4. Generate Unity prefabs
    unity_generator = UnityPrefabGenerator(
        str(OUTPUT_DIR / "unity"),
        str(UNITY_DIR),
        building_processor,
        texture_atlas_manager
    )
    
    prefab_result = unity_generator.generate_all_prefabs(
        str(OUTPUT_DIR / "building_data/metadata/buildings_metadata.json")
    )
    logger.info(f"4. Generated {prefab_result['success_count']} Unity prefabs")
    
    # 5. Generate materials
    material_pipeline = UnityMaterialPipeline(
        str(OUTPUT_DIR / "materials"),
        str(TEXTURES_DIR),
        str(UNITY_DIR)
    )
    
    # Create individual materials
    with open(OUTPUT_DIR / "building_data/metadata/buildings_metadata.json", 'r') as f:
        buildings_metadata = json.load(f)
    
    # Create one material manually for testing
    material_result = material_pipeline.create_material_for_building(
        "test_building_1",
        str(TEXTURES_DIR / "brick.jpg"),
        "brick",
        buildings_metadata.get("test_building_1", {})
    )
    logger.info(f"5. Created material: {material_result['success'] if 'success' in material_result else 'Failed'}")
    
    # 6. Export to tile server
    tile_server = TileServerIntegration(
        str(OUTPUT_DIR),
        str(SERVER_DIR)
    )
    
    server_result = tile_server.upload_assets()
    logger.info(f"6. Uploaded to tile server: {server_result['success'] if 'success' in server_result else 'Failed'}")
    
    config_result = tile_server.generate_unity_server_config()
    logger.info(f"7. Generated Unity server config: {config_result['success'] if 'success' in config_result else 'Failed'}")
    
    return {
        "building_result": building_result,
        "prefab_result": prefab_result,
        "material_result": material_result,
        "server_result": server_result,
        "config_result": config_result
    }

if __name__ == "__main__":
    try:
        print("=== Starting Integrated Pipeline Test ===")
        setup_directories()
        results = run_pipeline()
        print("\n=== Integrated Pipeline Results ===")
        print(f"OSM Building Processing: {results['building_result']['success_count']} of {results['building_result']['total']} buildings")
        print(f"Unity Prefab Generation: {results['prefab_result']['success_count']} of {results['prefab_result']['total']} prefabs")
        print(f"Unity Server Config Generated: {results['config_result']['success'] if 'success' in results['config_result'] else 'Failed'}")
        print("\n=== Pipeline Test Complete ===")
    except Exception as e:
        logger.error(f"Pipeline test failed: {str(e)}")
        print(f"ERROR: Pipeline test failed: {str(e)}")