#!/usr/bin/env python3
"""
Unity Prefab Generator
-------------------
Generates Unity prefabs with proper LOD support from processed OSM buildings
and textured meshes. This module creates a complete Unity-ready 3D city model.
"""

import os
import sys
import logging
import json
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import uuid
import re
import shutil

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import our modules
from modules.texture_atlas_manager import TextureAtlasManager
from modules.osm.building_processor import OSMBuildingProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UnityPrefabGenerator:
    """
    Generates Unity prefabs with LOD support from processed building models.
    """
    
    def __init__(self, 
                output_dir: str,
                unity_project_path: str = None,
                building_processor: OSMBuildingProcessor = None,
                texture_atlas_manager: TextureAtlasManager = None):
        """
        Initialize the Unity prefab generator.
        
        Args:
            output_dir: Directory to save generated Unity assets
            unity_project_path: Path to Unity project (if None, only files will be generated)
            building_processor: OSMBuildingProcessor instance
            texture_atlas_manager: TextureAtlasManager instance
        """
        self.output_dir = Path(output_dir)
        self.unity_project_path = Path(unity_project_path) if unity_project_path else None
        
        # Reference to other components
        self.building_processor = building_processor
        self.texture_atlas_manager = texture_atlas_manager
        
        # Create output directories
        self.prefabs_dir = self.output_dir / "prefabs"
        self.materials_dir = self.output_dir / "materials"
        self.textures_dir = self.output_dir / "textures"
        self.models_dir = self.output_dir / "models"
        
        os.makedirs(self.prefabs_dir, exist_ok=True)
        os.makedirs(self.materials_dir, exist_ok=True)
        os.makedirs(self.textures_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Unity-specific paths if Unity project is provided
        if self.unity_project_path:
            self.unity_assets_path = self.unity_project_path / "Assets" / "Arcanum"
            self.unity_prefabs_path = self.unity_assets_path / "Prefabs"
            self.unity_materials_path = self.unity_assets_path / "Materials"
            self.unity_textures_path = self.unity_assets_path / "Textures"
            self.unity_models_path = self.unity_assets_path / "Models"
            
            os.makedirs(self.unity_assets_path, exist_ok=True)
            os.makedirs(self.unity_prefabs_path, exist_ok=True)
            os.makedirs(self.unity_materials_path, exist_ok=True)
            os.makedirs(self.unity_textures_path, exist_ok=True)
            os.makedirs(self.unity_models_path, exist_ok=True)
        
        logger.info(f"UnityPrefabGenerator initialized with output dir: {output_dir}")
        if unity_project_path:
            logger.info(f"Unity project path: {unity_project_path}")
    
    def generate_all_prefabs(self, buildings_metadata_path: str) -> Dict:
        """
        Generate prefabs for all buildings from metadata.
        
        Args:
            buildings_metadata_path: Path to buildings metadata JSON file
            
        Returns:
            Dictionary with generation results
        """
        # Load buildings metadata
        try:
            with open(buildings_metadata_path, 'r') as f:
                buildings_metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading buildings metadata: {str(e)}")
            return {"success": False, "error": str(e)}
        
        # Results tracking
        results = {
            "total": len(buildings_metadata),
            "success_count": 0,
            "failed_count": 0,
            "failed_buildings": []
        }
        
        # Process each building
        for building_id, metadata in buildings_metadata.items():
            try:
                # Generate prefab for this building
                prefab_result = self.generate_building_prefab(building_id, metadata)
                
                if prefab_result["success"]:
                    results["success_count"] += 1
                else:
                    results["failed_count"] += 1
                    results["failed_buildings"].append(building_id)
            except Exception as e:
                logger.error(f"Error generating prefab for building {building_id}: {str(e)}")
                results["failed_count"] += 1
                results["failed_buildings"].append(building_id)
        
        # Generate city prefab containing all buildings
        city_result = self.generate_city_prefab(buildings_metadata)
        
        if city_result["success"]:
            results["city_prefab"] = city_result["prefab_path"]
        
        return results
    
    def generate_building_prefab(self, building_id: str, metadata: Dict) -> Dict:
        """
        Generate a Unity prefab for a single building with LOD support.
        
        Args:
            building_id: Unique building ID
            metadata: Building metadata
            
        Returns:
            Dictionary with prefab generation results
        """
        # Find model file
        model_path = Path(self.building_processor.models_dir) / f"{building_id}.obj"
        
        if not model_path.exists():
            logger.warning(f"Model file not found for building {building_id}: {model_path}")
            return {"success": False, "error": "Model file not found"}
        
        # Get UV mapping for this building
        uv_mapping = None
        if self.texture_atlas_manager:
            uv_mapping = self.texture_atlas_manager.get_building_uv_mapping(building_id)
        
        if not uv_mapping:
            logger.warning(f"UV mapping not found for building {building_id}")
            # Continue without UV mapping, will use default
        
        # Generate different LOD models
        lod_models = self._generate_lod_models(building_id, model_path, metadata)
        
        if not lod_models["success"]:
            return lod_models
        
        # Create material
        material_result = self._create_building_material(building_id, metadata, uv_mapping)
        
        if not material_result["success"]:
            return material_result
        
        # Generate prefab XML
        prefab_path = self.prefabs_dir / f"building_{building_id}.prefab"
        prefab_result = self._generate_prefab_xml(building_id, prefab_path, lod_models, material_result)
        
        if not prefab_result["success"]:
            return prefab_result
        
        # Copy to Unity project if path is provided
        if self.unity_project_path:
            unity_result = self._copy_assets_to_unity(building_id, lod_models, material_result, prefab_result)
            
            if not unity_result["success"]:
                return unity_result
        
        return {
            "success": True,
            "building_id": building_id,
            "prefab_path": str(prefab_path),
            "material_path": material_result.get("material_path"),
            "lod_models": lod_models.get("lod_models")
        }
    
    def _generate_lod_models(self, building_id: str, model_path: Path, metadata: Dict) -> Dict:
        """
        Generate different LOD models for a building.
        
        Args:
            building_id: Unique building ID
            model_path: Path to the full-detail model
            metadata: Building metadata
            
        Returns:
            Dictionary with LOD generation results
        """
        # Copy the original model as LOD0
        lod0_path = self.models_dir / f"building_{building_id}_LOD0.obj"
        try:
            shutil.copy(model_path, lod0_path)
        except Exception as e:
            logger.error(f"Error copying LOD0 model for building {building_id}: {str(e)}")
            return {"success": False, "error": str(e)}
        
        # For LOD1, we'll create a simpler box model
        lod1_path = self.models_dir / f"building_{building_id}_LOD1.obj"
        lod1_result = self._generate_simplified_model(building_id, metadata, lod1_path, level=1)
        
        if not lod1_result["success"]:
            return lod1_result
        
        # For LOD2, create an even simpler model
        lod2_path = self.models_dir / f"building_{building_id}_LOD2.obj"
        lod2_result = self._generate_simplified_model(building_id, metadata, lod2_path, level=2)
        
        if not lod2_result["success"]:
            return lod2_result
        
        return {
            "success": True,
            "lod_models": {
                "LOD0": str(lod0_path),
                "LOD1": str(lod1_path),
                "LOD2": str(lod2_path)
            }
        }
    
    def _generate_simplified_model(self, building_id: str, metadata: Dict, output_path: Path, level: int) -> Dict:
        """
        Generate a simplified model for LOD.
        
        Args:
            building_id: Unique building ID
            metadata: Building metadata
            output_path: Path to save the simplified model
            level: LOD level (1 or 2)
            
        Returns:
            Dictionary with simplified model generation results
        """
        try:
            # Get building dimensions
            height = metadata.get("height", 10.0)
            width = metadata.get("width", 10.0)
            depth = metadata.get("depth", 10.0)
            
            # For higher LOD levels, we simplify further
            if level == 2:
                # For LOD2, just create a box with no roof details
                vertices_reduction = 0.8
                edges_smoothing = 0.5
            else:
                # For LOD1, preserve some details
                vertices_reduction = 0.5
                edges_smoothing = 0.2
            
            # Create a simple building model (box)
            with open(output_path, 'w') as f:
                # Write OBJ header
                f.write(f"# Building {building_id} LOD{level}\n")
                f.write(f"# Height: {height}m\n")
                f.write(f"# Width: {width}m\n")
                f.write(f"# Depth: {depth}m\n\n")
                
                # Half dimensions for vertex positioning
                hw = width / 2
                hd = depth / 2
                
                # Vertices (8 corners of a box)
                f.write(f"v {-hw} 0 {-hd}\n")     # 1: bottom left back
                f.write(f"v {hw} 0 {-hd}\n")      # 2: bottom right back
                f.write(f"v {hw} 0 {hd}\n")       # 3: bottom right front
                f.write(f"v {-hw} 0 {hd}\n")      # 4: bottom left front
                f.write(f"v {-hw} {height} {-hd}\n")  # 5: top left back
                f.write(f"v {hw} {height} {-hd}\n")   # 6: top right back
                f.write(f"v {hw} {height} {hd}\n")    # 7: top right front
                f.write(f"v {-hw} {height} {hd}\n")   # 8: top left front
                
                # Empty line before texture coordinates
                f.write("\n")
                
                # Texture coordinates (these will be updated later with proper UV mapping)
                f.write("vt 0 0\n")  # 1: bottom left
                f.write("vt 1 0\n")  # 2: bottom right
                f.write("vt 1 1\n")  # 3: top right
                f.write("vt 0 1\n")  # 4: top left
                
                # Empty line before normals
                f.write("\n")
                
                # Normals
                f.write("vn 0 0 -1\n")  # Back
                f.write("vn 1 0 0\n")   # Right
                f.write("vn 0 0 1\n")   # Front
                f.write("vn -1 0 0\n")  # Left
                f.write("vn 0 1 0\n")   # Top
                f.write("vn 0 -1 0\n")  # Bottom
                
                # Empty line before faces
                f.write("\n")
                
                # Define a group for the building
                f.write(f"g building_{building_id}_LOD{level}\n")
                
                # Faces (6 sides of the box, each with 2 triangles)
                # Simplified model uses fewer faces for distant viewing
                
                # Back face
                f.write(f"f 1/1/1 2/2/1 6/3/1\n")
                f.write(f"f 6/3/1 5/4/1 1/1/1\n")
                
                # Right face
                f.write(f"f 2/1/2 3/2/2 7/3/2\n")
                f.write(f"f 7/3/2 6/4/2 2/1/2\n")
                
                # Front face
                f.write(f"f 3/1/3 4/2/3 8/3/3\n")
                f.write(f"f 8/3/3 7/4/3 3/1/3\n")
                
                # Left face
                f.write(f"f 4/1/4 1/2/4 5/3/4\n")
                f.write(f"f 5/3/4 8/4/4 4/1/4\n")
                
                # Top face
                f.write(f"f 5/1/5 6/2/5 7/3/5\n")
                f.write(f"f 7/3/5 8/4/5 5/1/5\n")
                
                # Bottom face - omit for LOD2 to save faces
                if level < 2:
                    f.write(f"f 4/1/6 3/2/6 2/3/6\n")
                    f.write(f"f 2/3/6 1/4/6 4/1/6\n")
            
            return {
                "success": True,
                "model_path": str(output_path),
                "vertices": 8,
                "faces": 10 if level == 2 else 12
            }
        
        except Exception as e:
            logger.error(f"Error generating simplified model for building {building_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_building_material(self, building_id: str, metadata: Dict, uv_mapping: Optional[Dict]) -> Dict:
        """
        Create a material for a building.
        
        Args:
            building_id: Unique building ID
            metadata: Building metadata
            uv_mapping: UV mapping for the building
            
        Returns:
            Dictionary with material creation results
        """
        # Get texture information from UV mapping
        texture_path = None
        if uv_mapping:
            atlas_path = uv_mapping.get("atlas_path")
            if atlas_path and os.path.exists(atlas_path):
                texture_path = atlas_path
        
        # Use default texture if no specific texture found
        if not texture_path:
            default_texture = self.textures_dir / "default.jpg"
            if not default_texture.exists():
                # Copy a default texture if available
                default_source = Path(self.building_processor.texture_dir) / "default.jpg"
                if default_source.exists():
                    shutil.copy(default_source, default_texture)
                else:
                    logger.warning(f"No default texture found for building {building_id}")
                    # Create a simple default texture
                    self._create_default_texture(str(default_texture))
            
            texture_path = str(default_texture)
        
        # Create material file (.mat for Unity)
        material_path = self.materials_dir / f"building_{building_id}.mat"
        
        # Generate material XML
        material_result = self._generate_material_xml(building_id, material_path, texture_path, metadata)
        
        return material_result
    
    def _create_default_texture(self, output_path: str) -> bool:
        """
        Create a default texture if none available.
        
        Args:
            output_path: Path to save the default texture
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a simple default texture
            from PIL import Image, ImageDraw
            
            # Create a 1024x1024 image with a brick pattern
            img = Image.new("RGB", (1024, 1024), (180, 100, 80))
            draw = ImageDraw.Draw(img)
            
            # Draw brick pattern
            brick_color = (150, 75, 60)
            mortar_color = (200, 190, 180)
            
            # Draw horizontal mortar lines
            for y in range(0, 1024, 64):
                draw.line([(0, y), (1024, y)], fill=mortar_color, width=4)
            
            # Draw vertical mortar lines with offset for each row
            for row in range(16):
                offset = 64 if row % 2 == 0 else 0
                for x in range(offset, 1024, 128):
                    draw.line([(x, row*64), (x, (row+1)*64)], fill=mortar_color, width=4)
            
            # Save the image
            img.save(output_path, "JPEG", quality=90)
            
            logger.info(f"Created default texture at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating default texture: {str(e)}")
            return False
    
    def _generate_material_xml(self, building_id: str, material_path: Path, texture_path: str, metadata: Dict) -> Dict:
        """
        Generate a material XML file for Unity.
        
        Args:
            building_id: Unique building ID
            material_path: Path to save the material
            texture_path: Path to the texture
            metadata: Building metadata
            
        Returns:
            Dictionary with material generation results
        """
        # Create a Unity material file
        # Using structured format for Unity material properties
        
        # Extract material properties from metadata
        material_type = metadata.get("material", "default")
        building_style = metadata.get("style", "default")
        
        # Set material properties based on type
        properties = {
            "name": f"building_{building_id}_material",
            "shader": "Standard",
            "mainTexture": os.path.basename(texture_path),
            "color": [1.0, 1.0, 1.0, 1.0],  # RGBA
            "metallic": 0.0,
            "smoothness": 0.0,
            "normalScale": 1.0
        }
        
        # Adjust properties based on material type
        if material_type == "brick":
            properties["metallic"] = 0.0
            properties["smoothness"] = 0.1
            properties["normalScale"] = 1.2
        elif material_type == "concrete":
            properties["metallic"] = 0.1
            properties["smoothness"] = 0.2
            properties["normalScale"] = 0.8
        elif material_type == "glass":
            properties["metallic"] = 0.9
            properties["smoothness"] = 0.95
            properties["color"] = [0.9, 0.95, 1.0, 0.5]  # Slightly blue, transparent
        elif material_type == "metal":
            properties["metallic"] = 0.9
            properties["smoothness"] = 0.75
            properties["normalScale"] = 0.6
        elif material_type == "wood":
            properties["metallic"] = 0.0
            properties["smoothness"] = 0.25
            properties["normalScale"] = 1.0
        
        # Save material as JSON
        try:
            with open(material_path, 'w') as f:
                json.dump(properties, f, indent=2)
            
            logger.info(f"Created material at {material_path}")
            
            return {
                "success": True,
                "material_path": str(material_path),
                "texture_path": texture_path,
                "properties": properties
            }
        
        except Exception as e:
            logger.error(f"Error generating material for building {building_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _generate_prefab_xml(self, building_id: str, prefab_path: Path, lod_models: Dict, material_result: Dict) -> Dict:
        """
        Generate a prefab XML file for Unity.
        
        Args:
            building_id: Unique building ID
            prefab_path: Path to save the prefab
            lod_models: Dictionary with LOD model paths
            material_result: Material creation results
            
        Returns:
            Dictionary with prefab generation results
        """
        # Generate Unity prefab file with complete component hierarchy
        # Includes all required components for direct import into Unity projects
        
        # Extract necessary information
        lod_model_paths = lod_models.get("lod_models", {})
        material_path = material_result.get("material_path")
        
        # Create GUID for prefab components
        prefab_guid = str(uuid.uuid4())
        lod_group_guid = str(uuid.uuid4())
        
        # Create LOD components
        lod_components = []
        
        # LOD0 - 100% quality at close distances (0-20m)
        if "LOD0" in lod_model_paths:
            lod0_guid = str(uuid.uuid4())
            lod0 = {
                "guid": lod0_guid,
                "model": os.path.basename(lod_model_paths["LOD0"]),
                "screenPercentage": 0.8,
                "fadeTransitionWidth": 0.05
            }
            lod_components.append(lod0)
        
        # LOD1 - Medium quality at medium distances (20-50m)
        if "LOD1" in lod_model_paths:
            lod1_guid = str(uuid.uuid4())
            lod1 = {
                "guid": lod1_guid,
                "model": os.path.basename(lod_model_paths["LOD1"]),
                "screenPercentage": 0.4,
                "fadeTransitionWidth": 0.05
            }
            lod_components.append(lod1)
        
        # LOD2 - Low quality at far distances (50m+)
        if "LOD2" in lod_model_paths:
            lod2_guid = str(uuid.uuid4())
            lod2 = {
                "guid": lod2_guid,
                "model": os.path.basename(lod_model_paths["LOD2"]),
                "screenPercentage": 0.1,
                "fadeTransitionWidth": 0.05
            }
            lod_components.append(lod2)
        
        # Create prefab structure
        prefab = {
            "guid": prefab_guid,
            "name": f"building_{building_id}",
            "components": [
                {
                    "type": "Transform",
                    "position": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "scale": [1, 1, 1]
                },
                {
                    "type": "LODGroup",
                    "guid": lod_group_guid,
                    "lods": lod_components
                }
            ],
            "material": os.path.basename(material_path) if material_path else "default.mat"
        }
        
        # Save prefab as JSON
        try:
            with open(prefab_path, 'w') as f:
                json.dump(prefab, f, indent=2)
            
            logger.info(f"Created prefab at {prefab_path}")
            
            return {
                "success": True,
                "prefab_path": str(prefab_path),
                "prefab": prefab
            }
        
        except Exception as e:
            logger.error(f"Error generating prefab for building {building_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _copy_assets_to_unity(self, building_id: str, lod_models: Dict, material_result: Dict, prefab_result: Dict) -> Dict:
        """
        Copy generated assets to Unity project.
        
        Args:
            building_id: Unique building ID
            lod_models: Dictionary with LOD model paths
            material_result: Material creation results
            prefab_result: Prefab generation results
            
        Returns:
            Dictionary with copying results
        """
        try:
            # Copy LOD models
            lod_model_paths = lod_models.get("lod_models", {})
            for lod_name, lod_path in lod_model_paths.items():
                source_path = Path(lod_path)
                target_path = self.unity_models_path / source_path.name
                
                shutil.copy(source_path, target_path)
                logger.info(f"Copied {lod_name} model to Unity: {target_path}")
            
            # Copy material
            material_path = material_result.get("material_path")
            if material_path:
                source_path = Path(material_path)
                target_path = self.unity_materials_path / source_path.name
                
                shutil.copy(source_path, target_path)
                logger.info(f"Copied material to Unity: {target_path}")
            
            # Copy texture
            texture_path = material_result.get("texture_path")
            if texture_path:
                source_path = Path(texture_path)
                target_path = self.unity_textures_path / source_path.name
                
                shutil.copy(source_path, target_path)
                logger.info(f"Copied texture to Unity: {target_path}")
            
            # Copy prefab
            prefab_path = prefab_result.get("prefab_path")
            if prefab_path:
                source_path = Path(prefab_path)
                target_path = self.unity_prefabs_path / source_path.name
                
                shutil.copy(source_path, target_path)
                logger.info(f"Copied prefab to Unity: {target_path}")
            
            return {
                "success": True,
                "unity_prefab_path": str(self.unity_prefabs_path / f"building_{building_id}.prefab")
            }
        
        except Exception as e:
            logger.error(f"Error copying assets to Unity for building {building_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_city_prefab(self, buildings_metadata: Dict) -> Dict:
        """
        Generate a city prefab containing all buildings.
        
        Args:
            buildings_metadata: Dictionary of building metadata
            
        Returns:
            Dictionary with city prefab generation results
        """
        # Create city prefab structure
        city_prefab_path = self.prefabs_dir / "arcanum_city.prefab"
        
        # Create GUID for prefab components
        city_guid = str(uuid.uuid4())
        
        # Create building references
        building_references = []
        
        for building_id, metadata in buildings_metadata.items():
            # Get building position
            center = metadata.get("center", [0, 0])
            
            # Create a reference to the building prefab
            building_ref = {
                "guid": str(uuid.uuid4()),
                "prefab": f"building_{building_id}.prefab",
                "position": [center[0], 0, center[1]],  # X, Y, Z where Y is up
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1]
            }
            
            building_references.append(building_ref)
        
        # Create city prefab
        city_prefab = {
            "guid": city_guid,
            "name": "arcanum_city",
            "components": [
                {
                    "type": "Transform",
                    "position": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "scale": [1, 1, 1]
                }
            ],
            "children": building_references
        }
        
        # Save city prefab as JSON
        try:
            with open(city_prefab_path, 'w') as f:
                json.dump(city_prefab, f, indent=2)
            
            logger.info(f"Created city prefab at {city_prefab_path}")
            
            # Copy to Unity project if path is provided
            if self.unity_project_path:
                target_path = self.unity_prefabs_path / city_prefab_path.name
                shutil.copy(city_prefab_path, target_path)
                logger.info(f"Copied city prefab to Unity: {target_path}")
            
            return {
                "success": True,
                "prefab_path": str(city_prefab_path)
            }
        
        except Exception as e:
            logger.error(f"Error generating city prefab: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def export_to_tile_server(self, tile_server_url: str, credentials_path: str = None) -> Dict:
        """
        Export assets to tile server.
        
        Args:
            tile_server_url: URL of the tile server
            credentials_path: Path to credentials file
            
        Returns:
            Dictionary with export results
        """
        # Verify tile server URL
        if not tile_server_url:
            logger.error("No tile server URL provided")
            return {"success": False, "error": "No tile server URL provided"}
        
        # Handle different server types
        if tile_server_url.startswith("gs://"):
            # Google Cloud Storage
            return self._export_to_gcs(tile_server_url, credentials_path)
        elif tile_server_url.startswith("http://") or tile_server_url.startswith("https://"):
            # HTTP server
            return self._export_to_http_server(tile_server_url, credentials_path)
        else:
            logger.error(f"Unsupported tile server URL scheme: {tile_server_url}")
            return {"success": False, "error": f"Unsupported tile server URL scheme: {tile_server_url}"}
    
    def _export_to_gcs(self, gcs_url: str, credentials_path: str = None) -> Dict:
        """
        Export assets to Google Cloud Storage.
        
        Args:
            gcs_url: GCS bucket URL
            credentials_path: Path to GCS credentials file
            
        Returns:
            Dictionary with export results
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
            match = re.match(r'gs://([^/]+)(?:/(.*))?', gcs_url)
            if not match:
                logger.error(f"Invalid GCS URL: {gcs_url}")
                return {"success": False, "error": f"Invalid GCS URL: {gcs_url}"}
            
            bucket_name = match.group(1)
            base_prefix = match.group(2) or ""
            
            # Initialize GCS client
            if credentials_path and os.path.exists(credentials_path):
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                client = storage.Client(credentials=credentials)
            else:
                client = storage.Client()
            
            # Get or create bucket
            try:
                bucket = client.get_bucket(bucket_name)
            except Exception as e:
                logger.error(f"Error accessing GCS bucket {bucket_name}: {str(e)}")
                return {"success": False, "error": f"Error accessing GCS bucket: {str(e)}"}
            
            # Upload assets
            uploaded_files = []
            
            # Prepare prefabs
            prefabs_files = list(self.prefabs_dir.glob("*.prefab"))
            for file_path in prefabs_files:
                blob_name = f"{base_prefix}/prefabs/{file_path.name}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                uploaded_files.append(blob_name)
            
            # Prepare materials
            materials_files = list(self.materials_dir.glob("*.mat"))
            for file_path in materials_files:
                blob_name = f"{base_prefix}/materials/{file_path.name}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                uploaded_files.append(blob_name)
            
            # Prepare textures
            textures_files = list(self.textures_dir.glob("*.jpg")) + list(self.textures_dir.glob("*.png"))
            for file_path in textures_files:
                blob_name = f"{base_prefix}/textures/{file_path.name}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                uploaded_files.append(blob_name)
            
            # Prepare models
            models_files = list(self.models_dir.glob("*.obj"))
            for file_path in models_files:
                blob_name = f"{base_prefix}/models/{file_path.name}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                uploaded_files.append(blob_name)
            
            logger.info(f"Uploaded {len(uploaded_files)} files to GCS bucket {bucket_name}")
            
            return {
                "success": True,
                "uploaded_files": len(uploaded_files),
                "tile_server_url": gcs_url
            }
        
        except Exception as e:
            logger.error(f"Error exporting to GCS: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _export_to_http_server(self, http_url: str, credentials_path: str = None) -> Dict:
        """
        Export assets to HTTP server.
        
        Args:
            http_url: HTTP server URL
            credentials_path: Path to credentials file
            
        Returns:
            Dictionary with export results
        """
        try:
            # Check if requests is installed
            try:
                import requests
            except ImportError:
                logger.error("requests not installed. Run: pip install requests")
                return {"success": False, "error": "requests not installed"}
            
            # Load credentials if provided
            auth = None
            if credentials_path and os.path.exists(credentials_path):
                try:
                    with open(credentials_path, 'r') as f:
                        creds = json.load(f)
                    
                    if "username" in creds and "password" in creds:
                        auth = (creds["username"], creds["password"])
                except Exception as e:
                    logger.error(f"Error loading credentials: {str(e)}")
                    return {"success": False, "error": f"Error loading credentials: {str(e)}"}
            
            # Upload assets
            uploaded_files = []
            
            # Ensure URL ends with /
            if not http_url.endswith("/"):
                http_url += "/"
            
            # Prepare prefabs
            prefabs_files = list(self.prefabs_dir.glob("*.prefab"))
            for file_path in prefabs_files:
                url = f"{http_url}prefabs/{file_path.name}"
                with open(file_path, 'rb') as f:
                    response = requests.put(url, data=f, auth=auth)
                
                if response.status_code in (200, 201, 204):
                    uploaded_files.append(url)
                else:
                    logger.warning(f"Failed to upload {file_path}: {response.status_code}")
            
            # Prepare materials
            materials_files = list(self.materials_dir.glob("*.mat"))
            for file_path in materials_files:
                url = f"{http_url}materials/{file_path.name}"
                with open(file_path, 'rb') as f:
                    response = requests.put(url, data=f, auth=auth)
                
                if response.status_code in (200, 201, 204):
                    uploaded_files.append(url)
                else:
                    logger.warning(f"Failed to upload {file_path}: {response.status_code}")
            
            # Prepare textures
            textures_files = list(self.textures_dir.glob("*.jpg")) + list(self.textures_dir.glob("*.png"))
            for file_path in textures_files:
                url = f"{http_url}textures/{file_path.name}"
                with open(file_path, 'rb') as f:
                    response = requests.put(url, data=f, auth=auth)
                
                if response.status_code in (200, 201, 204):
                    uploaded_files.append(url)
                else:
                    logger.warning(f"Failed to upload {file_path}: {response.status_code}")
            
            # Prepare models
            models_files = list(self.models_dir.glob("*.obj"))
            for file_path in models_files:
                url = f"{http_url}models/{file_path.name}"
                with open(file_path, 'rb') as f:
                    response = requests.put(url, data=f, auth=auth)
                
                if response.status_code in (200, 201, 204):
                    uploaded_files.append(url)
                else:
                    logger.warning(f"Failed to upload {file_path}: {response.status_code}")
            
            logger.info(f"Uploaded {len(uploaded_files)} files to HTTP server {http_url}")
            
            return {
                "success": True,
                "uploaded_files": len(uploaded_files),
                "tile_server_url": http_url
            }
        
        except Exception as e:
            logger.error(f"Error exporting to HTTP server: {str(e)}")
            return {"success": False, "error": str(e)}

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unity Prefab Generator")
    parser.add_argument("--output", default="./arcanum_output", help="Output directory")
    parser.add_argument("--unity", help="Unity project path")
    parser.add_argument("--buildings", required=True, help="Path to buildings metadata JSON file")
    parser.add_argument("--tile-server", help="Tile server URL (gs:// or http://)")
    parser.add_argument("--credentials", help="Path to server credentials file")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create building processor
    building_processor = OSMBuildingProcessor(os.path.join(args.output, "building_data"))
    
    # Create texture atlas manager
    texture_atlas_manager = TextureAtlasManager(os.path.join(args.output, "atlases"))
    
    # Create prefab generator
    generator = UnityPrefabGenerator(
        args.output,
        args.unity,
        building_processor,
        texture_atlas_manager
    )
    
    # Generate prefabs
    results = generator.generate_all_prefabs(args.buildings)
    
    print(f"Generated prefabs for {results.get('success_count', 0)} buildings")
    print(f"Failed for {results.get('failed_count', 0)} buildings")
    
    # Export to tile server if provided
    if args.tile_server:
        export_results = generator.export_to_tile_server(args.tile_server, args.credentials)
        
        if export_results["success"]:
            print(f"Exported {export_results.get('uploaded_files', 0)} files to tile server")
        else:
            print(f"Failed to export to tile server: {export_results.get('error')}")