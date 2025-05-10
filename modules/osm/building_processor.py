#!/usr/bin/env python3
"""
OSM Building Processor
-------------------
Processes OSM building data and applies UV mapping for texture application.
This module connects OSM building data with the TextureAtlasManager.
"""

import os
import sys
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import math
import hashlib
import random

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

# Import our modules
from modules.texture_atlas_manager import TextureAtlasManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class OSMBuildingProcessor:
    """
    Processes OSM building data, generates 3D models and applies UV mapping.
    """
    
    def __init__(self, 
                 output_dir: str,
                 texture_atlas_manager: TextureAtlasManager = None,
                 texture_dir: str = None,
                 default_height: float = 10.0):
        """
        Initialize the OSM building processor.
        
        Args:
            output_dir: Directory to save processed building data
            texture_atlas_manager: TextureAtlasManager instance
            texture_dir: Directory containing building textures
            default_height: Default building height if not specified in OSM data
        """
        self.output_dir = Path(output_dir)
        self.texture_dir = texture_dir if texture_dir else str(self.output_dir / "textures")
        self.default_height = default_height
        
        # Create texture atlas manager if not provided
        if texture_atlas_manager:
            self.texture_atlas_manager = texture_atlas_manager
        else:
            # Create a new texture atlas manager
            texture_output_dir = self.output_dir / "atlases"
            os.makedirs(texture_output_dir, exist_ok=True)
            self.texture_atlas_manager = TextureAtlasManager(str(texture_output_dir))
        
        # Create output directories
        self.models_dir = self.output_dir / "models"
        self.metadata_dir = self.output_dir / "metadata"
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.texture_dir, exist_ok=True)
        
        # Store building metadata for later use
        self.buildings_metadata = {}
        
        logger.info(f"OSMBuildingProcessor initialized with output dir: {output_dir}")
    
    def process_osm_buildings(self, buildings_file: str) -> Dict:
        """
        Process OSM buildings from JSON file.
        
        Args:
            buildings_file: Path to JSON file containing OSM buildings data
            
        Returns:
            Dictionary with processing results
        """
        # Load buildings
        try:
            with open(buildings_file, 'r') as f:
                buildings_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading buildings data: {str(e)}")
            return {"success": False, "error": str(e)}
        
        # Extract buildings
        buildings = buildings_data.get("features", [])
        
        # Process each building
        results = {
            "total": len(buildings),
            "success_count": 0,
            "failed_count": 0,
            "failed_buildings": []
        }
        
        for building in buildings:
            try:
                building_id = self._get_building_id(building)
                
                # Process the building
                building_result = self.process_building(building_id, building)
                
                if building_result["success"]:
                    results["success_count"] += 1
                else:
                    results["failed_count"] += 1
                    results["failed_buildings"].append(building_id)
            except Exception as e:
                logger.error(f"Error processing building: {str(e)}")
                results["failed_count"] += 1
                results["failed_buildings"].append("unknown")
        
        # Export building metadata
        metadata_path = self.metadata_dir / "buildings_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.buildings_metadata, f, indent=2)
            
            logger.info(f"Exported building metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error exporting building metadata: {str(e)}")
        
        return results
    
    def _get_building_id(self, building: Dict) -> str:
        """
        Get a unique ID for a building.
        
        Args:
            building: OSM building data
            
        Returns:
            Unique building ID
        """
        # Check if the building has an ID
        if "id" in building:
            return str(building["id"])
        
        # Check if properties has an ID
        if "properties" in building and "id" in building["properties"]:
            return str(building["properties"]["id"])
        
        # Create a hash of the geometry as ID
        if "geometry" in building:
            geometry_str = json.dumps(building["geometry"])
            return hashlib.md5(geometry_str.encode()).hexdigest()
        
        # Fallback to a random ID
        return f"building_{random.randint(1000000, 9999999)}"
    
    def process_building(self, building_id: str, building_data: Dict) -> Dict:
        """
        Process a single building.
        
        Args:
            building_id: Unique building ID
            building_data: OSM building data
            
        Returns:
            Dictionary with processing results
        """
        # Extract building properties
        properties = building_data.get("properties", {})
        geometry = building_data.get("geometry", {})
        
        # Skip if no geometry
        if not geometry or "coordinates" not in geometry:
            logger.warning(f"Building {building_id} has no valid geometry")
            return {"success": False, "error": "No valid geometry"}
        
        # Extract building height
        height = properties.get("height", self.default_height)
        
        # Try different height fields if not found
        if not height:
            height = properties.get("building:height", self.default_height)
        
        # Try to convert to float
        try:
            height = float(height)
        except (ValueError, TypeError):
            height = self.default_height
        
        # Extract building type/style
        building_type = properties.get("building", "yes")
        building_style = properties.get("building:architecture", "default")
        
        # Prepare building metadata
        building_metadata = {
            "id": building_id,
            "height": height,
            "type": building_type,
            "style": building_style,
            "levels": properties.get("building:levels", 1),
            "material": properties.get("building:material", "default")
        }
        
        # Calculate building footprint dimensions
        dimensions = self._calculate_building_dimensions(geometry)
        building_metadata.update(dimensions)
        
        # Store building metadata
        self.buildings_metadata[building_id] = building_metadata
        
        # Generate building model
        model_result = self._generate_building_model(building_id, geometry, building_metadata)
        
        if not model_result["success"]:
            return model_result
        
        # Assign texture and UV mapping
        texture_result = self._assign_building_texture(building_id, building_metadata)
        
        if not texture_result["success"]:
            return texture_result
        
        # Update building model with UV mapping
        uv_result = self._apply_uv_mapping_to_model(building_id, model_result["model_path"], texture_result)
        
        return {
            "success": uv_result["success"],
            "building_id": building_id,
            "model_path": model_result["model_path"],
            "texture_path": texture_result.get("atlas_path"),
            "uv_mapping": uv_result.get("uv_mapping")
        }
    
    def _calculate_building_dimensions(self, geometry: Dict) -> Dict:
        """
        Calculate building footprint dimensions.
        
        Args:
            geometry: OSM building geometry
            
        Returns:
            Dictionary with width, depth, and center coordinates
        """
        geo_type = geometry.get("type", "")
        coordinates = geometry.get("coordinates", [])
        
        if geo_type == "Polygon" and coordinates:
            # Extract outer ring
            outer_ring = coordinates[0]
            
            if len(outer_ring) < 3:
                return {"width": 10.0, "depth": 10.0, "center": [0, 0]}
            
            # Calculate min/max coordinates
            min_x = min(p[0] for p in outer_ring)
            max_x = max(p[0] for p in outer_ring)
            min_y = min(p[1] for p in outer_ring)
            max_y = max(p[1] for p in outer_ring)
            
            # Calculate dimensions (in meters)
            # This is a simplified calculation that doesn't account for projection distortion
            # For more accurate calculations, use a proper GIS library
            width = (max_x - min_x) * 111320 * math.cos(math.radians((min_y + max_y) / 2))
            depth = (max_y - min_y) * 110540
            
            # Ensure minimum size
            width = max(width, 3.0)
            depth = max(depth, 3.0)
            
            # Calculate center
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            return {
                "width": width,
                "depth": depth,
                "center": [center_x, center_y]
            }
        
        # Default dimensions
        return {"width": 10.0, "depth": 10.0, "center": [0, 0]}
    
    def _generate_building_model(self, building_id: str, geometry: Dict, metadata: Dict) -> Dict:
        """
        Generate a 3D model for a building.
        
        Args:
            building_id: Unique building ID
            geometry: OSM building geometry
            metadata: Building metadata
            
        Returns:
            Dictionary with model generation results
        """
        # Create a detailed 3D model for the building based on OSM data
        # Generate geometry with proper structure for Unity import
        
        model_path = self.models_dir / f"{building_id}.obj"
        
        try:
            # Get building dimensions
            height = metadata.get("height", 10.0)
            width = metadata.get("width", 10.0)
            depth = metadata.get("depth", 10.0)
            
            # Create a simple building model (box)
            with open(model_path, 'w') as f:
                # Write OBJ header
                f.write(f"# Building {building_id}\n")
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
                f.write(f"g building_{building_id}\n")
                
                # Faces (6 sides of the box, each with 2 triangles)
                # Each face includes vertex/texture/normal indices
                
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
                
                # Bottom face
                f.write(f"f 4/1/6 3/2/6 2/3/6\n")
                f.write(f"f 2/3/6 1/4/6 4/1/6\n")
            
            return {
                "success": True,
                "model_path": str(model_path),
                "vertices": 8,
                "faces": 12
            }
        
        except Exception as e:
            logger.error(f"Error generating building model for {building_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _assign_building_texture(self, building_id: str, metadata: Dict) -> Dict:
        """
        Assign a texture to a building using the TextureAtlasManager.
        
        Args:
            building_id: Unique building ID
            metadata: Building metadata
            
        Returns:
            Dictionary with texture assignment results
        """
        # Determine texture path based on building type and style
        building_type = metadata.get("type", "yes")
        building_style = metadata.get("style", "default")
        building_material = metadata.get("material", "default")
        
        # Create a deterministic texture filename based on building attributes
        texture_base = f"{building_type}_{building_style}_{building_material}"
        
        # Find a matching texture in the texture directory
        texture_path = None
        
        for ext in [".jpg", ".jpeg", ".png"]:
            # Try specific combination
            candidate = os.path.join(self.texture_dir, f"{texture_base}{ext}")
            if os.path.exists(candidate):
                texture_path = candidate
                break
                
            # Try by type
            candidate = os.path.join(self.texture_dir, f"{building_type}{ext}")
            if os.path.exists(candidate):
                texture_path = candidate
                break
                
            # Try by material
            candidate = os.path.join(self.texture_dir, f"{building_material}{ext}")
            if os.path.exists(candidate):
                texture_path = candidate
                break
        
        # Use default texture if no match found
        if not texture_path:
            default_texture = os.path.join(self.texture_dir, "default.jpg")
            
            # Create a default texture if it doesn't exist
            if not os.path.exists(default_texture):
                self._create_default_texture(default_texture)
            
            texture_path = default_texture
        
        # Assign texture to building using TextureAtlasManager
        result = self.texture_atlas_manager.assign_texture_to_building(building_id, texture_path, metadata)
        
        return result
    
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
    
    def _apply_uv_mapping_to_model(self, building_id: str, model_path: str, texture_result: Dict) -> Dict:
        """
        Apply UV mapping to a building model.
        
        Args:
            building_id: Unique building ID
            model_path: Path to the building model
            texture_result: Texture assignment results
            
        Returns:
            Dictionary with UV mapping results
        """
        # Get UV mapping from texture result
        uv_mapping = texture_result.get("uv_mapping", {})
        uv_sets = uv_mapping.get("uv_sets", {})
        
        # Get default UV set
        default_uv = uv_sets.get("default", [])
        
        if not default_uv:
            logger.warning(f"No UV mapping available for building {building_id}")
            return {"success": False, "error": "No UV mapping available"}
        
        # Open the model file
        try:
            with open(model_path, 'r') as f:
                model_lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading model file {model_path}: {str(e)}")
            return {"success": False, "error": str(e)}
        
        # Find texture coordinate lines
        vt_start = None
        vt_end = None
        
        for i, line in enumerate(model_lines):
            if line.startswith("vt "):
                if vt_start is None:
                    vt_start = i
                vt_end = i
        
        if vt_start is None or vt_end is None:
            logger.warning(f"No texture coordinates found in model {model_path}")
            return {"success": False, "error": "No texture coordinates found in model"}
        
        # Replace texture coordinates with UV mapping
        new_vt_lines = []
        for uv in default_uv:
            new_vt_lines.append(f"vt {uv[0]} {uv[1]}\n")
        
        # Replace the existing texture coordinates
        model_lines[vt_start:vt_end+1] = new_vt_lines
        
        # Write the updated model
        try:
            with open(model_path, 'w') as f:
                f.writelines(model_lines)
        except Exception as e:
            logger.error(f"Error writing updated model file {model_path}: {str(e)}")
            return {"success": False, "error": str(e)}
        
        return {
            "success": True,
            "uv_mapping": uv_mapping
        }

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OSM Building Processor")
    parser.add_argument("--output", default="./arcanum_output", help="Output directory")
    parser.add_argument("--buildings", required=True, help="Path to OSM buildings JSON file")
    parser.add_argument("--textures", help="Path to directory containing textures")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create texture directory if not provided
    texture_dir = args.textures if args.textures else os.path.join(args.output, "textures")
    os.makedirs(texture_dir, exist_ok=True)
    
    # Initialize texture atlas manager
    texture_atlas_manager = TextureAtlasManager(os.path.join(args.output, "atlases"))
    
    # Initialize building processor
    processor = OSMBuildingProcessor(args.output, texture_atlas_manager, texture_dir)
    
    # Process buildings
    results = processor.process_osm_buildings(args.buildings)
    
    print(f"Processed {results.get('total', 0)} buildings:")
    print(f"  - Success: {results.get('success_count', 0)}")
    print(f"  - Failed: {results.get('failed_count', 0)}")