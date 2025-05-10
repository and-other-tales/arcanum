#!/usr/bin/env python3
"""
Texture Atlas Manager
-------------------
This module handles the association between transformed textures and OSM building meshes,
creating texture atlases and UV mappings for 3D models.
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
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TextureAtlasManager:
    """
    Handles the creation and management of texture atlases for building meshes.
    Associates transformed textures with the appropriate building surfaces.
    """
    
    def __init__(self, 
                output_dir: str,
                atlas_size: Tuple[int, int] = (4096, 4096),
                building_texture_size: Tuple[int, int] = (1024, 1024),
                default_texture_path: str = None):
        """
        Initialize the texture atlas manager.
        
        Args:
            output_dir: Directory to save texture atlases
            atlas_size: Size of the texture atlas (width, height)
            building_texture_size: Default size for individual building textures
            default_texture_path: Path to default texture for buildings without matches
        """
        self.output_dir = Path(output_dir)
        self.atlas_size = atlas_size
        self.building_texture_size = building_texture_size
        self.default_texture_path = default_texture_path
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track atlas packing information
        self.atlases = []
        self.building_to_atlas_map = {}
        self.texture_to_atlas_map = {}
        
        # Cache for UV coordinates of buildings
        self.uv_mapping_cache = {}
        
        logger.info(f"TextureAtlasManager initialized with output dir: {output_dir}")
    
    def assign_texture_to_building(self, 
                                 building_id: str, 
                                 texture_path: str,
                                 building_metadata: Dict = None) -> Dict:
        """
        Assign a texture to a building and return UV mapping details.
        
        Args:
            building_id: Unique identifier for the building
            texture_path: Path to the texture for this building
            building_metadata: Additional building information (height, style, etc.)
            
        Returns:
            Dictionary with UV mapping information
        """
        # Check if texture exists
        if not os.path.exists(texture_path):
            logger.warning(f"Texture not found: {texture_path}")
            if self.default_texture_path and os.path.exists(self.default_texture_path):
                texture_path = self.default_texture_path
            else:
                logger.error(f"No default texture available for building {building_id}")
                return {"success": False, "error": "Texture not found"}
        
        # Check if we already have this texture in an atlas
        if texture_path in self.texture_to_atlas_map:
            atlas_info = self.texture_to_atlas_map[texture_path]
            self.building_to_atlas_map[building_id] = atlas_info
            
            # Calculate UV mapping for this building
            uv_mapping = self._calculate_uv_mapping(building_id, atlas_info, building_metadata)
            self.uv_mapping_cache[building_id] = uv_mapping
            
            return {
                "success": True,
                "atlas_id": atlas_info["atlas_id"],
                "atlas_path": atlas_info["atlas_path"],
                "uv_mapping": uv_mapping
            }
        
        # Find or create an atlas with space for this texture
        atlas_info = self._find_or_create_atlas(texture_path)
        
        if atlas_info["success"]:
            # Update maps
            self.texture_to_atlas_map[texture_path] = atlas_info
            self.building_to_atlas_map[building_id] = atlas_info
            
            # Calculate UV mapping for this building
            uv_mapping = self._calculate_uv_mapping(building_id, atlas_info, building_metadata)
            self.uv_mapping_cache[building_id] = uv_mapping
            
            return {
                "success": True,
                "atlas_id": atlas_info["atlas_id"],
                "atlas_path": atlas_info["atlas_path"],
                "uv_mapping": uv_mapping
            }
        else:
            logger.error(f"Failed to add texture to atlas for building {building_id}")
            return {"success": False, "error": "Failed to add texture to atlas"}
    
    def _find_or_create_atlas(self, texture_path: str) -> Dict:
        """
        Find an existing atlas with space or create a new one.
        
        Args:
            texture_path: Path to the texture to add
            
        Returns:
            Dictionary with atlas information
        """
        # Get texture size
        try:
            img = Image.open(texture_path)
            texture_size = img.size
        except Exception as e:
            logger.error(f"Error opening texture {texture_path}: {str(e)}")
            # Use default building texture size
            texture_size = self.building_texture_size
        
        # Check existing atlases for space
        for atlas in self.atlases:
            # Check if this atlas has space
            if self._has_space_for_texture(atlas, texture_size):
                # Add texture to this atlas
                result = self._add_texture_to_atlas(atlas, texture_path, texture_size)
                if result["success"]:
                    return result
        
        # No existing atlas has space, create a new one
        atlas_id = len(self.atlases)
        atlas_path = self.output_dir / f"atlas_{atlas_id}.jpg"
        
        # Create a new atlas
        new_atlas = {
            "atlas_id": atlas_id,
            "atlas_path": str(atlas_path),
            "size": self.atlas_size,
            "textures": [],
            "next_position": (0, 0),
            "rows": []
        }
        
        # Add the texture to the new atlas
        result = self._add_texture_to_atlas(new_atlas, texture_path, texture_size)
        
        if result["success"]:
            # Add new atlas to the list
            self.atlases.append(new_atlas)
            return result
        else:
            logger.error(f"Failed to add texture to new atlas: {result.get('error')}")
            return result
    
    def _has_space_for_texture(self, atlas: Dict, texture_size: Tuple[int, int]) -> bool:
        """
        Check if the atlas has space for a texture of the given size.
        
        Args:
            atlas: Atlas information dictionary
            texture_size: Size of the texture (width, height)
            
        Returns:
            True if the atlas has space, False otherwise
        """
        next_x, next_y = atlas["next_position"]
        
        # Check if we can add the texture to the current row
        if next_x + texture_size[0] <= atlas["size"][0]:
            return True
        
        # Check if we can start a new row
        row_height = 0
        if atlas["rows"]:
            row_height = atlas["rows"][-1]["height"]
        
        next_row_y = next_y + row_height
        
        if next_row_y + texture_size[1] <= atlas["size"][1]:
            return True
        
        # No space in this atlas
        return False
    
    def _add_texture_to_atlas(self, atlas: Dict, texture_path: str, texture_size: Tuple[int, int]) -> Dict:
        """
        Add a texture to the atlas and update the atlas image.
        
        Args:
            atlas: Atlas information dictionary
            texture_path: Path to the texture
            texture_size: Size of the texture (width, height)
            
        Returns:
            Dictionary with result information
        """
        next_x, next_y = atlas["next_position"]
        
        # Check if we can add the texture to the current row
        if next_x + texture_size[0] <= atlas["size"][0]:
            # Add to current row
            position = (next_x, next_y)
            
            # Update next position
            atlas["next_position"] = (next_x + texture_size[0], next_y)
            
            # Update rows if needed
            if not atlas["rows"]:
                atlas["rows"].append({
                    "y": next_y,
                    "height": texture_size[1]
                })
            elif atlas["rows"][-1]["height"] < texture_size[1]:
                atlas["rows"][-1]["height"] = texture_size[1]
                
        else:
            # Start a new row
            row_height = 0
            if atlas["rows"]:
                row_height = atlas["rows"][-1]["height"]
            
            next_row_y = next_y + row_height
            
            if next_row_y + texture_size[1] <= atlas["size"][1]:
                # Can start a new row
                position = (0, next_row_y)
                
                # Update next position
                atlas["next_position"] = (texture_size[0], next_row_y)
                
                # Add a new row
                atlas["rows"].append({
                    "y": next_row_y,
                    "height": texture_size[1]
                })
            else:
                # No space in this atlas
                return {
                    "success": False, 
                    "error": f"No space in atlas {atlas['atlas_id']} for texture {texture_path}"
                }
        
        # Add texture to atlas with detailed metadata for efficient rendering
        texture_info = {
            "texture_path": texture_path,
            "position": position,
            "size": texture_size,
            "format": "RGB" if texture_size[0] > 512 else "DXT1",  # Use compression for smaller textures
            "filtering": "bilinear",                               # Texture filtering mode
            "mip_levels": max(1, int(math.log2(min(texture_size)))), # Calculate appropriate mip levels
            "hash": hashlib.md5(open(texture_path, 'rb').read()).hexdigest()[:8] # For texture identification and caching
        }
        atlas["textures"].append(texture_info)
        
        # Update the actual atlas image
        self._update_atlas_image(atlas)
        
        # Create result
        return {
            "success": True,
            "atlas_id": atlas["atlas_id"],
            "atlas_path": atlas["atlas_path"],
            "position": position,
            "size": texture_size
        }
    
    def _update_atlas_image(self, atlas: Dict) -> bool:
        """
        Update the atlas image by compositing all textures.
        
        Args:
            atlas: Atlas information dictionary
            
        Returns:
            True if successful, False otherwise
        """
        # Create a new atlas image if it doesn't exist
        if not os.path.exists(atlas["atlas_path"]):
            try:
                # Create a blank atlas image
                atlas_img = Image.new("RGB", atlas["size"], (0, 0, 0))
                atlas_img.save(atlas["atlas_path"])
            except Exception as e:
                logger.error(f"Error creating atlas image: {str(e)}")
                return False
        
        try:
            # Open the existing atlas
            atlas_img = Image.open(atlas["atlas_path"])
            
            # Get the last added texture (we only need to add this one)
            if atlas["textures"]:
                texture_info = atlas["textures"][-1]
                
                # Open the texture
                texture_img = Image.open(texture_info["texture_path"])
                
                # Resize if needed
                if texture_img.size != texture_info["size"]:
                    texture_img = texture_img.resize(texture_info["size"], Image.LANCZOS)
                
                # Paste the texture onto the atlas
                atlas_img.paste(texture_img, texture_info["position"])
                
                # Save the updated atlas
                atlas_img.save(atlas["atlas_path"])
                
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error updating atlas image: {str(e)}")
            return False
    
    def _calculate_uv_mapping(self, building_id: str, atlas_info: Dict, building_metadata: Dict = None) -> Dict:
        """
        Calculate UV mapping coordinates for a building.
        
        Args:
            building_id: Unique identifier for the building
            atlas_info: Atlas information for the texture
            building_metadata: Additional building information
            
        Returns:
            Dictionary with UV mapping information
        """
        # Get atlas size
        atlas_id = atlas_info["atlas_id"]
        atlas_path = atlas_info["atlas_path"]
        atlas_width, atlas_height = self.atlas_size
        
        # Get texture position and size in the atlas
        position = atlas_info["position"]
        size = atlas_info["size"]
        
        # Calculate normalized UV coordinates
        # UV coordinates are normalized to 0-1
        u_min = position[0] / atlas_width
        v_min = position[1] / atlas_height
        u_max = (position[0] + size[0]) / atlas_width
        v_max = (position[1] + size[1]) / atlas_height
        
        # Create different UV mappings based on building type/metadata
        uv_sets = {}
        
        # Default UV mapping (simple quad)
        uv_sets["default"] = [
            (u_min, v_min),  # Bottom left
            (u_max, v_min),  # Bottom right
            (u_max, v_max),  # Top right
            (u_min, v_max)   # Top left
        ]
        
        # If we have building metadata, create more specific UV mappings
        if building_metadata:
            # Get building height for facade mapping
            building_height = building_metadata.get("height", 10.0)
            building_width = building_metadata.get("width", 10.0)
            building_depth = building_metadata.get("depth", 10.0)
            
            # Calculate aspect ratios for proper texture tiling
            width_ratio = building_width / 10.0
            height_ratio = building_height / 10.0
            depth_ratio = building_depth / 10.0
            
            # Create UV mappings for different building sides
            # Front facade
            uv_sets["front"] = [
                (u_min, v_min),                   # Bottom left
                (u_min + width_ratio, v_min),     # Bottom right
                (u_min + width_ratio, v_max),     # Top right
                (u_min, v_max)                    # Top left
            ]
            
            # Back facade
            uv_sets["back"] = [
                (u_min, v_min),                   # Bottom left
                (u_min + width_ratio, v_min),     # Bottom right
                (u_min + width_ratio, v_max),     # Top right
                (u_min, v_max)                    # Top left
            ]
            
            # Left side
            uv_sets["left"] = [
                (u_min, v_min),                   # Bottom left
                (u_min + depth_ratio, v_min),     # Bottom right
                (u_min + depth_ratio, v_max),     # Top right
                (u_min, v_max)                    # Top left
            ]
            
            # Right side
            uv_sets["right"] = [
                (u_min, v_min),                   # Bottom left
                (u_min + depth_ratio, v_min),     # Bottom right
                (u_min + depth_ratio, v_max),     # Top right
                (u_min, v_max)                    # Top left
            ]
            
            # Roof
            uv_sets["roof"] = [
                (u_min, v_min),                   # Bottom left
                (u_min + width_ratio, v_min),     # Bottom right
                (u_min + width_ratio, v_min + depth_ratio),  # Top right
                (u_min, v_min + depth_ratio)      # Top left
            ]
        
        return {
            "atlas_id": atlas_id,
            "atlas_path": atlas_path,
            "uv_sets": uv_sets
        }
    
    def get_building_uv_mapping(self, building_id: str) -> Optional[Dict]:
        """
        Get the UV mapping for a building.
        
        Args:
            building_id: Unique identifier for the building
            
        Returns:
            Dictionary with UV mapping information or None if not found
        """
        if building_id in self.uv_mapping_cache:
            return self.uv_mapping_cache[building_id]
        
        return None
    
    def export_uv_mappings(self, output_path: str) -> bool:
        """
        Export all UV mappings to a JSON file.
        
        Args:
            output_path: Path to save the UV mappings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.uv_mapping_cache, f, indent=2)
            
            logger.info(f"Exported UV mappings to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting UV mappings: {str(e)}")
            return False
    
    def process_buildings_batch(self, 
                             buildings: List[Dict], 
                             textures_dir: str,
                             texture_naming_pattern: str = "{building_id}.jpg") -> Dict:
        """
        Process a batch of buildings, assigning textures and calculating UV mappings.
        
        Args:
            buildings: List of building dictionaries (must have 'id' field)
            textures_dir: Directory containing textures
            texture_naming_pattern: Pattern for texture filenames
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "total": len(buildings),
            "success_count": 0,
            "failed_count": 0,
            "failed_buildings": []
        }
        
        for building in buildings:
            building_id = building.get("id")
            
            if not building_id:
                logger.warning(f"Building without ID: {building}")
                results["failed_count"] += 1
                results["failed_buildings"].append("unknown")
                continue
            
            # Determine texture path
            texture_filename = texture_naming_pattern.format(building_id=building_id)
            texture_path = os.path.join(textures_dir, texture_filename)
            
            # Assign texture to building
            result = self.assign_texture_to_building(building_id, texture_path, building)
            
            if result["success"]:
                results["success_count"] += 1
            else:
                results["failed_count"] += 1
                results["failed_buildings"].append(building_id)
        
        return results

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Texture Atlas Manager")
    parser.add_argument("--output", default="./atlases", help="Output directory for texture atlases")
    parser.add_argument("--buildings", help="Path to buildings JSON file")
    parser.add_argument("--textures", help="Path to directory containing textures")
    args = parser.parse_args()
    
    # Create texture atlas manager
    manager = TextureAtlasManager(args.output)
    
    if args.buildings and args.textures:
        # Load buildings
        with open(args.buildings, 'r') as f:
            buildings = json.load(f)
        
        # Process buildings
        results = manager.process_buildings_batch(buildings, args.textures)
        
        print(f"Processed {results['total']} buildings:")
        print(f"  - Success: {results['success_count']}")
        print(f"  - Failed: {results['failed_count']}")
        
        # Export UV mappings
        manager.export_uv_mappings(os.path.join(args.output, "uv_mappings.json"))
    else:
        print("Please provide buildings and textures paths to process")