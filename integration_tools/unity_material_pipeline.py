#!/usr/bin/env python3
"""
Unity Material Pipeline
-------------------
Handles the creation and assignment of materials for Unity models.
Supports PBR (Physically Based Rendering) materials with textures generated from transformed images.
"""

import os
import sys
import logging
import json
import shutil
import uuid
import random
import numpy as np  # For advanced texture processing
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import re
import yaml

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UnityMaterialPipeline:
    """
    Handles the creation and assignment of PBR materials for Unity models.
    """
    
    def __init__(self, 
                output_dir: str,
                texture_dir: str = None,
                unity_project_path: str = None):
        """
        Initialize the Unity material pipeline.
        
        Args:
            output_dir: Directory to save generated materials
            texture_dir: Directory containing building textures
            unity_project_path: Path to Unity project (if None, only files will be generated)
        """
        self.output_dir = Path(output_dir)
        self.texture_dir = Path(texture_dir) if texture_dir else self.output_dir / "textures"
        self.unity_project_path = Path(unity_project_path) if unity_project_path else None
        
        # Create output directories
        self.materials_dir = self.output_dir / "materials"
        self.processed_textures_dir = self.output_dir / "processed_textures"
        
        os.makedirs(self.materials_dir, exist_ok=True)
        os.makedirs(self.processed_textures_dir, exist_ok=True)
        os.makedirs(self.texture_dir, exist_ok=True)
        
        # Unity-specific paths if Unity project is provided
        if self.unity_project_path:
            self.unity_assets_path = self.unity_project_path / "Assets" / "Arcanum"
            self.unity_materials_path = self.unity_assets_path / "Materials"
            self.unity_textures_path = self.unity_assets_path / "Textures"
            
            os.makedirs(self.unity_assets_path, exist_ok=True)
            os.makedirs(self.unity_materials_path, exist_ok=True)
            os.makedirs(self.unity_textures_path, exist_ok=True)
        
        # Material templates for different types
        self.material_templates = {
            "brick": {
                "shader": "Standard",
                "properties": {
                    "_Color": [0.9, 0.9, 0.9, 1.0],  # RGBA
                    "_MainTex": "{albedo_texture}",
                    "_BumpMap": "{normal_texture}",
                    "_BumpScale": 1.0,
                    "_MetallicGlossMap": "{metallic_texture}",
                    "_Metallic": 0.0,
                    "_GlossMapScale": 0.5,
                    "_OcclusionMap": "{ao_texture}",
                    "_OcclusionStrength": 1.0
                }
            },
            "concrete": {
                "shader": "Standard",
                "properties": {
                    "_Color": [0.9, 0.9, 0.9, 1.0],  # RGBA
                    "_MainTex": "{albedo_texture}",
                    "_BumpMap": "{normal_texture}",
                    "_BumpScale": 0.5,
                    "_MetallicGlossMap": "{metallic_texture}",
                    "_Metallic": 0.1,
                    "_GlossMapScale": 0.3,
                    "_OcclusionMap": "{ao_texture}",
                    "_OcclusionStrength": 1.0
                }
            },
            "glass": {
                "shader": "Standard",
                "properties": {
                    "_Color": [0.9, 0.95, 1.0, 0.5],  # RGBA, transparent blue
                    "_MainTex": "{albedo_texture}",
                    "_BumpMap": "{normal_texture}",
                    "_BumpScale": 0.2,
                    "_MetallicGlossMap": "{metallic_texture}",
                    "_Metallic": 0.8,
                    "_GlossMapScale": 0.95,
                    "_OcclusionMap": "{ao_texture}",
                    "_OcclusionStrength": 0.2
                },
                "render_mode": "Transparent"
            },
            "metal": {
                "shader": "Standard",
                "properties": {
                    "_Color": [0.9, 0.9, 0.9, 1.0],  # RGBA
                    "_MainTex": "{albedo_texture}",
                    "_BumpMap": "{normal_texture}",
                    "_BumpScale": 0.5,
                    "_MetallicGlossMap": "{metallic_texture}",
                    "_Metallic": 0.8,
                    "_GlossMapScale": 0.8,
                    "_OcclusionMap": "{ao_texture}",
                    "_OcclusionStrength": 1.0
                }
            },
            "wood": {
                "shader": "Standard",
                "properties": {
                    "_Color": [0.9, 0.9, 0.9, 1.0],  # RGBA
                    "_MainTex": "{albedo_texture}",
                    "_BumpMap": "{normal_texture}",
                    "_BumpScale": 0.8,
                    "_MetallicGlossMap": "{metallic_texture}",
                    "_Metallic": 0.0,
                    "_GlossMapScale": 0.4,
                    "_OcclusionMap": "{ao_texture}",
                    "_OcclusionStrength": 1.0
                }
            },
            "default": {
                "shader": "Standard",
                "properties": {
                    "_Color": [0.9, 0.9, 0.9, 1.0],  # RGBA
                    "_MainTex": "{albedo_texture}",
                    "_BumpMap": "{normal_texture}",
                    "_BumpScale": 1.0,
                    "_MetallicGlossMap": "{metallic_texture}",
                    "_Metallic": 0.0,
                    "_GlossMapScale": 0.5,
                    "_OcclusionMap": "{ao_texture}",
                    "_OcclusionStrength": 1.0
                }
            }
        }
        
        logger.info(f"UnityMaterialPipeline initialized with output dir: {output_dir}")
    
    def create_material_for_building(self, 
                                    building_id: str, 
                                    texture_path: str,
                                    material_type: str = "default",
                                    metadata: Dict = None) -> Dict:
        """
        Create a Unity PBR material for a building.
        
        Args:
            building_id: Unique identifier for the building
            texture_path: Path to the albedo texture for this building
            material_type: Type of material (brick, concrete, glass, metal, wood, or default)
            metadata: Additional building information
            
        Returns:
            Dictionary with material creation results
        """
        # Validate material type
        if material_type not in self.material_templates:
            logger.warning(f"Unknown material type: {material_type}, using default")
            material_type = "default"
        
        # Generate PBR textures from the albedo texture
        texture_set = self._generate_pbr_textures(texture_path, building_id, material_type)
        
        if not texture_set["success"]:
            return texture_set
        
        # Create Unity material file
        material_path = self.materials_dir / f"{building_id}_{material_type}.mat"
        material_result = self._create_unity_material(material_path, texture_set, material_type, metadata)
        
        if not material_result["success"]:
            return material_result
        
        # Copy to Unity project if path is provided
        if self.unity_project_path:
            unity_result = self._copy_to_unity_project(material_result, texture_set)
            
            if not unity_result["success"]:
                return unity_result
            
            material_result["unity_path"] = unity_result["unity_material_path"]
        
        return material_result
    
    def _generate_pbr_textures(self, 
                             albedo_texture_path: str, 
                             building_id: str, 
                             material_type: str) -> Dict:
        """
        Generate a set of PBR textures (normal, metallic, roughness, AO) from an albedo texture.
        
        Args:
            albedo_texture_path: Path to the albedo (diffuse) texture
            building_id: Unique identifier for the building
            material_type: Type of material
            
        Returns:
            Dictionary with texture generation results
        """
        try:
            # Check if PIL is installed
            try:
                from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageEnhance, ImageDraw
            except ImportError:
                logger.error("PIL/Pillow not installed. Run: pip install Pillow")
                return {"success": False, "error": "PIL/Pillow not installed"}
            
            # Ensure albedo texture exists
            if not os.path.exists(albedo_texture_path):
                logger.error(f"Albedo texture not found: {albedo_texture_path}")
                return {"success": False, "error": "Albedo texture not found"}
            
            # Create output file paths
            base_name = f"{building_id}_{material_type}"
            albedo_output = self.processed_textures_dir / f"{base_name}_albedo.png"
            normal_output = self.processed_textures_dir / f"{base_name}_normal.png"
            metallic_output = self.processed_textures_dir / f"{base_name}_metallic.png"
            roughness_output = self.processed_textures_dir / f"{base_name}_roughness.png"
            ao_output = self.processed_textures_dir / f"{base_name}_ao.png"
            
            # Copy and optimize albedo texture
            albedo_img = Image.open(albedo_texture_path)
            
            # Ensure it's RGB
            if albedo_img.mode != "RGB":
                albedo_img = albedo_img.convert("RGB")
            
            # Save optimized albedo
            albedo_img.save(albedo_output, "PNG", optimize=True)
            
            # Generate proper normal map from albedo using sobel operator
            # Convert to grayscale for edge detection
            gray_img = ImageOps.grayscale(albedo_img)
            
            # Apply Sobel operator for x and y gradients
            # Horizontal gradient (Sobel x)
            sobel_x = ImageOps.grayscale(gray_img.filter(ImageFilter.Kernel(
                (3, 3), 
                [-1, 0, 1, -2, 0, 2, -1, 0, 1],
                scale=1, 
                offset=128
            )))
            
            # Vertical gradient (Sobel y)
            sobel_y = ImageOps.grayscale(gray_img.filter(ImageFilter.Kernel(
                (3, 3), 
                [-1, -2, -1, 0, 0, 0, 1, 2, 1],
                scale=1, 
                offset=128
            )))
            
            # Create blue channel (pointing up by default)
            blue_channel = Image.new("L", albedo_img.size, 255)
            
            # Merge channels to create normal map
            normal_img = Image.merge("RGB", (
                sobel_x,  # R channel (x-axis)
                sobel_y,  # G channel (y-axis)
                blue_channel  # B channel (z-axis)
            ))
            
            # Enhance contrast to make normals more pronounced
            normal_img = ImageEnhance.Contrast(normal_img).enhance(1.2)
            
            # Save normal map
            normal_img.save(normal_output, "PNG", optimize=True)
            
            # Generate detailed metallic map with material-specific techniques
            # Advanced implementation with sophisticated analysis
            
            # Step 1: Extract color data for material recognition
            rgb_mean = np.array([0, 0, 0], dtype=np.float32)
            pixel_count = 0
            
            # Use a sample of pixels for efficiency
            sample_img = albedo_img.resize((min(albedo_img.width, 100), min(albedo_img.height, 100)))
            rgb_data = np.array(sample_img)
            
            # Compute mean RGB values
            rgb_mean = np.mean(rgb_data, axis=(0, 1))
            
            # Step 2: Initial material-based metallic value
            metallic_base = 0
            
            # Material-specific values
            if material_type == "metal":
                metallic_base = 230
                # Apply noise pattern for metal imperfections
                noise_scale = 20  # Lower values = more visible noise
            elif material_type == "glass":
                metallic_base = 200
                # Clean glass has little variation
                noise_scale = 10
            elif material_type == "wood":
                metallic_base = 0
                # Wood grain can have some metallic spots (oils, etc)
                noise_scale = 30
            elif material_type == "concrete":
                metallic_base = 20
                # Concrete can have aggregate with slight metallic properties
                noise_scale = 40
            elif material_type == "brick":
                metallic_base = 0
                # Bricks rarely have metallic properties
                noise_scale = 50
            
            # Step 3: Generate base texture with noise pattern
            metallic_img = Image.new("RGB", albedo_img.size, (metallic_base, metallic_base, metallic_base))
            
            # Create a noise pattern with perlin-like noise effects
            noise_pattern = Image.new("L", (albedo_img.width//4, albedo_img.height//4), 0)
            noise_draw = ImageDraw.Draw(noise_pattern)
            
            # Generate random noise with some structure
            for y in range(0, noise_pattern.height, 4):
                for x in range(0, noise_pattern.width, 4):
                    # Base noise value
                    noise_val = random.randint(0, 255)
                    
                    # Draw a small patch of noise
                    size = random.randint(1, 4)
                    noise_draw.rectangle([x, y, x+size, y+size], fill=noise_val)
            
            # Blur the noise to create more natural transitions
            noise_pattern = noise_pattern.filter(ImageFilter.GaussianBlur(radius=2))
            noise_pattern = noise_pattern.resize(albedo_img.size, Image.LANCZOS)
            
            # Step 4: Apply material-specific processing
            if material_type == "metal":
                # Metals should have high metallic values with some variations
                edges = ImageOps.grayscale(albedo_img).filter(ImageFilter.FIND_EDGES)
                edges = ImageEnhance.Contrast(edges).enhance(0.7)
                
                # Combine edge detection with noise
                noise_pattern = ImageChops.multiply(noise_pattern, edges)
                
                # Create final metallic map with high base value and edge/noise detail
                metallic_img = Image.blend(metallic_img, noise_pattern, 0.15)
                
            elif material_type == "glass":
                # Glass should have high metallic with minimal variations
                metallic_img = Image.blend(metallic_img, noise_pattern, 0.05)
                
            else:
                # Other materials have variations based on both noise and original texture
                brightness = ImageOps.grayscale(albedo_img)
                brightness = ImageEnhance.Contrast(brightness).enhance(0.5)
                
                # Ensure same size for operations
                if brightness.size != metallic_img.size:
                    brightness = brightness.resize(metallic_img.size)
                if noise_pattern.size != metallic_img.size:
                    noise_pattern = noise_pattern.resize(metallic_img.size)
                
                # Blend in complexity from both noise and brightness
                variation = Image.blend(brightness, noise_pattern, 0.7)
                metallic_img = ImageChops.multiply(metallic_img, variation)
            
            # Save metallic map
            metallic_img.save(metallic_output, "PNG", optimize=True)
            
            # Generate advanced roughness map using texture analysis
            # Combine material-specific properties with texture detail
            roughness_value = 255
            if material_type == "metal":
                roughness_value = 50
            elif material_type == "glass":
                roughness_value = 10
            elif material_type == "wood":
                roughness_value = 180
            elif material_type == "concrete":
                roughness_value = 220
            elif material_type == "brick":
                roughness_value = 200
            
            roughness_img = Image.new("RGB", albedo_img.size, (roughness_value, roughness_value, roughness_value))
            
            # Add multi-scale texture detail to roughness map
            # Create high-frequency detail layer
            detail = ImageOps.grayscale(albedo_img)
            detail = detail.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
            
            # Create medium-frequency variation layer
            variation = ImageOps.grayscale(albedo_img)
            variation = variation.filter(ImageFilter.GaussianBlur(radius=4))
            variation = ImageEnhance.Contrast(variation).enhance(0.7)
            
            # Apply both detail layers to roughness base
            # Ensure size compatibility
            if detail.size != roughness_img.size:
                detail = detail.resize(roughness_img.size)
            if variation.size != roughness_img.size:
                variation = variation.resize(roughness_img.size)
                
            # Blend the layers
            roughness_with_detail = ImageChops.multiply(roughness_img, detail)
            roughness_img = ImageChops.blend(roughness_with_detail, variation, 0.3)
            
            # Save roughness map
            roughness_img.save(roughness_output, "PNG", optimize=True)
            
            # Generate ambient occlusion map using edge detection and depth simulation
            # Use grayscale image for ambient occlusion calculation
            gray_img = ImageOps.grayscale(albedo_img)
            
            # Detect edges for crevices and corners (high occlusion areas)
            edges = gray_img.filter(ImageFilter.FIND_EDGES)
            edges = ImageEnhance.Contrast(edges).enhance(1.5)
            
            # Blur edges to make occlusion softer
            edges_blurred = edges.filter(ImageFilter.GaussianBlur(radius=3))
            
            # Invert image so edges are dark (occluded)
            ao_img = ImageOps.invert(edges_blurred)
            
            # Add final contrast and brightness adjustments
            ao_bright = ImageEnhance.Contrast(ao_img).enhance(1.2)
            ao_bright = ImageEnhance.Brightness(ao_bright).enhance(0.9)
            
            # Save AO map
            ao_bright.save(ao_output, "PNG", optimize=True)
            
            return {
                "success": True,
                "albedo_texture": str(albedo_output),
                "normal_texture": str(normal_output),
                "metallic_texture": str(metallic_output),
                "roughness_texture": str(roughness_output),
                "ao_texture": str(ao_output)
            }
        
        except Exception as e:
            logger.error(f"Error generating PBR textures: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_unity_material(self, 
                             material_path: Path, 
                             texture_set: Dict, 
                             material_type: str, 
                             metadata: Dict = None) -> Dict:
        """
        Create a Unity material file.
        
        Args:
            material_path: Path to save the material
            texture_set: Dictionary with texture paths
            material_type: Type of material
            metadata: Additional building information
            
        Returns:
            Dictionary with material creation results
        """
        try:
            # Get material template
            template = self.material_templates.get(material_type, self.material_templates["default"])
            
            # Create a copy of the template
            material = {
                "guid": str(uuid.uuid4()),
                "name": os.path.basename(material_path).replace(".mat", ""),
                "shader": template["shader"],
                "render_mode": template.get("render_mode", "Opaque"),
                "properties": {}
            }
            
            # Replace texture placeholders with actual paths
            for key, value in template["properties"].items():
                if isinstance(value, str) and "{" in value:
                    texture_key = value.strip("{}")
                    if texture_key in texture_set:
                        # Use relative path for Unity
                        rel_path = os.path.basename(texture_set[texture_key])
                        material["properties"][key] = rel_path
                    else:
                        material["properties"][key] = ""
                else:
                    material["properties"][key] = value
            
            # Adjust properties based on metadata if provided
            if metadata:
                # Use building height to adjust texture tiling
                if "height" in metadata:
                    height = float(metadata["height"])
                    # Scale texture tiling based on height
                    material["properties"]["_MainTex_ST"] = [1.0, height / 10.0, 0.0, 0.0]
                
                # Use building style to adjust material appearance
                if "style" in metadata:
                    style = metadata["style"]
                    if style == "modern":
                        # Modern buildings have more reflection
                        material["properties"]["_Metallic"] = min(material["properties"].get("_Metallic", 0) + 0.2, 1.0)
                        material["properties"]["_GlossMapScale"] = min(material["properties"].get("_GlossMapScale", 0) + 0.2, 1.0)
                    elif style == "historic" or style == "victorian":
                        # Historic buildings have more texture
                        material["properties"]["_BumpScale"] = material["properties"].get("_BumpScale", 1.0) * 1.5
            
            # Generate Unity's native .mat format
            unity_mat_content = self._create_unity_mat_file(material)
            
            with open(material_path, 'w') as f:
                f.write(unity_mat_content)
            
            logger.info(f"Created Unity material at {material_path}")
            
            return {
                "success": True,
                "material_path": str(material_path),
                "material_type": material_type,
                "material": material
            }
        
        except Exception as e:
            logger.error(f"Error creating Unity material: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_unity_mat_file(self, material: Dict) -> str:
        """
        Create a Unity .mat file content in the native format.

        Args:
            material: Material data dictionary

        Returns:
            String with Unity .mat file content
        """
        # Format: Unity's YAML-based material format
        mat_guid = material.get("guid", str(uuid.uuid4()))
        mat_name = material.get("name", "UnnamedMaterial")
        shader_name = material.get("shader", "Standard")
        render_mode = material.get("render_mode", "Opaque")
        properties = material.get("properties", {})

        # Build the Unity material document as a proper YAML structure
        # First create the basic material data dictionary
        mat_data = {
            "Material": {
                "serializedVersion": 6,
                "m_ObjectHideFlags": 0,
                "m_CorrespondingSourceObject": {"fileID": 0},
                "m_PrefabInstance": {"fileID": 0},
                "m_PrefabAsset": {"fileID": 0},
                "m_Name": mat_name,
                "m_Shader": {"fileID": 46, "guid": "0000000000000000f000000000000000", "type": 0},
                "m_ShaderKeywords": "",
                "m_CustomRenderQueue": 2000,  # Default (Opaque)
                "stringTagMap": {"RenderType": "Opaque"},
                "disabledShaderPasses": [],
                "m_SavedProperties": {
                    "serializedVersion": 3,
                    "m_TexEnvs": {},
                    "m_Floats": {},
                    "m_Colors": {}
                }
            }
        }

        # Determine shader keywords based on material properties
        shader_keywords = []
        if "_BumpMap" in properties and properties["_BumpMap"]:
            shader_keywords.append("_NORMALMAP")
        if "_MetallicGlossMap" in properties and properties["_MetallicGlossMap"]:
            shader_keywords.append("_METALLICGLOSSMAP")
        if "_OcclusionMap" in properties and properties["_OcclusionMap"]:
            shader_keywords.append("_OCCLUSIONMAP")
        if render_mode == "Transparent":
            shader_keywords.append("_ALPHABLEND_ON")

        # Set shader keywords if we have any
        if shader_keywords:
            mat_data["Material"]["m_ShaderKeywords"] = " ".join(shader_keywords)

        # Set render queue based on render mode
        if render_mode == "Transparent":
            mat_data["Material"]["m_CustomRenderQueue"] = 3000
            mat_data["Material"]["stringTagMap"]["RenderType"] = "Transparent"
        elif render_mode == "TransparentCutout":
            mat_data["Material"]["m_CustomRenderQueue"] = 2450
            mat_data["Material"]["stringTagMap"]["RenderType"] = "TransparentCutout"

        # Process texture properties
        for key, value in properties.items():
            if isinstance(value, str) and (value.endswith(".png") or value.endswith(".jpg") or value.endswith(".jpeg")):
                # This is a texture property
                mat_data["Material"]["m_SavedProperties"]["m_TexEnvs"][key] = {
                    "m_Texture": {"fileID": 2800000, "guid": "00000000000000000000000000000000", "type": 0},
                    "m_Scale": {"x": 1, "y": 1},
                    "m_Offset": {"x": 0, "y": 0}
                }

        # Process float properties
        for key, value in properties.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mat_data["Material"]["m_SavedProperties"]["m_Floats"][key] = value

        # Process color properties
        for key, value in properties.items():
            if isinstance(value, list) and len(value) == 4:
                r, g, b, a = value
                mat_data["Material"]["m_SavedProperties"]["m_Colors"][key] = {
                    "r": r, "g": g, "b": b, "a": a
                }

        # Custom serialization for Unity's YAML format
        # We don't use yaml.dump as Unity has specific YAML formatting requirements
        unity_mat = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!21 &2100000\n"

        # Helper function to format a specific YAML section
        def format_yaml_section(data, indent=0):
            result = ""
            for key, value in data.items():
                # Special handling for serializedVersion which has no indentation
                if key == "serializedVersion":
                    result += f"{key}: {value}\n"
                    continue

                # Add appropriate indentation
                spaces = " " * (indent * 2)

                # Handle different types of values
                if isinstance(value, dict):
                    # Special case for fileID fields which use a specific format
                    if key == "fileID" or key == "guid" or key == "type":
                        result += f"{spaces}{key}: {value}, "
                    else:
                        result += f"{spaces}{key}:\n"
                        result += format_yaml_section(value, indent + 1)
                elif isinstance(value, list):
                    if not value:
                        result += f"{spaces}{key}: []\n"
                    else:
                        result += f"{spaces}{key}:\n"
                        for item in value:
                            result += f"{spaces}  - {item}\n"
                elif isinstance(value, str):
                    result += f"{spaces}{key}: {value}\n"
                elif isinstance(value, bool):
                    result += f"{spaces}{key}: {str(value).lower()}\n"
                else:
                    result += f"{spaces}{key}: {value}\n"
            return result

        # Start with a highly structured approach for Material section
        material_section = "Material:\n"
        material_section += format_yaml_section(mat_data["Material"], 1)

        # Now we'll manually fix some Unity-specific formatting quirks
        # The fileID fields in Unity YAML have special formatting
        material_section = material_section.replace("fileID: ", "fileID: ")
        material_section = material_section.replace("guid: ", "guid: ")
        material_section = material_section.replace("type: ", "type: ")

        # Fix m_TexEnvs, m_Floats, and m_Colors sections to match Unity format
        # Unity uses an array-like format for these properties
        tex_envs_section = ""
        floats_section = ""
        colors_section = ""

        # Format texture properties
        for key, value in mat_data["Material"]["m_SavedProperties"]["m_TexEnvs"].items():
            tex_envs_section += f"    - {key}:\n"
            tex_envs_section += f"        m_Texture: {{fileID: {value['m_Texture']['fileID']}, guid: {value['m_Texture']['guid']}, type: {value['m_Texture']['type']}}}\n"
            tex_envs_section += f"        m_Scale: {{x: {value['m_Scale']['x']}, y: {value['m_Scale']['y']}}}\n"
            tex_envs_section += f"        m_Offset: {{x: {value['m_Offset']['x']}, y: {value['m_Offset']['y']}}}\n"

        # Format float properties
        for key, value in mat_data["Material"]["m_SavedProperties"]["m_Floats"].items():
            floats_section += f"    - {key}: {value}\n"

        # Format color properties
        for key, value in mat_data["Material"]["m_SavedProperties"]["m_Colors"].items():
            colors_section += f"    - {key}: {{r: {value['r']}, g: {value['g']}, b: {value['b']}, a: {value['a']}}}\n"

        # Build the final formatted YAML
        unity_mat = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!21 &2100000\n"
        unity_mat += "Material:\n"
        unity_mat += f"  serializedVersion: {mat_data['Material']['serializedVersion']}\n"
        unity_mat += f"  m_ObjectHideFlags: {mat_data['Material']['m_ObjectHideFlags']}\n"
        unity_mat += f"  m_CorrespondingSourceObject: {{fileID: {mat_data['Material']['m_CorrespondingSourceObject']['fileID']}}}\n"
        unity_mat += f"  m_PrefabInstance: {{fileID: {mat_data['Material']['m_PrefabInstance']['fileID']}}}\n"
        unity_mat += f"  m_PrefabAsset: {{fileID: {mat_data['Material']['m_PrefabAsset']['fileID']}}}\n"
        unity_mat += f"  m_Name: {mat_data['Material']['m_Name']}\n"
        unity_mat += f"  m_Shader: {{fileID: {mat_data['Material']['m_Shader']['fileID']}, guid: {mat_data['Material']['m_Shader']['guid']}, type: {mat_data['Material']['m_Shader']['type']}}}\n"
        unity_mat += f"  m_ShaderKeywords: {mat_data['Material']['m_ShaderKeywords']}\n"
        unity_mat += f"  m_CustomRenderQueue: {mat_data['Material']['m_CustomRenderQueue']}\n"
        unity_mat += "  stringTagMap:\n"
        unity_mat += f"    RenderType: {mat_data['Material']['stringTagMap']['RenderType']}\n"
        unity_mat += "  disabledShaderPasses: []\n"
        unity_mat += "  m_SavedProperties:\n"
        unity_mat += "    serializedVersion: 3\n"
        unity_mat += "    m_TexEnvs:\n"
        unity_mat += tex_envs_section
        unity_mat += "    m_Floats:\n"
        unity_mat += floats_section
        unity_mat += "    m_Colors:\n"
        unity_mat += colors_section

        return unity_mat
    
    def _copy_to_unity_project(self, material_result: Dict, texture_set: Dict) -> Dict:
        """
        Copy material and textures to Unity project.
        
        Args:
            material_result: Dictionary with material creation results
            texture_set: Dictionary with texture paths
            
        Returns:
            Dictionary with copy results
        """
        try:
            # Copy material
            material_path = material_result["material_path"]
            unity_material_path = self.unity_materials_path / os.path.basename(material_path)
            
            shutil.copy(material_path, unity_material_path)
            logger.info(f"Copied material to Unity: {unity_material_path}")
            
            # Copy textures
            copied_textures = []
            
            for texture_key, texture_path in texture_set.items():
                if texture_key.endswith("_texture") and os.path.exists(texture_path):
                    unity_texture_path = self.unity_textures_path / os.path.basename(texture_path)
                    
                    shutil.copy(texture_path, unity_texture_path)
                    copied_textures.append(str(unity_texture_path))
                    logger.info(f"Copied {texture_key} to Unity: {unity_texture_path}")
            
            return {
                "success": True,
                "unity_material_path": str(unity_material_path),
                "unity_textures": copied_textures
            }
        
        except Exception as e:
            logger.error(f"Error copying to Unity project: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def process_building_batch(self, 
                             buildings_metadata: Dict, 
                             texture_dir: str = None) -> Dict:
        """
        Process a batch of buildings, creating materials for each.
        
        Args:
            buildings_metadata: Dictionary of building metadata
            texture_dir: Directory containing textures (overrides constructor value)
            
        Returns:
            Dictionary with processing results
        """
        # Use provided texture directory or default
        texture_dir = texture_dir or self.texture_dir
        
        # Results tracking
        results = {
            "total": len(buildings_metadata),
            "success_count": 0,
            "failed_count": 0,
            "failed_buildings": [],
            "materials": {}
        }
        
        # Process each building
        for building_id, metadata in buildings_metadata.items():
            try:
                # Determine material type from metadata
                material_type = "default"
                if "material" in metadata:
                    material_type = metadata["material"]
                
                # Find texture for this building
                texture_path = None
                
                # Try different file extensions
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = os.path.join(texture_dir, f"{building_id}{ext}")
                    if os.path.exists(candidate):
                        texture_path = candidate
                        break
                
                # If not found by ID, try by material type
                if not texture_path:
                    for ext in [".jpg", ".jpeg", ".png"]:
                        candidate = os.path.join(texture_dir, f"{material_type}{ext}")
                        if os.path.exists(candidate):
                            texture_path = candidate
                            break
                
                # Use default texture if none found
                if not texture_path:
                    default_texture = os.path.join(texture_dir, "default.jpg")
                    if os.path.exists(default_texture):
                        texture_path = default_texture
                    else:
                        logger.warning(f"No texture found for building {building_id}")
                        results["failed_count"] += 1
                        results["failed_buildings"].append(building_id)
                        continue
                
                # Create material
                material_result = self.create_material_for_building(
                    building_id,
                    texture_path,
                    material_type,
                    metadata
                )
                
                if material_result["success"]:
                    results["success_count"] += 1
                    results["materials"][building_id] = {
                        "material_path": material_result["material_path"],
                        "material_type": material_type
                    }
                    
                    if "unity_path" in material_result:
                        results["materials"][building_id]["unity_path"] = material_result["unity_path"]
                else:
                    results["failed_count"] += 1
                    results["failed_buildings"].append(building_id)
            
            except Exception as e:
                logger.error(f"Error processing building {building_id}: {str(e)}")
                results["failed_count"] += 1
                results["failed_buildings"].append(building_id)
        
        return results

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unity Material Pipeline")
    parser.add_argument("--output", default="./arcanum_output", help="Output directory")
    parser.add_argument("--textures", help="Directory containing input textures")
    parser.add_argument("--unity", help="Unity project path")
    parser.add_argument("--buildings", required=True, help="Path to buildings metadata JSON file")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize material pipeline
    pipeline = UnityMaterialPipeline(args.output, args.textures, args.unity)
    
    # Load buildings metadata
    try:
        with open(args.buildings, 'r') as f:
            buildings_metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error loading buildings metadata: {str(e)}")
        sys.exit(1)
    
    # Process buildings
    results = pipeline.process_building_batch(buildings_metadata)
    
    print(f"Processed {results.get('total', 0)} buildings:")
    print(f"  - Success: {results.get('success_count', 0)}")
    print(f"  - Failed: {results.get('failed_count', 0)}")