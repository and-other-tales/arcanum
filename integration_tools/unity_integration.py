#!/usr/bin/env python3
"""
Unity Integration Module for Arcanum
------------------------------------
This module provides integration with Unity for exporting Arcanum 3D models and scenes.
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class UnityExporter:
    """Class for exporting assets to Unity"""
    
    def __init__(self, unity_project_path: str = None, output_dir: str = None):
        """Initialize the Unity exporter.
        
        Args:
            unity_project_path: Path to Unity project
            output_dir: Path to Arcanum output directory
        """
        self.unity_project_path = unity_project_path
        self.output_dir = output_dir or os.path.expanduser("~/arcanum_output")
        
        # Check if Unity project exists
        if unity_project_path and not os.path.exists(unity_project_path):
            logger.warning(f"Unity project path does not exist: {unity_project_path}")
    
    def export_terrain(self, terrain_dir: str, target_dir: str = None) -> bool:
        """Export terrain heightmaps to Unity.
        
        Args:
            terrain_dir: Directory containing terrain heightmaps
            target_dir: Target directory in Unity project (relative to Assets)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Default target directory if not specified
            if not target_dir:
                target_dir = "Terrain/Heightmaps"
            
            # Check if terrain directory exists
            if not os.path.exists(terrain_dir):
                logger.error(f"Terrain directory does not exist: {terrain_dir}")
                return False
            
            # Create target directory in Unity project
            unity_terrain_dir = os.path.join(self.unity_project_path, "Assets", target_dir)
            os.makedirs(unity_terrain_dir, exist_ok=True)
            
            # Copy heightmaps to Unity project
            copied_files = 0
            for file in os.listdir(terrain_dir):
                if file.endswith(".raw") or file.endswith(".exr") or file.endswith(".png"):
                    src = os.path.join(terrain_dir, file)
                    dst = os.path.join(unity_terrain_dir, file)
                    shutil.copy2(src, dst)
                    copied_files += 1
            
            logger.info(f"Exported {copied_files} terrain files to Unity project at {unity_terrain_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting terrain: {str(e)}")
            return False
    
    def export_buildings(self, buildings_dir: str, target_dir: str = None) -> bool:
        """Export building models to Unity.
        
        Args:
            buildings_dir: Directory containing building models
            target_dir: Target directory in Unity project (relative to Assets)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Default target directory if not specified
            if not target_dir:
                target_dir = "Models/Buildings"
            
            # Check if buildings directory exists
            if not os.path.exists(buildings_dir):
                logger.error(f"Buildings directory does not exist: {buildings_dir}")
                return False
            
            # Create target directory in Unity project
            unity_buildings_dir = os.path.join(self.unity_project_path, "Assets", target_dir)
            os.makedirs(unity_buildings_dir, exist_ok=True)
            
            # Copy building models to Unity project
            copied_files = 0
            for file in os.listdir(buildings_dir):
                if file.endswith(".fbx") or file.endswith(".obj") or file.endswith(".glb"):
                    src = os.path.join(buildings_dir, file)
                    dst = os.path.join(unity_buildings_dir, file)
                    shutil.copy2(src, dst)
                    copied_files += 1
            
            logger.info(f"Exported {copied_files} building models to Unity project at {unity_buildings_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting buildings: {str(e)}")
            return False
    
    def export_textures(self, textures_dir: str, target_dir: str = None) -> bool:
        """Export textures to Unity.
        
        Args:
            textures_dir: Directory containing textures
            target_dir: Target directory in Unity project (relative to Assets)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Default target directory if not specified
            if not target_dir:
                target_dir = "Textures/Arcanum"
            
            # Check if textures directory exists
            if not os.path.exists(textures_dir):
                logger.error(f"Textures directory does not exist: {textures_dir}")
                return False
            
            # Create target directory in Unity project
            unity_textures_dir = os.path.join(self.unity_project_path, "Assets", target_dir)
            os.makedirs(unity_textures_dir, exist_ok=True)
            
            # Copy textures to Unity project
            copied_files = 0
            for file in os.listdir(textures_dir):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".tga", ".psd")):
                    src = os.path.join(textures_dir, file)
                    dst = os.path.join(unity_textures_dir, file)
                    shutil.copy2(src, dst)
                    copied_files += 1
            
            logger.info(f"Exported {copied_files} textures to Unity project at {unity_textures_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting textures: {str(e)}")
            return False
    
    def generate_prefabs(self, metadata_file: str) -> bool:
        """Generate prefabs based on metadata.
        
        Args:
            metadata_file: Path to metadata JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if metadata file exists
            if not os.path.exists(metadata_file):
                logger.error(f"Metadata file does not exist: {metadata_file}")
                return False
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # This is a placeholder - in a real implementation, we'd generate
            # a Unity Editor script to create prefabs based on the metadata
            unity_editor_script = os.path.join(
                self.unity_project_path, 
                "Assets", 
                "Editor", 
                "Scripts", 
                "ArcanumPrefabGenerator.cs"
            )
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(unity_editor_script), exist_ok=True)
            
            # Write C# script template
            with open(unity_editor_script, 'w') as f:
                f.write(f"""
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;

public class ArcanumPrefabGenerator : EditorWindow
{{
    [MenuItem("Arcanum/Generate Prefabs")]
    public static void GeneratePrefabs()
    {{
        // This is a generated script that would create prefabs based on:
        // {metadata_file}
        
        Debug.Log("Generating Arcanum prefabs...");
        
        // Actual implementation would load models and create prefabs
        // based on the metadata JSON structure
    }}
}}
""")
            
            logger.info(f"Generated Unity Editor script at {unity_editor_script}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating prefabs: {str(e)}")
            return False
    
    def export_all(self) -> bool:
        """Export all assets to Unity project.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.unity_project_path:
            logger.error("Unity project path not specified")
            return False
        
        success = True
        
        # Export terrain
        terrain_dir = os.path.join(self.output_dir, "processed_data/terrain/heightmaps")
        if os.path.exists(terrain_dir):
            if not self.export_terrain(terrain_dir):
                success = False
        else:
            logger.warning(f"Terrain directory not found: {terrain_dir}")
        
        # Export buildings
        buildings_dir = os.path.join(self.output_dir, "3d_models/buildings")
        if os.path.exists(buildings_dir):
            if not self.export_buildings(buildings_dir):
                success = False
        else:
            logger.warning(f"Buildings directory not found: {buildings_dir}")
        
        # Export textures
        textures_dir = os.path.join(self.output_dir, "processed_data/textures")
        if os.path.exists(textures_dir):
            if not self.export_textures(textures_dir):
                success = False
        else:
            logger.warning(f"Textures directory not found: {textures_dir}")
        
        # Export metadata
        metadata_file = os.path.join(self.output_dir, "unity_assets/metadata.json")
        if os.path.exists(metadata_file):
            if not self.generate_prefabs(metadata_file):
                success = False
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
        
        return success