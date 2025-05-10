#!/usr/bin/env python3
"""
Export Manager for Arcanum
------------------------
This module provides functionality for exporting Arcanum-generated cities to various formats,
such as 3D models (GLB, GLTF), Unity assets, and more.
"""

import os
import sys
import logging
import json
import shutil
import subprocess
import tempfile
import zipfile
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Import helpers for parallel processing
try:
    from modules.parallel import task_manager
    PARALLEL_AVAILABLE = True
except ImportError:
    logger.warning("Parallel task management not available, falling back to sequential processing")
    PARALLEL_AVAILABLE = False

class ExportManager:
    """Manages the export of Arcanum-generated cities to various formats."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the export manager.
        
        Args:
            output_dir: Base directory for exports, or None for default
        """
        # Default output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      "exports")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Register supported export formats
        self.export_formats = {
            "obj": self.export_obj,
            "glb": self.export_glb,
            "gltf": self.export_gltf,
            "fbx": self.export_fbx,
            "unity": self.export_unity,
            "unreal": self.export_unreal,
            "cesium": self.export_cesium,
            "mapbox": self.export_mapbox
        }
        
        logger.info(f"Export manager initialized with output directory: {output_dir}")
        
    def export_buildings(self, buildings_path: str, processed_dir: str, 
                         format: str = "glb", export_id: str = None, 
                         options: Dict = None) -> Dict[str, Any]:
        """
        Export buildings to the specified format.
        
        Args:
            buildings_path: Path to the buildings data file
            processed_dir: Directory with processed building data
            format: Export format (obj, glb, gltf, fbx, unity, unreal, cesium, mapbox)
            export_id: Optional ID for the export (auto-generated if not provided)
            options: Format-specific export options
            
        Returns:
            Dictionary with export results
        """
        # Generate export ID if not provided
        if export_id is None:
            export_id = f"export_{str(uuid.uuid4())[:8]}"
        
        # Create export directory
        export_path = os.path.join(self.output_dir, export_id)
        os.makedirs(export_path, exist_ok=True)
        
        # Default options
        if options is None:
            options = {}
        
        # Check if format is supported
        if format.lower() not in self.export_formats:
            logger.error(f"Unsupported export format: {format}")
            return {
                "success": False,
                "error": f"Unsupported export format: {format}",
                "export_id": export_id,
                "export_path": export_path
            }
        
        try:
            # Call the appropriate export function
            export_func = self.export_formats[format.lower()]
            result = export_func(buildings_path, processed_dir, export_path, options)
            
            # Add common fields to result
            result.update({
                "export_id": export_id,
                "export_path": export_path,
                "format": format
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error exporting to {format}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "export_id": export_id,
                "export_path": export_path
            }
    
    def export_obj(self, buildings_path: str, processed_dir: str, 
                  export_path: str, options: Dict = None) -> Dict[str, Any]:
        """
        Export buildings to OBJ format.
        
        Args:
            buildings_path: Path to the buildings data file
            processed_dir: Directory with processed building data
            export_path: Output directory for the export
            options: Export options
            
        Returns:
            Dictionary with export results
        """
        logger.info(f"Exporting buildings to OBJ format: {export_path}")
        
        try:
            # Create models directory in export path
            models_dir = os.path.join(export_path, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Get list of processed buildings
            building_files = []
            textures_dir = os.path.join(processed_dir, "textures")
            models_src_dir = os.path.join(processed_dir, "models")
            
            if os.path.exists(models_src_dir):
                for filename in os.listdir(models_src_dir):
                    if filename.endswith(".obj"):
                        building_files.append(os.path.join(models_src_dir, filename))
            
            # Use parallel processing if available
            if PARALLEL_AVAILABLE and len(building_files) > 10:
                # Define task for converting each building
                def convert_building_obj(src_path, dst_dir, progress_callback=None):
                    try:
                        # Copy OBJ and MTL files
                        basename = os.path.basename(src_path)
                        dst_path = os.path.join(dst_dir, basename)
                        shutil.copy2(src_path, dst_path)
                        
                        # Copy MTL file if it exists
                        mtl_src = os.path.splitext(src_path)[0] + ".mtl"
                        if os.path.exists(mtl_src):
                            mtl_dst = os.path.splitext(dst_path)[0] + ".mtl"
                            shutil.copy2(mtl_src, mtl_dst)
                        
                        # Success
                        if progress_callback:
                            progress_callback.update(1.0, f"Exported {basename}")
                            
                        return {"success": True, "filename": basename}
                        
                    except Exception as e:
                        logger.error(f"Error exporting {src_path}: {str(e)}")
                        return {"success": False, "error": str(e), "filename": os.path.basename(src_path)}
                
                # Create tasks for parallel execution
                tasks = []
                for i, building_file in enumerate(building_files):
                    tasks.append({
                        "name": f"Export OBJ {i+1}/{len(building_files)}",
                        "func": convert_building_obj,
                        "args": (building_file, models_dir)
                    })
                
                # Execute tasks in parallel
                if tasks:
                    results = task_manager.execute_parallel(tasks)
                    
                    # Count successful exports
                    success_count = sum(1 for r in results.values() if r.get("result", {}).get("success", False))
                    logger.info(f"Successfully exported {success_count}/{len(tasks)} buildings to OBJ")
                    
                    return {
                        "success": True,
                        "exported_count": success_count,
                        "total_count": len(tasks),
                        "models_dir": models_dir
                    }
                    
            else:
                # Sequential processing
                exported_count = 0
                for building_file in building_files:
                    try:
                        # Copy OBJ and MTL files
                        basename = os.path.basename(building_file)
                        dst_path = os.path.join(models_dir, basename)
                        shutil.copy2(building_file, dst_path)
                        
                        # Copy MTL file if it exists
                        mtl_src = os.path.splitext(building_file)[0] + ".mtl"
                        if os.path.exists(mtl_src):
                            mtl_dst = os.path.splitext(dst_path)[0] + ".mtl"
                            shutil.copy2(mtl_src, mtl_dst)
                            
                        exported_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error exporting {building_file}: {str(e)}")
            
                logger.info(f"Successfully exported {exported_count}/{len(building_files)} buildings to OBJ")
                
                return {
                    "success": True,
                    "exported_count": exported_count,
                    "total_count": len(building_files),
                    "models_dir": models_dir
                }
                
            # If we have no building files
            if not building_files:
                logger.warning("No building files found for export")
                return {
                    "success": True,
                    "exported_count": 0,
                    "total_count": 0,
                    "models_dir": models_dir
                }
            
        except Exception as e:
            logger.error(f"Error in OBJ export: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def export_glb(self, buildings_path: str, processed_dir: str, 
                  export_path: str, options: Dict = None) -> Dict[str, Any]:
        """
        Export buildings to GLB format.
        
        Args:
            buildings_path: Path to the buildings data file
            processed_dir: Directory with processed building data
            export_path: Output directory for the export
            options: Export options
            
        Returns:
            Dictionary with export results
        """
        logger.info(f"Exporting buildings to GLB format: {export_path}")
        
        try:
            # Create models directory in export path
            models_dir = os.path.join(export_path, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Get list of processed buildings
            glb_files = []
            models_src_dir = os.path.join(processed_dir, "models")
            
            if os.path.exists(models_src_dir):
                for filename in os.listdir(models_src_dir):
                    if filename.endswith(".glb"):
                        glb_files.append(os.path.join(models_src_dir, filename))
            
            # Check if we need conversion from OBJ
            obj_files = []
            if not glb_files:
                for filename in os.listdir(models_src_dir):
                    if filename.endswith(".obj"):
                        obj_files.append(os.path.join(models_src_dir, filename))
                
                # Check if we have obj2gltf available for conversion
                if obj_files:
                    try:
                        # Try to use obj2gltf if installed
                        subprocess.run(["obj2gltf", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        has_obj2gltf = True
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        has_obj2gltf = False
                        logger.warning("obj2gltf not found, conversion from OBJ to GLB not available")
                        
                    # Convert OBJ files to GLB if possible
                    if has_obj2gltf:
                        logger.info(f"Converting {len(obj_files)} OBJ files to GLB format")
                        
                        # Use parallel processing if available
                        if PARALLEL_AVAILABLE and len(obj_files) > 5:
                            def convert_obj_to_glb(obj_path, dst_dir, progress_callback=None):
                                try:
                                    basename = os.path.basename(obj_path)
                                    glb_name = os.path.splitext(basename)[0] + ".glb"
                                    glb_path = os.path.join(dst_dir, glb_name)
                                    
                                    # Convert using obj2gltf
                                    if progress_callback:
                                        progress_callback.update(0.5, f"Converting {basename}")
                                        
                                    subprocess.run([
                                        "obj2gltf", "-i", obj_path, "-o", glb_path
                                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    
                                    if progress_callback:
                                        progress_callback.update(1.0, f"Converted {basename} to {glb_name}")
                                        
                                    return {"success": True, "filename": glb_name, "source": basename}
                                    
                                except Exception as e:
                                    logger.error(f"Error converting {obj_path}: {str(e)}")
                                    return {"success": False, "error": str(e), "filename": os.path.basename(obj_path)}
                            
                            # Create tasks for parallel execution
                            tasks = []
                            for i, obj_file in enumerate(obj_files):
                                tasks.append({
                                    "name": f"Convert OBJ to GLB {i+1}/{len(obj_files)}",
                                    "func": convert_obj_to_glb,
                                    "args": (obj_file, models_dir)
                                })
                            
                            # Execute tasks in parallel
                            results = task_manager.execute_parallel(tasks)
                            
                            # Count successful conversions
                            success_count = sum(1 for r in results.values() if r.get("result", {}).get("success", False))
                            logger.info(f"Successfully converted {success_count}/{len(tasks)} OBJ files to GLB")
                            
                            return {
                                "success": True,
                                "exported_count": success_count,
                                "total_count": len(tasks),
                                "models_dir": models_dir,
                                "converted_from_obj": True
                            }
                        
                        else:
                            # Sequential conversion
                            converted_count = 0
                            for obj_file in obj_files:
                                try:
                                    # Convert to GLB
                                    basename = os.path.basename(obj_file)
                                    glb_name = os.path.splitext(basename)[0] + ".glb"
                                    glb_path = os.path.join(models_dir, glb_name)
                                    
                                    subprocess.run([
                                        "obj2gltf", "-i", obj_file, "-o", glb_path
                                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    
                                    converted_count += 1
                                    
                                except Exception as e:
                                    logger.error(f"Error converting {obj_file}: {str(e)}")
                            
                            logger.info(f"Successfully converted {converted_count}/{len(obj_files)} OBJ files to GLB")
                            
                            return {
                                "success": True,
                                "exported_count": converted_count,
                                "total_count": len(obj_files),
                                "models_dir": models_dir,
                                "converted_from_obj": True
                            }
            
            # Copy existing GLB files
            if glb_files:
                exported_count = 0
                for glb_file in glb_files:
                    try:
                        # Copy GLB file
                        basename = os.path.basename(glb_file)
                        dst_path = os.path.join(models_dir, basename)
                        shutil.copy2(glb_file, dst_path)
                        
                        exported_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error exporting {glb_file}: {str(e)}")
                
                logger.info(f"Successfully exported {exported_count}/{len(glb_files)} buildings to GLB")
                
                return {
                    "success": True,
                    "exported_count": exported_count,
                    "total_count": len(glb_files),
                    "models_dir": models_dir
                }
            
            # If we reach here, we couldn't find any suitable files
            if not obj_files and not glb_files:
                logger.warning("No building files found for export")
                return {
                    "success": True,
                    "exported_count": 0,
                    "total_count": 0,
                    "models_dir": models_dir
                }
                
        except Exception as e:
            logger.error(f"Error in GLB export: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def export_gltf(self, buildings_path: str, processed_dir: str, 
                   export_path: str, options: Dict = None) -> Dict[str, Any]:
        """
        Export buildings to GLTF format.
        
        Args:
            buildings_path: Path to the buildings data file
            processed_dir: Directory with processed building data
            export_path: Output directory for the export
            options: Export options
            
        Returns:
            Dictionary with export results
        """
        logger.info(f"Exporting buildings to GLTF format: {export_path}")
        
        # GLTF export is similar to GLB, but we convert/save to .gltf instead
        try:
            # Create models directory in export path
            models_dir = os.path.join(export_path, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Get list of processed buildings
            gltf_files = []
            models_src_dir = os.path.join(processed_dir, "models")
            
            if os.path.exists(models_src_dir):
                for filename in os.listdir(models_src_dir):
                    if filename.endswith(".gltf"):
                        gltf_files.append(os.path.join(models_src_dir, filename))
            
            # Check if we have .glb files that we can convert
            glb_files = []
            if not gltf_files:
                for filename in os.listdir(models_src_dir):
                    if filename.endswith(".glb"):
                        glb_files.append(os.path.join(models_src_dir, filename))
                        
                # If we have GLB files, extract them to GLTF
                if glb_files:
                    try:
                        # Try to use gltf-transform
                        subprocess.run(["gltf-transform", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        has_gltf_transform = True
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        has_gltf_transform = False
                        logger.warning("gltf-transform not found, extraction from GLB to GLTF limited")
                        
                    # Convert GLB to GLTF
                    if has_gltf_transform:
                        logger.info(f"Converting {len(glb_files)} GLB files to GLTF format")
                        
                        # Use parallel processing if available
                        if PARALLEL_AVAILABLE and len(glb_files) > 5:
                            # Define task for converting GLB to GLTF
                            def convert_glb_to_gltf(glb_path, dst_dir, progress_callback=None):
                                try:
                                    basename = os.path.basename(glb_path)
                                    gltf_name = os.path.splitext(basename)[0] + ".gltf"
                                    gltf_path = os.path.join(dst_dir, gltf_name)
                                    
                                    # Convert using gltf-transform
                                    if progress_callback:
                                        progress_callback.update(0.5, f"Converting {basename}")
                                        
                                    subprocess.run([
                                        "gltf-transform", "extract", glb_path, gltf_path
                                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    
                                    if progress_callback:
                                        progress_callback.update(1.0, f"Converted {basename} to {gltf_name}")
                                        
                                    return {"success": True, "filename": gltf_name, "source": basename}
                                    
                                except Exception as e:
                                    logger.error(f"Error converting {glb_path}: {str(e)}")
                                    return {"success": False, "error": str(e), "filename": os.path.basename(glb_path)}
                            
                            # Create tasks for parallel execution
                            tasks = []
                            for i, glb_file in enumerate(glb_files):
                                tasks.append({
                                    "name": f"Convert GLB to GLTF {i+1}/{len(glb_files)}",
                                    "func": convert_glb_to_gltf,
                                    "args": (glb_file, models_dir)
                                })
                            
                            # Execute tasks in parallel
                            results = task_manager.execute_parallel(tasks)
                            
                            # Count successful conversions
                            success_count = sum(1 for r in results.values() if r.get("result", {}).get("success", False))
                            logger.info(f"Successfully converted {success_count}/{len(tasks)} GLB files to GLTF")
                            
                            return {
                                "success": True,
                                "exported_count": success_count,
                                "total_count": len(tasks),
                                "models_dir": models_dir,
                                "converted_from_glb": True
                            }
                        
                        else:
                            # Sequential conversion
                            converted_count = 0
                            for glb_file in glb_files:
                                try:
                                    # Convert to GLTF
                                    basename = os.path.basename(glb_file)
                                    gltf_name = os.path.splitext(basename)[0] + ".gltf"
                                    gltf_path = os.path.join(models_dir, gltf_name)
                                    
                                    subprocess.run([
                                        "gltf-transform", "extract", glb_file, gltf_path
                                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    
                                    converted_count += 1
                                    
                                except Exception as e:
                                    logger.error(f"Error converting {glb_file}: {str(e)}")
                            
                            logger.info(f"Successfully converted {converted_count}/{len(glb_files)} GLB files to GLTF")
                            
                            return {
                                "success": True,
                                "exported_count": converted_count,
                                "total_count": len(glb_files),
                                "models_dir": models_dir,
                                "converted_from_glb": True
                            }
            
            # Copy existing GLTF files
            if gltf_files:
                exported_count = 0
                for gltf_file in gltf_files:
                    try:
                        # Copy GLTF file
                        basename = os.path.basename(gltf_file)
                        dst_path = os.path.join(models_dir, basename)
                        shutil.copy2(gltf_file, dst_path)
                        
                        # Also copy associated .bin files
                        bin_src = os.path.splitext(gltf_file)[0] + ".bin"
                        if os.path.exists(bin_src):
                            bin_dst = os.path.splitext(dst_path)[0] + ".bin"
                            shutil.copy2(bin_src, bin_dst)
                        
                        exported_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error exporting {gltf_file}: {str(e)}")
                
                logger.info(f"Successfully exported {exported_count}/{len(gltf_files)} buildings to GLTF")
                
                return {
                    "success": True,
                    "exported_count": exported_count,
                    "total_count": len(gltf_files),
                    "models_dir": models_dir
                }
            
            # If we reach here with no models, fall back to OBJ conversion
            obj_files = []
            for filename in os.listdir(models_src_dir):
                if filename.endswith(".obj"):
                    obj_files.append(os.path.join(models_src_dir, filename))
                    
            if obj_files:
                # Use OBJ to GLB export and then convert to GLTF
                glb_result = self.export_glb(buildings_path, processed_dir, export_path, options)
                
                if glb_result.get("success", False) and glb_result.get("exported_count", 0) > 0:
                    # Now convert GLB to GLTF
                    try:
                        if has_gltf_transform:
                            glb_models_dir = glb_result.get("models_dir")
                            
                            # Get GLB files
                            converted_glb_files = []
                            for filename in os.listdir(glb_models_dir):
                                if filename.endswith(".glb"):
                                    converted_glb_files.append(os.path.join(glb_models_dir, filename))
                            
                            # Convert them to GLTF
                            converted_count = 0
                            for glb_file in converted_glb_files:
                                try:
                                    # Convert to GLTF
                                    basename = os.path.basename(glb_file)
                                    gltf_name = os.path.splitext(basename)[0] + ".gltf"
                                    gltf_path = os.path.join(models_dir, gltf_name)
                                    
                                    subprocess.run([
                                        "gltf-transform", "extract", glb_file, gltf_path
                                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    
                                    converted_count += 1
                                    
                                except Exception as e:
                                    logger.error(f"Error converting {glb_file}: {str(e)}")
                            
                            logger.info(f"Successfully converted {converted_count}/{len(converted_glb_files)} GLB files to GLTF")
                            
                            return {
                                "success": True,
                                "exported_count": converted_count,
                                "total_count": len(converted_glb_files),
                                "models_dir": models_dir,
                                "converted_from_obj": True
                            }
                    except Exception as e:
                        logger.error(f"Error converting GLB to GLTF: {str(e)}")
                        
            # If we reach here, we couldn't find any suitable files
            if not obj_files and not glb_files and not gltf_files:
                logger.warning("No building files found for export")
                return {
                    "success": True,
                    "exported_count": 0,
                    "total_count": 0,
                    "models_dir": models_dir
                }
                
        except Exception as e:
            logger.error(f"Error in GLTF export: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def export_fbx(self, buildings_path: str, processed_dir: str, 
                  export_path: str, options: Dict = None) -> Dict[str, Any]:
        """
        Export buildings to FBX format.
        
        Args:
            buildings_path: Path to the buildings data file
            processed_dir: Directory with processed building data
            export_path: Output directory for the export
            options: Export options
            
        Returns:
            Dictionary with export results
        """
        logger.info(f"Exporting buildings to FBX format: {export_path}")
        
        try:
            # Create models directory in export path
            models_dir = os.path.join(export_path, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Get list of processed buildings
            fbx_files = []
            models_src_dir = os.path.join(processed_dir, "models")
            
            if os.path.exists(models_src_dir):
                for filename in os.listdir(models_src_dir):
                    if filename.endswith(".fbx"):
                        fbx_files.append(os.path.join(models_src_dir, filename))
            
            # Check if we need conversion
            if not fbx_files:
                logger.warning("No FBX files found and automatic conversion to FBX not supported")
                logger.info("Consider using Blender for OBJ/GLB to FBX conversion")
                
                return {
                    "success": False,
                    "error": "No FBX files found and conversion not supported",
                    "models_dir": models_dir
                }
            
            # Copy existing FBX files
            exported_count = 0
            for fbx_file in fbx_files:
                try:
                    # Copy FBX file
                    basename = os.path.basename(fbx_file)
                    dst_path = os.path.join(models_dir, basename)
                    shutil.copy2(fbx_file, dst_path)
                    
                    exported_count += 1
                    
                except Exception as e:
                    logger.error(f"Error exporting {fbx_file}: {str(e)}")
            
            logger.info(f"Successfully exported {exported_count}/{len(fbx_files)} buildings to FBX")
            
            return {
                "success": True,
                "exported_count": exported_count,
                "total_count": len(fbx_files),
                "models_dir": models_dir
            }
                
        except Exception as e:
            logger.error(f"Error in FBX export: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_unity(self, buildings_path: str, processed_dir: str, 
                    export_path: str, options: Dict = None) -> Dict[str, Any]:
        """
        Export buildings as a Unity package.
        
        Args:
            buildings_path: Path to the buildings data file
            processed_dir: Directory with processed building data
            export_path: Output directory for the export
            options: Export options
            
        Returns:
            Dictionary with export results
        """
        logger.info(f"Exporting buildings as Unity package: {export_path}")
        
        try:
            # Create Unity package directory
            unity_dir = os.path.join(export_path, "unity")
            os.makedirs(unity_dir, exist_ok=True)
            
            # Create directory structure for Unity package
            package_structure = [
                "Assets/ArcanumCity/Models",
                "Assets/ArcanumCity/Materials",
                "Assets/ArcanumCity/Textures",
                "Assets/ArcanumCity/Prefabs",
                "Assets/ArcanumCity/Scenes"
            ]
            
            for dir_path in package_structure:
                os.makedirs(os.path.join(unity_dir, dir_path), exist_ok=True)
            
            # Export models to glTF format
            gltf_result = self.export_gltf(buildings_path, processed_dir, export_path, options)
            
            if not gltf_result.get("success", False):
                logger.error("Failed to export models to GLTF format")
                return {
                    "success": False,
                    "error": "Failed to export models to GLTF format",
                    "unity_dir": unity_dir
                }
            
            # Copy models to Unity package
            models_src_dir = gltf_result.get("models_dir")
            models_dst_dir = os.path.join(unity_dir, "Assets/ArcanumCity/Models")
            
            copied_count = 0
            model_files = []
            
            for filename in os.listdir(models_src_dir):
                if filename.endswith(".gltf") or filename.endswith(".bin"):
                    src_path = os.path.join(models_src_dir, filename)
                    dst_path = os.path.join(models_dst_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    model_files.append(filename)
                    copied_count += 1
            
            # Create metadata file
            metadata = {
                "package_name": "ArcanumCity",
                "version": "1.0.0",
                "created_at": time.time(),
                "description": "Arcanum City Generator export for Unity",
                "models": model_files,
                "building_count": gltf_result.get("exported_count", 0)
            }
            
            with open(os.path.join(unity_dir, "Assets/ArcanumCity/metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Create a basic Unity scene file
            scene_content = """
# Arcanum City Unity Scene
# Import this file into your Unity project

# Models:
{models}

# Center point:
center: {center}

# This is a placeholder for a real Unity scene file.
# In a real implementation, we would generate a proper Unity scene file.
            """.format(
                models="\n".join([f"- {model}" for model in model_files if model.endswith(".gltf")]),
                center="0,0,0"  # Placeholder
            )
            
            with open(os.path.join(unity_dir, "Assets/ArcanumCity/Scenes/ArcanumCity.unity"), "w") as f:
                f.write(scene_content)
            
            # Create a Unity package file (zip format)
            package_path = os.path.join(export_path, "ArcanumCity.unitypackage")
            
            with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(unity_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, unity_dir)
                        zipf.write(file_path, rel_path)
            
            logger.info(f"Successfully created Unity package at {package_path}")
            
            return {
                "success": True,
                "exported_count": gltf_result.get("exported_count", 0),
                "unity_dir": unity_dir,
                "package_path": package_path
            }
                
        except Exception as e:
            logger.error(f"Error in Unity export: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_unreal(self, buildings_path: str, processed_dir: str, 
                     export_path: str, options: Dict = None) -> Dict[str, Any]:
        """
        Export buildings for Unreal Engine.
        
        Args:
            buildings_path: Path to the buildings data file
            processed_dir: Directory with processed building data
            export_path: Output directory for the export
            options: Export options
            
        Returns:
            Dictionary with export results
        """
        logger.info(f"Exporting buildings for Unreal Engine: {export_path}")
        
        try:
            # Create Unreal package directory
            unreal_dir = os.path.join(export_path, "unreal")
            os.makedirs(unreal_dir, exist_ok=True)
            
            # Create directory structure for Unreal package
            package_structure = [
                "Content/ArcanumCity/Models",
                "Content/ArcanumCity/Materials",
                "Content/ArcanumCity/Textures",
                "Content/ArcanumCity/Maps"
            ]
            
            for dir_path in package_structure:
                os.makedirs(os.path.join(unreal_dir, dir_path), exist_ok=True)
            
            # For Unreal, FBX is preferred but gltf works too
            # Try FBX first
            fbx_result = self.export_fbx(buildings_path, processed_dir, export_path, options)
            
            if fbx_result.get("success", False) and fbx_result.get("exported_count", 0) > 0:
                # Use FBX files
                models_src_dir = fbx_result.get("models_dir")
                models_dst_dir = os.path.join(unreal_dir, "Content/ArcanumCity/Models")
                
                copied_count = 0
                model_files = []
                
                for filename in os.listdir(models_src_dir):
                    if filename.endswith(".fbx"):
                        src_path = os.path.join(models_src_dir, filename)
                        dst_path = os.path.join(models_dst_dir, filename)
                        shutil.copy2(src_path, dst_path)
                        model_files.append(filename)
                        copied_count += 1
                
                model_format = "fbx"
                exported_count = fbx_result.get("exported_count", 0)
                
            else:
                # Fall back to GLTF
                gltf_result = self.export_gltf(buildings_path, processed_dir, export_path, options)
                
                if not gltf_result.get("success", False) or gltf_result.get("exported_count", 0) == 0:
                    logger.error("Failed to export models in a format suitable for Unreal Engine")
                    return {
                        "success": False,
                        "error": "Failed to export models in a format suitable for Unreal Engine",
                        "unreal_dir": unreal_dir
                    }
                
                # Copy GLTF files to Unreal package
                models_src_dir = gltf_result.get("models_dir")
                models_dst_dir = os.path.join(unreal_dir, "Content/ArcanumCity/Models")
                
                copied_count = 0
                model_files = []
                
                for filename in os.listdir(models_src_dir):
                    if filename.endswith(".gltf") or filename.endswith(".bin"):
                        src_path = os.path.join(models_src_dir, filename)
                        dst_path = os.path.join(models_dst_dir, filename)
                        shutil.copy2(src_path, dst_path)
                        model_files.append(filename)
                        copied_count += 1
                
                model_format = "gltf"
                exported_count = gltf_result.get("exported_count", 0)
            
            # Create metadata file
            metadata = {
                "package_name": "ArcanumCity",
                "version": "1.0.0",
                "created_at": time.time(),
                "description": "Arcanum City Generator export for Unreal Engine",
                "models": model_files,
                "model_format": model_format,
                "building_count": exported_count
            }
            
            with open(os.path.join(unreal_dir, "Content/ArcanumCity/metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Create a basic level blueprint file (as a placeholder)
            blueprint_content = """
# Arcanum City Unreal Level Blueprint
# Import this file into your Unreal Engine project

# Models:
{models}

# Center point:
center: {center}

# This is a placeholder for a real Unreal Engine level blueprint.
# In a real implementation, we would generate proper Unreal Engine files.
            """.format(
                models="\n".join([f"- {model}" for model in model_files if model.endswith(f".{model_format}")]),
                center="0,0,0"  # Placeholder
            )
            
            with open(os.path.join(unreal_dir, "Content/ArcanumCity/Maps/ArcanumCity.txt"), "w") as f:
                f.write(blueprint_content)
            
            # Create a zip archive
            package_path = os.path.join(export_path, "ArcanumCity_Unreal.zip")
            
            with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(unreal_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, unreal_dir)
                        zipf.write(file_path, rel_path)
            
            logger.info(f"Successfully created Unreal Engine package at {package_path}")
            
            return {
                "success": True,
                "exported_count": exported_count,
                "unreal_dir": unreal_dir,
                "package_path": package_path,
                "model_format": model_format
            }
                
        except Exception as e:
            logger.error(f"Error in Unreal export: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_cesium(self, buildings_path: str, processed_dir: str, 
                     export_path: str, options: Dict = None) -> Dict[str, Any]:
        """
        Export buildings to Cesium 3D Tiles format.
        
        Args:
            buildings_path: Path to the buildings data file
            processed_dir: Directory with processed building data
            export_path: Output directory for the export
            options: Export options
            
        Returns:
            Dictionary with export results
        """
        logger.info(f"Exporting buildings to Cesium 3D Tiles format: {export_path}")
        
        try:
            # Create 3D Tiles directory
            tiles_dir = os.path.join(export_path, "cesium")
            os.makedirs(tiles_dir, exist_ok=True)
            
            # Check if we have the capability to convert to 3D Tiles
            try:
                # See if 3d-tiles-tools is available
                subprocess.run(["3d-tiles-tools", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                has_3d_tiles_tools = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                has_3d_tiles_tools = False
                logger.warning("3d-tiles-tools not found, using alternative method")
            
            # Export glTF files first
            gltf_result = self.export_gltf(buildings_path, processed_dir, export_path, options)
            
            if not gltf_result.get("success", False) or gltf_result.get("exported_count", 0) == 0:
                logger.error("Failed to export models to glTF format")
                return {
                    "success": False,
                    "error": "Failed to export models to glTF format",
                    "tiles_dir": tiles_dir
                }
            
            # Get source directory with glTF files
            models_src_dir = gltf_result.get("models_dir")
            
            # List of glTF files
            gltf_files = []
            for filename in os.listdir(models_src_dir):
                if filename.endswith(".gltf"):
                    gltf_files.append(os.path.join(models_src_dir, filename))
            
            if has_3d_tiles_tools:
                # Use 3d-tiles-tools for conversion
                logger.info("Using 3d-tiles-tools for conversion to 3D Tiles")
                
                # Create a temporary JSON file with glTF paths
                gltf_list_file = os.path.join(export_path, "gltf_files.json")
                with open(gltf_list_file, "w") as f:
                    json.dump(gltf_files, f)
                
                # Run the conversion command
                subprocess.run([
                    "3d-tiles-tools", "gltfToB3dm", 
                    "--input", gltf_list_file,
                    "--output", os.path.join(tiles_dir, "b3dm")
                ], check=True)
                
                # Generate tileset.json
                tileset_json = {
                    "asset": {
                        "version": "1.0",
                        "generator": "Arcanum City Generator"
                    },
                    "geometricError": 100,
                    "root": {
                        "boundingVolume": {
                            "region": [
                                -2.0, 0.5, -1.5, 1.0, 0, 100
                            ]
                        },
                        "geometricError": 10,
                        "refine": "ADD",
                        "children": []
                    }
                }
                
                # Add each b3dm file as a child in the tileset
                for i, gltf_file in enumerate(gltf_files):
                    basename = os.path.basename(gltf_file)
                    b3dm_name = os.path.splitext(basename)[0] + ".b3dm"
                    
                    child = {
                        "boundingVolume": {
                            "region": [
                                -2.0 + (i * 0.01), 0.5, -1.5 + (i * 0.01), 1.0, 0, 100
                            ]
                        },
                        "geometricError": 0,
                        "content": {
                            "uri": f"b3dm/{b3dm_name}"
                        }
                    }
                    
                    tileset_json["root"]["children"].append(child)
                
                # Write tileset.json
                with open(os.path.join(tiles_dir, "tileset.json"), "w") as f:
                    json.dump(tileset_json, f, indent=2)
                
            else:
                # Create a simple tileset with direct glTF references
                logger.info("Creating simple 3D Tiles tileset with glTF references")
                
                # Copy all glTF files
                for gltf_file in gltf_files:
                    basename = os.path.basename(gltf_file)
                    dst_path = os.path.join(tiles_dir, basename)
                    shutil.copy2(gltf_file, dst_path)
                    
                    # Also copy bin files
                    bin_src = os.path.splitext(gltf_file)[0] + ".bin"
                    if os.path.exists(bin_src):
                        bin_dst = os.path.splitext(dst_path)[0] + ".bin"
                        shutil.copy2(bin_src, bin_dst)
                
                # Create simple tileset.json
                tileset_json = {
                    "asset": {
                        "version": "1.0",
                        "generator": "Arcanum City Generator"
                    },
                    "geometricError": 100,
                    "root": {
                        "boundingVolume": {
                            "region": [
                                -2.0, 0.5, -1.5, 1.0, 0, 100
                            ]
                        },
                        "geometricError": 10,
                        "refine": "ADD",
                        "children": []
                    }
                }
                
                # Add each glTF file as a child in the tileset
                for i, gltf_file in enumerate(gltf_files):
                    basename = os.path.basename(gltf_file)
                    
                    child = {
                        "boundingVolume": {
                            "region": [
                                -2.0 + (i * 0.01), 0.5, -1.5 + (i * 0.01), 1.0, 0, 100
                            ]
                        },
                        "geometricError": 0,
                        "content": {
                            "uri": basename
                        }
                    }
                    
                    tileset_json["root"]["children"].append(child)
                
                # Write tileset.json
                with open(os.path.join(tiles_dir, "tileset.json"), "w") as f:
                    json.dump(tileset_json, f, indent=2)
            
            logger.info(f"Successfully created Cesium 3D Tiles at {tiles_dir}")
            
            return {
                "success": True,
                "exported_count": len(gltf_files),
                "tiles_dir": tiles_dir,
                "tileset_path": os.path.join(tiles_dir, "tileset.json"),
                "conversion_method": "3d-tiles-tools" if has_3d_tiles_tools else "simple"
            }
                
        except Exception as e:
            logger.error(f"Error in Cesium 3D Tiles export: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_mapbox(self, buildings_path: str, processed_dir: str, 
                     export_path: str, options: Dict = None) -> Dict[str, Any]:
        """
        Export buildings to Mapbox format.
        
        Args:
            buildings_path: Path to the buildings data file
            processed_dir: Directory with processed building data
            export_path: Output directory for the export
            options: Export options
            
        Returns:
            Dictionary with export results
        """
        logger.info(f"Exporting buildings to Mapbox format: {export_path}")
        
        try:
            # Create Mapbox directory
            mapbox_dir = os.path.join(export_path, "mapbox")
            os.makedirs(mapbox_dir, exist_ok=True)
            
            # Use geopackage directly to create a GeoJSON file
            try:
                import geopandas as gpd
                
                # Read buildings from GeoPackage
                buildings_gdf = gpd.read_file(buildings_path, layer='buildings')
                
                # Convert to GeoJSON
                geojson_path = os.path.join(mapbox_dir, "buildings.geojson")
                buildings_gdf.to_file(geojson_path, driver='GeoJSON')
                
                # Create a Mapbox tileset definition
                tileset_definition = {
                    "version": 1,
                    "name": "Arcanum Buildings",
                    "description": "Buildings generated by Arcanum City Generator",
                    "attribution": "Arcanum City Generator",
                    "minzoom": 10,
                    "maxzoom": 16,
                    "bounds": buildings_gdf.total_bounds.tolist(),
                    "sources": [
                        {
                            "name": "buildings",
                            "path": "buildings.geojson",
                            "type": "geojson"
                        }
                    ]
                }
                
                # Write tileset definition
                with open(os.path.join(mapbox_dir, "tileset.json"), "w") as f:
                    json.dump(tileset_definition, f, indent=2)
                
                # Write README with instructions
                readme_content = """
# Arcanum City for Mapbox

This directory contains files to create a Mapbox tileset for buildings generated by Arcanum City Generator.

## Files

- `buildings.geojson`: GeoJSON file containing building geometries
- `tileset.json`: Mapbox tileset definition

## Usage with Mapbox

1. Create a Mapbox account at https://www.mapbox.com/
2. Install the Mapbox CLI: https://github.com/mapbox/mapbox-cli-py
3. Authenticate with your Mapbox account
4. Create a tileset with the command:
   ```
   mapbox upload username.arcanum-buildings buildings.geojson
   ```
5. Use the tileset in your Mapbox application

For more detailed instructions, visit: https://docs.mapbox.com/help/tutorials/upload-a-custom-tileset/
                """
                
                with open(os.path.join(mapbox_dir, "README.md"), "w") as f:
                    f.write(readme_content)
                
                # Create a zip archive
                package_path = os.path.join(export_path, "ArcanumCity_Mapbox.zip")
                
                with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(mapbox_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, mapbox_dir)
                            zipf.write(file_path, rel_path)
                
                logger.info(f"Successfully created Mapbox export at {package_path}")
                
                return {
                    "success": True,
                    "exported_count": len(buildings_gdf),
                    "mapbox_dir": mapbox_dir,
                    "package_path": package_path
                }
                
            except ImportError as e:
                logger.error(f"Required library not found: {str(e)}")
                return {
                    "success": False,
                    "error": f"Required library not found: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error in Mapbox export: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Convenience functions

def export_buildings(buildings_path: str, processed_dir: str, format: str = "glb", 
                    export_dir: str = None, options: Dict = None) -> Dict[str, Any]:
    """
    Export buildings to the specified format.
    
    Args:
        buildings_path: Path to the buildings data file
        processed_dir: Directory with processed building data
        format: Export format (obj, glb, gltf, fbx, unity, unreal, cesium, mapbox)
        export_dir: Output directory for the export
        options: Format-specific export options
        
    Returns:
        Dictionary with export results
    """
    # Create export manager
    manager = ExportManager(export_dir)
    
    # Perform export
    return manager.export_buildings(buildings_path, processed_dir, format, options=options)


# Run a demo if this module is executed directly
if __name__ == "__main__":
    import argparse
    
    # Set up logging to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Arcanum Export Manager")
    parser.add_argument("--buildings", required=True, help="Path to buildings data file")
    parser.add_argument("--processed", required=True, help="Directory with processed building data")
    parser.add_argument("--format", default="glb", choices=["obj", "glb", "gltf", "fbx", "unity", "unreal", "cesium", "mapbox"],
                      help="Export format")
    parser.add_argument("--output", help="Output directory for the export")
    args = parser.parse_args()
    
    # Perform the export
    result = export_buildings(
        buildings_path=args.buildings,
        processed_dir=args.processed,
        format=args.format,
        export_dir=args.output
    )
    
    # Print result
    if result["success"]:
        print(f"Export successful! Exported {result.get('exported_count', 0)} buildings.")
        print(f"Output located at: {result.get('export_path')}")
    else:
        print(f"Export failed: {result.get('error', 'Unknown error')}")