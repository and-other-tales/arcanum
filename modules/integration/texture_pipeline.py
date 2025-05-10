#!/usr/bin/env python3
"""
Texture Pipeline Integration Module
---------------------------------
This module integrates the texture atlas manager with other Arcanum modules
to provide a unified workflow for texture management.
"""

import os
import sys
import logging
import json
import shutil
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import required modules
from modules.texture_atlas_manager import TextureAtlasManager
from modules.visualization import preview
from modules.osm import building_processor

try:
    from integration_tools import texture_projection
    from integration_tools import unity_material_pipeline
    from integration_tools import street_view_integration
    from integration_tools import road_network
    from integration_tools import coverage_verification
    from modules.integration import road_network_integration
    INTEGRATION_TOOLS_AVAILABLE = True
except ImportError:
    INTEGRATION_TOOLS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TexturePipeline:
    """
    Integrates texture management with building processing, providing a unified workflow.
    """

    def __init__(self, output_dir: str,
                texture_dir: str = None,
                atlas_size: tuple = (4096, 4096),
                building_texture_size: tuple = (1024, 1024),
                api_key: Optional[str] = None):
        """
        Initialize the texture pipeline.

        Args:
            output_dir: Base directory for outputs
            texture_dir: Directory containing input textures
            atlas_size: Size of texture atlases (width, height)
            building_texture_size: Size of individual building textures (width, height)
            api_key: Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)
        """
        self.output_dir = Path(output_dir)
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")

        # Set default texture directory if not provided
        if texture_dir is None:
            texture_dir = os.path.join(output_dir, "processed_data", "textures")
        self.texture_dir = Path(texture_dir)

        # Create output directories
        self.atlases_dir = self.output_dir / "atlases"
        self.buildings_dir = self.output_dir / "building_data"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.unity_dir = self.output_dir / "unity"
        self.street_view_dir = self.output_dir / "street_view"
        self.road_network_dir = self.output_dir / "road_network"

        os.makedirs(self.atlases_dir, exist_ok=True)
        os.makedirs(self.buildings_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.unity_dir, exist_ok=True)
        os.makedirs(self.street_view_dir, exist_ok=True)
        os.makedirs(self.road_network_dir, exist_ok=True)

        # Initialize texture atlas manager
        self.texture_atlas_manager = TextureAtlasManager(
            str(self.atlases_dir),
            atlas_size=atlas_size,
            building_texture_size=building_texture_size
        )

        # Initialize building processor
        self.building_processor = building_processor.OSMBuildingProcessor(
            str(self.buildings_dir),
            self.texture_atlas_manager,
            str(self.texture_dir)
        )

        # Initialize integration tools if available
        self.texture_projector = None
        self.material_pipeline = None
        self.road_network_integration = None

        if INTEGRATION_TOOLS_AVAILABLE:
            self.texture_projector = texture_projection.TextureProjector(
                output_dir=str(self.output_dir / "projected_textures"),
                parallel_processing=True
            )

            self.material_pipeline = unity_material_pipeline.UnityMaterialPipeline(
                str(self.output_dir / "materials"),
                str(self.texture_dir),
                str(self.unity_dir)
            )

            self.road_network_integration = road_network_integration.RoadNetworkIntegration(
                str(self.output_dir),
                api_key=self.api_key
            )

        logger.info(f"Texture pipeline initialized with output directory: {output_dir}")
        
    def process_buildings(self, buildings_file: str) -> Dict[str, Any]:
        """
        Process all buildings from a JSON file and create texture atlases.
        
        Args:
            buildings_file: Path to JSON file with building data
            
        Returns:
            Dictionary with processing results
        """
        # Process buildings using the building processor
        result = self.building_processor.process_osm_buildings(buildings_file)
        
        if not result.get("success", False):
            logger.error(f"Failed to process buildings: {result.get('error', 'Unknown error')}")
            return result
        
        # Generate atlases for all buildings
        atlas_result = self.texture_atlas_manager.generate_atlases()
        
        if not atlas_result.get("success", False):
            logger.error(f"Failed to generate atlases: {atlas_result.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": atlas_result.get("error", "Unknown error"),
                "buildings_processed": result.get("success_count", 0)
            }
        
        # Generate preview visualizations
        self._generate_previews(result.get("buildings", {}), atlas_result.get("atlases", []))
        
        # Create Unity materials if integration tools available
        if self.material_pipeline:
            material_result = self._create_unity_materials(result.get("buildings", {}))
        else:
            material_result = {"success": False, "error": "Unity material pipeline not available"}
        
        return {
            "success": True,
            "buildings_processed": result.get("success_count", 0),
            "atlases_generated": len(atlas_result.get("atlases", [])),
            "materials_created": material_result.get("success_count", 0),
            "buildings_dir": str(self.buildings_dir),
            "atlases_dir": str(self.atlases_dir),
            "unity_dir": str(self.unity_dir)
        }
    
    def assign_texture_to_building(self, building_id: str, texture_path: str) -> Dict[str, Any]:
        """
        Assign a texture to a specific building.
        
        Args:
            building_id: ID of the building
            texture_path: Path to the texture file
            
        Returns:
            Dictionary with assignment results
        """
        # Check if texture exists
        if not os.path.exists(texture_path):
            return {
                "success": False,
                "error": f"Texture not found: {texture_path}"
            }
        
        # Load building metadata
        metadata_path = self.buildings_dir / "metadata" / "buildings_metadata.json"
        building_metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    buildings_data = json.load(f)
                    building_metadata = buildings_data.get(building_id, {})
            except Exception as e:
                logger.warning(f"Failed to load building metadata: {str(e)}")
        
        # Assign texture to building
        result = self.texture_atlas_manager.assign_texture_to_building(
            building_id,
            texture_path,
            building_metadata
        )
        
        if not result.get("success", False):
            return result
        
        # Update building metadata with texture assignment
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    buildings_data = json.load(f)
                
                if building_id in buildings_data:
                    buildings_data[building_id]["texture_id"] = os.path.basename(texture_path)
                    buildings_data[building_id]["texture_path"] = texture_path
                    buildings_data[building_id]["texture_assigned"] = True
                    buildings_data[building_id]["uv_mapping"] = result.get("uv_mapping", {})
                    
                    with open(metadata_path, "w") as f:
                        json.dump(buildings_data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to update building metadata: {str(e)}")
        
        # Generate preview for this building
        self._generate_building_preview(building_id, texture_path)
        
        # Create Unity material if available
        if self.material_pipeline:
            material_result = self.material_pipeline.create_material_for_building(
                building_id,
                texture_path,
                os.path.splitext(os.path.basename(texture_path))[0],
                building_metadata
            )
        else:
            material_result = {"success": False, "error": "Unity material pipeline not available"}
        
        return {
            "success": True,
            "building_id": building_id,
            "texture_path": texture_path,
            "uv_mapping": result.get("uv_mapping", {}),
            "material_created": material_result.get("success", False),
            "material_path": material_result.get("material_path", None)
        }
    
    def regenerate_atlases(self) -> Dict[str, Any]:
        """
        Regenerate texture atlases for all buildings.
        
        Returns:
            Dictionary with atlas generation results
        """
        # Generate atlases
        atlas_result = self.texture_atlas_manager.generate_atlases()
        
        if not atlas_result.get("success", False):
            return atlas_result
        
        # Generate atlas previews
        for atlas_info in atlas_result.get("atlases", []):
            atlas_path = atlas_info.get("path")
            mapping_data = atlas_info.get("mapping", {})
            
            if atlas_path and os.path.exists(atlas_path):
                atlas_name = os.path.basename(atlas_path)
                preview_path = self.visualizations_dir / f"atlas_preview_{atlas_name}"
                
                preview.generate_atlas_preview(
                    atlas_path,
                    mapping_data,
                    str(preview_path),
                    show_grid=True
                )
        
        return {
            "success": True,
            "atlas_count": len(atlas_result.get("atlases", [])),
            "atlases_dir": str(self.atlases_dir),
            "previews_dir": str(self.visualizations_dir)
        }
    
    def project_street_view_textures(self, street_view_dir: str) -> Dict[str, Any]:
        """
        Project Street View images onto building models.

        Args:
            street_view_dir: Directory containing Street View images

        Returns:
            Dictionary with projection results
        """
        if not self.texture_projector:
            return {
                "success": False,
                "error": "Texture projection module not available"
            }

        if not os.path.exists(street_view_dir):
            return {
                "success": False,
                "error": f"Street View directory not found: {street_view_dir}"
            }

        # Load building metadata
        metadata_path = self.buildings_dir / "metadata" / "buildings_metadata.json"
        if not os.path.exists(metadata_path):
            return {
                "success": False,
                "error": "Building metadata not found"
            }

        try:
            with open(metadata_path, "r") as f:
                buildings_data = json.load(f)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load building metadata: {str(e)}"
            }

        # Process each building
        processed_buildings = 0
        failed_buildings = 0

        for building_id, building_data in buildings_data.items():
            # Check if building has a mesh
            mesh_path = building_data.get("mesh_path")
            if not mesh_path or not os.path.exists(mesh_path):
                logger.warning(f"Mesh not found for building {building_id}")
                failed_buildings += 1
                continue

            # Project textures
            result = self.texture_projector.process_building(
                building_id,
                mesh_path,
                street_view_dir,
                building_data
            )

            if result.get("success", False):
                processed_buildings += 1

                # Assign the projected texture to the building
                texture_path = result.get("texture_path")
                if texture_path and os.path.exists(texture_path):
                    self.assign_texture_to_building(building_id, texture_path)
            else:
                failed_buildings += 1
                logger.warning(f"Failed to project textures for building {building_id}: {result.get('error', 'Unknown error')}")

        return {
            "success": processed_buildings > 0,
            "processed_count": processed_buildings,
            "failed_count": failed_buildings,
            "total_count": len(buildings_data),
            "projected_textures_dir": str(self.output_dir / "projected_textures")
        }

    def load_road_network(self, osm_path: str, layer: str = "roads") -> Dict[str, Any]:
        """
        Load a road network from an OSM GeoPackage file.

        Args:
            osm_path: Path to OSM GeoPackage file
            layer: Layer name in the GeoPackage containing roads

        Returns:
            Dictionary with loading results
        """
        if not self.road_network_integration:
            return {
                "success": False,
                "error": "Road network integration module not available"
            }

        success = self.road_network_integration.load_road_network(osm_path, layer)

        if success:
            return {
                "success": True,
                "osm_path": osm_path,
                "layer": layer
            }
        else:
            return {
                "success": False,
                "error": "Failed to load road network"
            }

    def sample_road_network_points(self, interval: float = 50.0) -> Dict[str, Any]:
        """
        Sample points along all roads at a specified interval.

        Args:
            interval: Distance between sample points in meters

        Returns:
            Dictionary with sampling results
        """
        if not self.road_network_integration:
            return {
                "success": False,
                "error": "Road network integration module not available"
            }

        return self.road_network_integration.sample_points_along_roads(interval)

    def fetch_street_view_along_roads(self,
                                     osm_path: str = None,
                                     sampling_interval: float = 50.0,
                                     max_points: Optional[int] = None,
                                     panorama: bool = True,
                                     max_search_radius: int = 1000,
                                     max_workers: int = 4) -> Dict[str, Any]:
        """
        Fetch Street View images along road network.

        Args:
            osm_path: Path to OSM GeoPackage file (if not already loaded)
            sampling_interval: Distance between sample points in meters
            max_points: Maximum number of points to process (None for all)
            panorama: Whether to fetch full panorama
            max_search_radius: Maximum radius to search for nearby imagery
            max_workers: Maximum number of worker threads

        Returns:
            Dictionary with results
        """
        if not self.road_network_integration:
            return {
                "success": False,
                "error": "Road network integration module not available"
            }

        # Load road network if provided and not already loaded
        if osm_path and not hasattr(self.road_network_integration, 'road_network'):
            success = self.road_network_integration.load_road_network(osm_path)
            if not success:
                return {
                    "success": False,
                    "error": f"Failed to load road network from {osm_path}"
                }

        # Fetch Street View images
        result = self.road_network_integration.fetch_street_view_along_roads(
            sampling_interval=sampling_interval,
            max_points=max_points,
            panorama=panorama,
            max_search_radius=max_search_radius,
            max_workers=max_workers
        )

        if result.get("success", False):
            # If successful, update the street_view_dir to use for texture projection
            self.street_view_dir = Path(result.get("output_dir", str(self.street_view_dir)))

        return result

    def project_street_view_from_roads(self,
                                     osm_path: str,
                                     sampling_interval: float = 50.0,
                                     max_points: Optional[int] = None,
                                     panorama: bool = True,
                                     max_search_radius: int = 1000,
                                     max_workers: int = 4) -> Dict[str, Any]:
        """
        Fetch Street View images along roads and project them onto buildings.

        Args:
            osm_path: Path to OSM GeoPackage file
            sampling_interval: Distance between sample points in meters
            max_points: Maximum number of points to process (None for all)
            panorama: Whether to fetch full panorama or regular images
            max_search_radius: Maximum radius to search for nearby imagery
            max_workers: Maximum number of worker threads

        Returns:
            Dictionary with results
        """
        # First, fetch Street View images along roads
        fetch_result = self.fetch_street_view_along_roads(
            osm_path=osm_path,
            sampling_interval=sampling_interval,
            max_points=max_points,
            panorama=panorama,
            max_search_radius=max_search_radius,
            max_workers=max_workers
        )

        if not fetch_result.get("success", False):
            return fetch_result

        # Then project the collected images onto buildings
        street_view_dir = fetch_result.get("output_dir", str(self.street_view_dir))

        projection_result = self.project_street_view_textures(street_view_dir)

        return {
            "success": projection_result.get("success", False),
            "fetched_images": fetch_result.get("images_downloaded", 0),
            "processed_buildings": projection_result.get("processed_count", 0),
            "failed_buildings": projection_result.get("failed_count", 0),
            "street_view_dir": street_view_dir,
            "projected_textures_dir": projection_result.get("projected_textures_dir", ""),
            "road_network_points": fetch_result.get("points_processed", 0),
            "points_with_imagery": fetch_result.get("points_with_imagery", 0)
        }

    def verify_coverage(self, buildings_file: str) -> Dict[str, Any]:
        """
        Verify Street View coverage for buildings.

        Args:
            buildings_file: Path to JSON file with building data

        Returns:
            Dictionary with verification results
        """
        if not self.road_network_integration:
            return {
                "success": False,
                "error": "Road network integration module not available"
            }

        return self.road_network_integration.verify_coverage(buildings_file)
    
    def _generate_previews(self, buildings: Dict[str, Any], atlases: List[Dict[str, Any]]) -> None:
        """
        Generate preview visualizations for buildings and atlases.
        
        Args:
            buildings: Dictionary of building data
            atlases: List of atlas information
        """
        # Generate atlas previews
        for atlas_info in atlases:
            atlas_path = atlas_info.get("path")
            mapping_data = atlas_info.get("mapping", {})
            
            if atlas_path and os.path.exists(atlas_path):
                atlas_name = os.path.basename(atlas_path)
                preview_path = self.visualizations_dir / f"atlas_preview_{atlas_name}"
                
                preview.generate_atlas_preview(
                    atlas_path,
                    mapping_data,
                    str(preview_path),
                    show_grid=True
                )
        
        # Generate building previews
        for building_id, building_data in buildings.items():
            mesh_path = building_data.get("mesh_path")
            texture_path = building_data.get("texture_path")
            
            if mesh_path and texture_path and os.path.exists(mesh_path) and os.path.exists(texture_path):
                self._generate_building_preview(building_id, texture_path, mesh_path)
    
    def _generate_building_preview(self, building_id: str, texture_path: str, mesh_path: str = None) -> None:
        """
        Generate preview visualizations for a building.
        
        Args:
            building_id: ID of the building
            texture_path: Path to the texture file
            mesh_path: Path to the 3D model file (optional)
        """
        # If mesh path not provided, try to find it from metadata
        if mesh_path is None:
            metadata_path = self.buildings_dir / "metadata" / "buildings_metadata.json"
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        buildings_data = json.load(f)
                        building_data = buildings_data.get(building_id, {})
                        mesh_path = building_data.get("mesh_path")
                except Exception as e:
                    logger.warning(f"Failed to load building metadata: {str(e)}")
        
        # Generate texture preview
        texture_preview_path = self.visualizations_dir / f"texture_{building_id}.png"
        preview.generate_texture_preview(
            texture_path,
            str(texture_preview_path),
            label=f"Texture for {building_id}"
        )
        
        # Generate texture mapping preview if mesh is available
        if mesh_path and os.path.exists(mesh_path):
            mapping_preview_path = self.visualizations_dir / f"texture_mapping_{building_id}.png"
            preview.preview_texture_mapping(
                mesh_path,
                texture_path,
                str(mapping_preview_path)
            )
    
    def _create_unity_materials(self, buildings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Unity materials for buildings.
        
        Args:
            buildings: Dictionary of building data
            
        Returns:
            Dictionary with material creation results
        """
        if not self.material_pipeline:
            return {
                "success": False,
                "error": "Unity material pipeline not available"
            }
        
        success_count = 0
        
        for building_id, building_data in buildings.items():
            texture_path = building_data.get("texture_path")
            
            if not texture_path or not os.path.exists(texture_path):
                continue
            
            # Create material
            texture_name = os.path.splitext(os.path.basename(texture_path))[0]
            result = self.material_pipeline.create_material_for_building(
                building_id,
                texture_path,
                texture_name,
                building_data
            )
            
            if result.get("success", False):
                success_count += 1
        
        return {
            "success": success_count > 0,
            "success_count": success_count,
            "total_count": len(buildings),
            "materials_dir": str(self.output_dir / "materials")
        }


# Convenience functions for direct usage
def process_buildings_with_textures(buildings_file: str, output_dir: str, texture_dir: str = None) -> Dict[str, Any]:
    """
    Process buildings from a JSON file and assign textures.
    
    Args:
        buildings_file: Path to JSON file with building data
        output_dir: Base directory for outputs
        texture_dir: Directory containing input textures
        
    Returns:
        Dictionary with processing results
    """
    pipeline = TexturePipeline(output_dir, texture_dir)
    return pipeline.process_buildings(buildings_file)

def assign_texture(building_id: str, texture_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Assign a texture to a specific building.
    
    Args:
        building_id: ID of the building
        texture_path: Path to the texture file
        output_dir: Base directory for outputs
        
    Returns:
        Dictionary with assignment results
    """
    pipeline = TexturePipeline(output_dir)
    return pipeline.assign_texture_to_building(building_id, texture_path)

def project_street_view(buildings_file: str, street_view_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Project Street View images onto buildings.
    
    Args:
        buildings_file: Path to JSON file with building data
        street_view_dir: Directory containing Street View images
        output_dir: Base directory for outputs
        
    Returns:
        Dictionary with projection results
    """
    pipeline = TexturePipeline(output_dir)
    
    # Process buildings first
    result = pipeline.process_buildings(buildings_file)
    
    if not result.get("success", False):
        return result
    
    # Then project Street View images
    return pipeline.project_street_view_textures(street_view_dir)


# Run a demo if this module is run directly
if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Arcanum Texture Pipeline")
    parser.add_argument("--buildings", help="Path to JSON file with building data")
    parser.add_argument("--output", help="Base directory for outputs")
    parser.add_argument("--textures", help="Directory containing input textures")
    parser.add_argument("--street-view", help="Directory containing Street View images")
    parser.add_argument("--atlas-size", help="Size of texture atlases (width,height)", default="4096,4096")
    parser.add_argument("--assign", help="Assign a texture to a building (building_id:texture_path)")
    args = parser.parse_args()
    
    # Set default output directory
    if not args.output:
        args.output = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  "output")
    
    # Parse atlas size
    atlas_size = tuple(map(int, args.atlas_size.split(",")))
    
    # Create texture pipeline
    pipeline = TexturePipeline(args.output, args.textures, atlas_size=atlas_size)
    
    # Process buildings if requested
    if args.buildings:
        print(f"Processing buildings from {args.buildings}...")
        result = pipeline.process_buildings(args.buildings)
        
        if result.get("success", False):
            print(f"Successfully processed {result.get('buildings_processed')} buildings")
            print(f"Generated {result.get('atlases_generated')} texture atlases")
            print(f"Created {result.get('materials_created')} Unity materials")
            print(f"Output directory: {args.output}")
        else:
            print(f"Failed to process buildings: {result.get('error', 'Unknown error')}")
    
    # Project Street View if requested
    if args.street_view and args.buildings:
        print(f"Projecting Street View textures from {args.street_view}...")
        result = pipeline.project_street_view_textures(args.street_view)
        
        if result.get("success", False):
            print(f"Successfully projected textures for {result.get('processed_count')} of {result.get('total_count')} buildings")
            print(f"Output directory: {result.get('projected_textures_dir')}")
        else:
            print(f"Failed to project Street View textures: {result.get('error', 'Unknown error')}")
    
    # Assign texture if requested
    if args.assign:
        try:
            building_id, texture_path = args.assign.split(":", 1)
            print(f"Assigning texture {texture_path} to building {building_id}...")
            
            result = pipeline.assign_texture_to_building(building_id, texture_path)
            
            if result.get("success", False):
                print(f"Successfully assigned texture to building {building_id}")
                if result.get("material_created", False):
                    print(f"Created Unity material: {result.get('material_path')}")
            else:
                print(f"Failed to assign texture: {result.get('error', 'Unknown error')}")
        except ValueError:
            print("Invalid format for --assign. Use: building_id:texture_path")
    
    print("Done.")