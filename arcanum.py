#!/usr/bin/env python3
"""
Arcanum City Generator
--------------------
This is the main entry point for the Arcanum city generator.
It provides a unified interface to the various modules and functionality.
"""

import os
import sys
import json
import logging
import argparse
import time
import platform
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import shutil

# Set up logging directory
log_dir = os.path.join(os.path.dirname(__file__), ".arcanum", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "arcanum.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".arcanum", "config")
os.makedirs(CONFIG_DIR, exist_ok=True)
CONFIG_FILE = os.path.join(CONFIG_DIR, "arcanum_config.json")

# Default configuration
DEFAULT_CONFIG = {
    "project_name": "Arcanum3D",
    "output_directory": "./arcanum_3d_output",
    "coordinate_system": "EPSG:4326",  # WGS84 (latitude/longitude)
    "center_point": [51.48182, -0.11258],  # Approximate center of Arcanum (London)
    "bounds": {
        "north": 51.49632,  # Northern extent (latitude)
        "south": 51.46732,  # Southern extent (latitude)
        "east": -0.09128,   # Eastern extent (longitude)
        "west": -0.13388    # Western extent (longitude)
    },
    "cell_size": 100,      # Grid cell size in meters
    "lod_levels": {
        "LOD0": 1000,      # Simple blocks for far viewing (1km+)
        "LOD1": 500,       # Basic buildings with accurate heights (500m-1km)
        "LOD2": 250,       # Buildings with roof structures (250-500m)
        "LOD3": 100,       # Detailed exteriors with architectural features (100-250m)
        "LOD4": 0          # Highly detailed models with facade elements (0-100m)
    },
    "comfyui_path": "./ComfyUI",
    "prompt": "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details",
    "negative_prompt": "photorealistic, modern, contemporary, bright colors, clear sky",
    "server_url": "gs://arcanum-tile-server",
    "credentials_path": "./key.json"
}

# Import our modules
try:
    from modules import osm, comfyui
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Arcanum modules: {str(e)}")
    MODULES_AVAILABLE = False

try:
    from integration_tools import google_3d_tiles_integration
    from integration_tools import storage_integration
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import integration tools: {str(e)}")
    INTEGRATION_AVAILABLE = False

#
# Core Functions
#

def setup_directory_structure(base_dir: str = None) -> str:
    """Create the necessary directory structure for the project."""
    if base_dir is None:
        base_dir = DEFAULT_CONFIG["output_directory"]
    
    # Create main directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create subdirectories for different data types
    subdirs = [
        "raw_data/satellite",
        "raw_data/lidar",
        "raw_data/vector",
        "raw_data/street_view",
        "processed_data/terrain",
        "processed_data/buildings",
        "processed_data/textures",
        "processed_data/vegetation",
        "3d_models/buildings",
        "3d_models/landmarks",
        "3d_models/street_furniture",
        "unity_assets/prefabs",
        "unity_assets/materials",
        "unity_assets/textures",
        "logs"
    ]
    
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
    
    logger.info("Directory structure set up complete")
    return base_dir

def download_osm_data(bounds: Dict[str, float], output_dir: str, grid_size: int = None, coordinate_system: str = "EPSG:4326", region: str = "london") -> Dict[str, Any]:
    """
    Download OpenStreetMap data using the Geofabrik downloader or fallback methods.

    Args:
        bounds: Dictionary with north, south, east, west boundaries
        output_dir: Directory to save output files
        grid_size: Size of grid cells in meters (uses grid-based approach if provided as fallback)
        coordinate_system: Coordinate system of the bounds
        region: Region name to download from Geofabrik (default: "london")

    Returns:
        Dictionary with status information
    """
    # Ensure OSM module is available
    if not MODULES_AVAILABLE:
        return {
            "success": False,
            "error": "Arcanum modules not available"
        }

    # Try to use Geofabrik downloader first (much more reliable)
    logger.info(f"Using Geofabrik OSM downloader for region: {region}")
    try:
        result = osm.geofabrik_bbox_download(
            bounds=bounds,
            output_dir=output_dir,
            region=region
        )

        if result.get("success", False):
            logger.info(f"Successfully downloaded OSM data from Geofabrik for {region}")
            return result
        else:
            logger.warning(f"Geofabrik download failed: {result.get('error', 'Unknown error')}")
            logger.warning("Falling back to standard OSM download methods")
    except Exception as e:
        logger.warning(f"Geofabrik download error: {str(e)}")
        logger.warning("Falling back to standard OSM download methods")

    # Fallback to original methods
    # Determine which approach to use based on grid_size
    if grid_size is not None and grid_size > 0:
        # Use grid-based approach for larger areas
        logger.info(f"Using grid-based OSM downloader with {grid_size}m cells")
        result = osm.download_osm_grid(
            bounds=bounds,
            output_dir=output_dir,
            cell_size_meters=grid_size
        )

        # Merge grid data
        if result.get("success_count", 0) > 0:
            merge_result = osm.merge_grid_data(output_dir)
            if merge_result.get("success", False):
                logger.info("Successfully merged grid data")
                return {
                    "success": True,
                    "message": f"Downloaded and merged OSM data for {result.get('success_count', 0)} grid cells",
                    "buildings_path": merge_result.get("buildings_path"),
                    "roads_path": merge_result.get("roads_path")
                }
            else:
                logger.warning(f"Grid data merge failed: {merge_result.get('error', 'Unknown error')}")

        return {
            "success": result.get("success_count", 0) > 0,
            "message": f"Downloaded OSM data for {result.get('success_count', 0)} grid cells"
        }
    else:
        # Use direct approach for smaller areas
        logger.info("Using direct OSM downloader")
        return osm.download_osm_data(
            bounds=bounds,
            output_dir=output_dir,
            coordinate_system=coordinate_system
        )

def transform_images(
    input_dir: str,
    output_dir: str,
    prompt: str = "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical",
    negative_prompt: str = "photorealistic, modern, contemporary, bright colors, clear sky",
    pattern: str = "*.jpg",
    use_controlnet: bool = True
) -> Dict[str, Any]:
    """
    Transform images in input_dir to Arcanum style and save to output_dir.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save transformed images
        prompt: Text prompt for image generation
        pattern: Glob pattern to match input images
        use_controlnet: Whether to use ControlNet for better structure preservation
        
    Returns:
        Dictionary with transformation results
    """
    # Ensure ComfyUI module is available
    if not MODULES_AVAILABLE:
        return {
            "success": False,
            "error": "Arcanum modules not available"
        }
    
    # Find all matching images
    import glob
    from pathlib import Path
    
    image_paths = []
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    if not image_paths:
        return {
            "success": False,
            "error": f"No images found in {input_dir}"
        }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Transform images
    logger.info(f"Transforming {len(image_paths)} images from {input_dir} to {output_dir}")
    result_paths = comfyui.batch_transform_images(
        image_paths=image_paths,
        output_dir=output_dir,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_controlnet=use_controlnet
    )
    
    return {
        "success": result_paths.get("success", False),
        "count": result_paths.get("count", 0),
        "total": len(image_paths),
        "output_dir": output_dir
    }

def run_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the complete Arcanum workflow.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with workflow results
    """
    try:
        # Set up directories
        base_dir = setup_directory_structure(config.get("output_directory"))
        
        # Step 1: Download OSM data
        logger.info("Step 1: Downloading OpenStreetMap data")
        osm_result = download_osm_data(
            bounds=config["bounds"],
            output_dir=base_dir,
            grid_size=config.get("cell_size"),
            coordinate_system=config.get("coordinate_system", "EPSG:4326")
        )
        
        if not osm_result.get("success", False):
            logger.warning(f"OSM data download failed: {osm_result.get('error', 'Unknown error')}")
        
        # Step 2: Download and transform satellite imagery
        # This is a placeholder for the actual implementation
        logger.info("Step 2: Satellite imagery processing not implemented in this version")
        
        # Step 3: Process buildings and roads
        # This is a placeholder for the actual implementation
        logger.info("Step 3: Building and road processing not implemented in this version")
        
        # Step 4: Apply Arcanum styling
        # This is a placeholder for the actual implementation
        logger.info("Step 4: Arcanum styling not fully implemented in this version")
        
        # Return workflow results
        return {
            "success": True,
            "output_dir": base_dir,
            "completed_at": datetime.now().isoformat(),
            "steps_completed": ["data_collection"],
            "steps_pending": ["satellite_processing", "building_processing", "styling"]
        }
    
    except Exception as e:
        logger.error(f"Workflow error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

#
# TUI Functions
#

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_header():
    """Print the Arcanum header."""
    header = """
    ▒▄▀▄▒█▀▄░▄▀▀▒▄▀▄░█▄░█░█▒█░█▄▒▄█
    ░█▀█░█▀▄░▀▄▄░█▀█░█▒▀█░▀▄█░█▒▀▒█

    City Generation Framework
"""
    print(header)

def load_config() -> Dict[str, Any]:
    """Load configuration from file or create default if not exists."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {CONFIG_FILE}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return DEFAULT_CONFIG
    else:
        logger.info(f"No configuration file found. Creating default at {CONFIG_FILE}")
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save configuration: {str(e)}")
        return False

def run_with_spinner(func: Callable, message: str, *args, **kwargs) -> Any:
    """Run a function with a spinner animation and return its result."""
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    i = 0
    
    # Start the task
    import threading
    result = {"value": None, "error": None, "completed": False}
    
    def worker():
        try:
            result["value"] = func(*args, **kwargs)
        except Exception as e:
            result["error"] = e
        finally:
            result["completed"] = True
    
    thread = threading.Thread(target=worker)
    thread.start()
    
    # Show spinner while task is running
    try:
        while not result["completed"]:
            print(f"\r{spinner_chars[i % len(spinner_chars)]} {message}...", end="")
            i += 1
            time.sleep(0.1)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return None
    finally:
        # Clear the spinner line
        print("\r" + " " * (len(message) + 10) + "\r", end="")
    
    # Handle result
    if result["error"]:
        print(f"\r❌ {message} failed: {str(result['error'])}")
        return None
    else:
        print(f"\r✅ {message} completed successfully")
        return result["value"]

def get_user_input(prompt: str, default: Any = None) -> str:
    """Get user input with a default value."""
    if default is not None:
        user_input = input(f"{prompt} [{default}]: ")
        return user_input if user_input.strip() else str(default)
    else:
        return input(f"{prompt}: ")

def process_entire_run(config: Dict[str, Any]) -> None:
    """Run the entire Arcanum processing pipeline."""
    clear_screen()
    print_header()
    print("\n🔄 Running complete Arcanum pipeline...\n")
    
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available. Please check your installation.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Step 1: Download OSM data
    print("Step 1: Downloading OpenStreetMap data...")
    osm_result = run_with_spinner(
        osm.geofabrik_bbox_download,
        "Downloading OpenStreetMap data from Geofabrik",
        bounds=config["bounds"],
        output_dir=config["output_directory"],
        region="london"  # Default to London, can be changed in config
    )
    
    if not osm_result or not osm_result.get("success", False):
        print("❌ Failed to download OSM data. Cannot continue.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Step 2: Download Tile Data & Imagery
    print("\nStep 2: Downloading 3D Tiles and imagery...")
    if INTEGRATION_AVAILABLE:
        tiles_result = run_with_spinner(
            google_3d_tiles_integration.fetch_tiles,
            "Downloading 3D tiles and imagery",
            config["bounds"],
            os.path.join(config["output_directory"], "raw_data")
        )
    else:
        print("⚠️ Integration tools not available. Skipping 3D tiles download.")
        tiles_result = {"success": False}
    
    # Step 3: Run transformation
    print("\nStep 3: Running image transformation...")
    # Find images in the raw_data directory
    imagery_dir = os.path.join(config["output_directory"], "raw_data", "satellite")
    transform_result = {"success": False}
    
    if os.path.exists(imagery_dir) and MODULES_AVAILABLE:
        transform_result = run_with_spinner(
            transform_images,
            "Transforming imagery",
            input_dir=imagery_dir,
            output_dir=os.path.join(config["output_directory"], "processed_data", "textures"),
            prompt=config["prompt"],
            negative_prompt=config["negative_prompt"]
        )
    else:
        print("⚠️ No imagery found or module not available. Skipping transformation.")
    
    # Step 4: Transfer to server
    print("\nStep 4: Transferring to server...")
    if INTEGRATION_AVAILABLE and config.get("server_url") and config.get("credentials_path"):
        upload_result = run_with_spinner(
            storage_integration.upload_directory,
            "Uploading to server",
            config["output_directory"],
            config["server_url"],
            config["credentials_path"]
        )
    else:
        print("⚠️ Server upload configuration missing or integration tools not available. Skipping upload.")
    
    print("\n✅ Arcanum processing completed!")
    print(f"\nOutput directory: {config['output_directory']}")
    input("\nPress Enter to return to the main menu...")

def download_mesh(config: Dict[str, Any]) -> None:
    """Download OSM data and build meshes."""
    clear_screen()
    print_header()
    print("\n🔄 Downloading mesh data...\n")
    
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available. Please check your installation.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Download OSM data from Geofabrik
    osm_result = run_with_spinner(
        osm.geofabrik_bbox_download,
        "Downloading OpenStreetMap data from Geofabrik",
        bounds=config["bounds"],
        output_dir=config["output_directory"],
        region="london"  # Default to London, can be changed in config later
    )
    
    if not osm_result or not osm_result.get("success", False):
        print(f"❌ Failed to download OSM data: {osm_result.get('error', 'Unknown error')}")
        input("\nPress Enter to return to the main menu...")
        return
    
    print(f"\n✅ Mesh data downloaded successfully")
    print(f"\nBuildings data: {osm_result.get('buildings_path', 'N/A')}")
    print(f"Roads data: {osm_result.get('roads_path', 'N/A')}")
    input("\nPress Enter to return to the main menu...")

def download_tiles(config: Dict[str, Any]) -> None:
    """Download 3D Tiles and Street View imagery."""
    clear_screen()
    print_header()
    print("\n🔄 Downloading 3D Tiles and Imagery...\n")
    
    if not INTEGRATION_AVAILABLE:
        print("❌ Required integration tools not available. Please check your installation.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Create directories
    raw_data_dir = os.path.join(config["output_directory"], "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Fetch 3D tiles
    tiles_result = run_with_spinner(
        google_3d_tiles_integration.fetch_tiles,
        "Downloading 3D tiles",
        config["bounds"],
        os.path.join(raw_data_dir, "tiles")
    )
    
    if not tiles_result or not tiles_result.get("success", False):
        print(f"❌ Failed to download 3D tiles: {tiles_result.get('error', 'Unknown error')}")
    else:
        print(f"✅ 3D tiles downloaded successfully")
    
    # Fetch Street View imagery if available
    print("\nDownloading Street View imagery...")
    center = config["center_point"]
    sv_result = run_with_spinner(
        google_3d_tiles_integration.fetch_street_view,
        "Downloading Street View images",
        center[0], center[1],
        os.path.join(raw_data_dir, "street_view")
    )
    
    if not sv_result or not sv_result.get("success", False):
        print(f"❌ Failed to download Street View imagery: {sv_result.get('error', 'Unknown error')}")
    else:
        print(f"✅ Street View imagery downloaded successfully")
    
    input("\nPress Enter to return to the main menu...")

def run_transformation(config: Dict[str, Any]) -> None:
    """Run the ComfyUI Flux transformation on images."""
    clear_screen()
    print_header()
    print("\n🔄 Running image transformation with Flux...\n")
    
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available. Please check your installation.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Check for source imagery
    source_dirs = [
        os.path.join(config["output_directory"], "raw_data", "satellite"),
        os.path.join(config["output_directory"], "raw_data", "street_view")
    ]
    
    image_paths = []
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            for f in os.listdir(source_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(source_dir, f))
    
    if not image_paths:
        print("❌ No source images found. Please download imagery first.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Create output directory
    output_dir = os.path.join(config["output_directory"], "processed_data", "textures")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if ComfyUI is available
    print(f"Found {len(image_paths)} images to process.")
    print(f"Using prompt: \"{config['prompt'][:50]}...\"")
    
    # Run transformation
    transform_result = run_with_spinner(
        transform_images,
        "Transforming images",
        input_dir=image_paths[0].split("/")[-2],
        output_dir=output_dir,
        prompt=config["prompt"],
        negative_prompt=config["negative_prompt"]
    )
    
    if not transform_result or not transform_result.get("success", False):
        print(f"❌ Image transformation failed: {transform_result.get('error', 'Unknown error')}")
        input("\nPress Enter to return to the main menu...")
        return
    
    print(f"\n✅ Successfully transformed {transform_result.get('count', 0)} of {len(image_paths)} images")
    print(f"\nOutput directory: {output_dir}")
    input("\nPress Enter to return to the main menu...")

def transfer_to_server(config: Dict[str, Any]) -> None:
    """Transfer processed data to the server."""
    clear_screen()
    print_header()
    print("\n🔄 Transferring data to server...\n")
    
    if not INTEGRATION_AVAILABLE:
        print("❌ Required integration tools not available. Please check your installation.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Check if server URL and credentials are provided
    if not config.get("server_url") or not config.get("credentials_path"):
        print("❌ Server URL or credentials path not configured.")
        print("Please update configuration first.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Check if credentials file exists
    if not os.path.exists(config["credentials_path"]):
        print(f"❌ Credentials file not found: {config['credentials_path']}")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Check if output directory exists
    if not os.path.exists(config["output_directory"]):
        print(f"❌ Output directory not found: {config['output_directory']}")
        print("Please generate data before transferring.")
        input("\nPress Enter to return to the main menu...")
        return
    
    # Transfer data
    upload_result = run_with_spinner(
        storage_integration.upload_directory,
        "Uploading to server",
        config["output_directory"],
        config["server_url"],
        config["credentials_path"]
    )
    
    if not upload_result or not upload_result.get("success", False):
        print(f"❌ Transfer failed: {upload_result.get('error', 'Unknown error')}")
        input("\nPress Enter to return to the main menu...")
        return
    
    print(f"\n✅ Data successfully transferred to {config['server_url']}")
    input("\nPress Enter to return to the main menu...")

def edit_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Edit the configuration settings."""
    clear_screen()
    print_header()
    print("\n⚙️ Edit Configuration\n")
    
    # Make a copy of the config to edit
    new_config = config.copy()
    
    while True:
        clear_screen()
        print_header()
        print("\n⚙️ Edit Configuration\n")
        
        print("Current Configuration:")
        print(f"1. Project Name: {new_config['project_name']}")
        print(f"2. Output Directory: {new_config['output_directory']}")
        print(f"3. Bounds (N,S,E,W): {new_config['bounds']['north']}, {new_config['bounds']['south']}, {new_config['bounds']['east']}, {new_config['bounds']['west']}")
        print(f"4. Cell Size: {new_config['cell_size']} meters")
        print(f"5. ComfyUI Path: {new_config['comfyui_path']}")
        print(f"6. Generation Prompt: {new_config['prompt'][:50]}...")
        print(f"7. Server URL: {new_config['server_url']}")
        print(f"8. Credentials Path: {new_config['credentials_path']}")
        print("\n9. Save and Return")
        print("0. Discard Changes")
        
        choice = input("\nEnter option number: ")
        
        if choice == "1":
            new_config["project_name"] = get_user_input("Enter project name", new_config["project_name"])
        
        elif choice == "2":
            new_config["output_directory"] = get_user_input("Enter output directory", new_config["output_directory"])
        
        elif choice == "3":
            print("\nEnter bounds (in WGS84/EPSG:4326 coordinates):")
            new_config["bounds"]["north"] = float(get_user_input("North latitude", new_config["bounds"]["north"]))
            new_config["bounds"]["south"] = float(get_user_input("South latitude", new_config["bounds"]["south"]))
            new_config["bounds"]["east"] = float(get_user_input("East longitude", new_config["bounds"]["east"]))
            new_config["bounds"]["west"] = float(get_user_input("West longitude", new_config["bounds"]["west"]))
            
            # Update center point based on new bounds
            new_config["center_point"][0] = (new_config["bounds"]["north"] + new_config["bounds"]["south"]) / 2
            new_config["center_point"][1] = (new_config["bounds"]["east"] + new_config["bounds"]["west"]) / 2
        
        elif choice == "4":
            new_config["cell_size"] = int(get_user_input("Enter cell size in meters", new_config["cell_size"]))
        
        elif choice == "5":
            new_config["comfyui_path"] = get_user_input("Enter ComfyUI path", new_config["comfyui_path"])
        
        elif choice == "6":
            new_config["prompt"] = get_user_input("Enter generation prompt", new_config["prompt"])
            new_config["negative_prompt"] = get_user_input("Enter negative prompt", new_config["negative_prompt"])
        
        elif choice == "7":
            new_config["server_url"] = get_user_input("Enter server URL (gs:// for Google Storage)", new_config["server_url"])
        
        elif choice == "8":
            new_config["credentials_path"] = get_user_input("Enter credentials file path", new_config["credentials_path"])
        
        elif choice == "9":
            if save_config(new_config):
                print("\n✅ Configuration saved successfully.")
                time.sleep(1)
                return new_config
            else:
                print("\n❌ Failed to save configuration.")
                time.sleep(1)
        
        elif choice == "0":
            return config
        
        else:
            print("\n⚠️ Invalid option. Please try again.")
            time.sleep(1)

def run_tui(config: Dict[str, Any]) -> int:
    """Run the Text User Interface."""
    # Load configuration
    
    while True:
        clear_screen()
        print_header()
        
        print("\nPlease select an option:\n")
        print("1. Process Entire Run")
        print("2. Download Mesh (OSM Buildings & Roads)")
        print("3. Download Tile Data & Imagery")
        print("4. Run Transformation (ComfyUI + Flux)")
        print("5. Transfer to Server")
        print("6. Configuration")
        print("\n0. Exit")
        
        try:
            choice = input("\nEnter option number: ")
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting Arcanum... Goodbye!")
            return 0
        
        if choice == "1":
            process_entire_run(config)
        
        elif choice == "2":
            download_mesh(config)
        
        elif choice == "3":
            download_tiles(config)
        
        elif choice == "4":
            run_transformation(config)
        
        elif choice == "5":
            transfer_to_server(config)
        
        elif choice == "6":
            config = edit_configuration(config)
        
        elif choice == "0":
            clear_screen()
            print("Thank you for using Arcanum City Generator!")
            return 0
        
        else:
            print("\n⚠️ Invalid option. Please try again.")
            time.sleep(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Arcanum 3D City Generator")
    
    # Define subcommands
    subparsers = parser.add_subparsers(dest="command", help="Operation mode")
    
    # TUI mode (default)
    tui_parser = subparsers.add_parser("start", help="Run interactive Text User Interface")
    
    # Original CLI workflow mode
    workflow_parser = subparsers.add_parser("generate", help="Run full Arcanum generation workflow")
    workflow_parser.add_argument("--output", help="Output directory", default="./arcanum_3d_output")
    workflow_parser.add_argument("--bounds", help="Area bounds (north,south,east,west)", default="51.5084,51.5064,-0.1258,-0.1298")
    workflow_parser.add_argument("--cell-size", type=int, help="Grid cell size in meters", default=100)
    
    # OSM download mode
    osm_parser = subparsers.add_parser("osm", help="Download OpenStreetMap data")
    osm_parser.add_argument("--output", help="Output directory", default="./arcanum_3d_output")
    osm_parser.add_argument("--bounds", help="Area bounds (north,south,east,west)", default="51.5084,51.5064,-0.1258,-0.1298")
    osm_parser.add_argument("--cell-size", type=int, help="Grid cell size in meters (0 for direct download)", default=100)
    
    # Style transform mode
    transform_parser = subparsers.add_parser("transform", help="Transform images to Arcanum style")
    transform_parser.add_argument("--input", help="Input directory containing images", required=True)
    transform_parser.add_argument("--output", help="Output directory for transformed images", default="./arcanum_output/final")
    transform_parser.add_argument("--prompt", help="Text prompt for image generation", 
                               default="arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details")
    transform_parser.add_argument("--no-controlnet", help="Disable ControlNet for faster processing", action="store_true")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # No command specified - default to TUI
    if args.command is None:
        return run_tui(config)
    
    # Run the selected command
    if args.command == "start":
        return run_tui(config)
        
    elif args.command == "generate":
        # Parse bounds
        bounds_values = args.bounds.split(",")
        if len(bounds_values) != 4:
            logger.error("Invalid bounds format. Use: north,south,east,west")
            return 1
        
        bounds = {
            "north": float(bounds_values[0]),
            "south": float(bounds_values[1]),
            "east": float(bounds_values[2]),
            "west": float(bounds_values[3])
        }
        
        # Update config
        config = DEFAULT_CONFIG.copy()
        config["output_directory"] = args.output
        config["bounds"] = bounds
        config["cell_size"] = args.cell_size
        
        # Run the workflow
        result = run_workflow(config)
        
        if result["success"]:
            logger.info(f"Arcanum generation completed successfully.")
            logger.info(f"Output directory: {result['output_dir']}")
            return 0
        else:
            logger.error(f"Arcanum generation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.command == "osm":
        # Parse bounds
        bounds_values = args.bounds.split(",")
        if len(bounds_values) != 4:
            logger.error("Invalid bounds format. Use: north,south,east,west")
            return 1
        
        bounds = {
            "north": float(bounds_values[0]),
            "south": float(bounds_values[1]),
            "east": float(bounds_values[2]),
            "west": float(bounds_values[3])
        }
        
        # Download OSM data
        result = download_osm_data(
            bounds=bounds,
            output_dir=args.output,
            grid_size=args.cell_size
        )
        
        if result.get("success", False):
            logger.info(f"OSM data download completed successfully.")
            logger.info(result.get("message", ""))
            return 0
        else:
            logger.error(f"OSM data download failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.command == "transform":
        # Transform images
        result = transform_images(
            input_dir=args.input,
            output_dir=args.output,
            prompt=args.prompt,
            use_controlnet=not args.no_controlnet
        )
        
        if result.get("success", False):
            logger.info(f"Transformed {result.get('count', 0)}/{result.get('total', 0)} images.")
            logger.info(f"Output directory: {result.get('output_dir')}")
            return 0
        else:
            logger.error(f"Image transformation failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nExiting Arcanum... Goodbye!")
        sys.exit(0)