#!/usr/bin/env python3
"""
Style Manager for Arcanum
-----------------------
This module provides functionality for managing and customizing styles for Arcanum city generation.
Styles control the visual appearance of the generated city, including architecture, textures, and atmosphere.
"""

import os
import sys
import logging
import json
import shutil
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_STYLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 "styles")
USER_STYLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               ".arcanum", "styles")

# Ensure directories exist
os.makedirs(DEFAULT_STYLES_DIR, exist_ok=True)
os.makedirs(USER_STYLES_DIR, exist_ok=True)

# Default style definitions
DEFAULT_STYLES = {
    "arcanum_victorian": {
        "name": "Arcanum Victorian",
        "description": "A gothic victorian style with steampunk elements, perfect for creating an alternative London.",
        "prompt": "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, bronze and copper accents, mechanical elements, high contrast",
        "negative_prompt": "photorealistic, modern, contemporary, bright colors, clear sky, plain, simple",
        "parameters": {
            "seed": 42,
            "cfg_scale": 7.5,
            "steps": 30,
            "width": 1024,
            "height": 1024,
            "sampler": "euler_a",
            "scheduler": "karras",
            "denoising_strength": 0.7
        },
        "building_modifiers": {
            "standard": "victorian brick building with ornate details, imposing structure",
            "landmark": "grand gothic cathedral with spires and flying buttresses, stained glass",
            "commercial": "steampunk factory with bronze pipes and smoke stacks, industrial gothic",
            "residential": "victorian townhouse with ornate windows and gables, dark timber",
            "industrial": "industrial revolution factory, brick and iron, mechanical elements"
        },
        "color_palette": {
            "primary": "#2B1D0E",
            "secondary": "#703419",
            "accent": "#B87E45",
            "highlight": "#C99E10",
            "shadow": "#120A02"
        },
        "texture_modifiers": {
            "brick": "weathered dark brick with moss",
            "stone": "aged carved stone with intricate details",
            "wood": "dark polished wood with brass fittings",
            "metal": "tarnished copper and brass with patina",
            "glass": "amber stained glass with elaborate patterns"
        },
        "atmosphere_modifiers": {
            "weather": "foggy",
            "time_of_day": "dusk",
            "ambient_light": "warm gas lamps",
            "special_effects": "steam and mist"
        }
    },
    "arcanum_art_deco": {
        "name": "Arcanum Art Deco",
        "description": "A bold art deco style with fantasy elements, inspired by the golden age of cinema and jazz.",
        "prompt": "arcanum art deco fantasy architecture, gold and black, geometric patterns, opulent details, luxury materials, ziggurat forms, dramatic lighting, fantasy city",
        "negative_prompt": "photorealistic, modern, minimalist, rustic, medieval, run down",
        "parameters": {
            "seed": 42,
            "cfg_scale": 7.5,
            "steps": 30,
            "width": 1024,
            "height": 1024,
            "sampler": "euler_a",
            "scheduler": "karras",
            "denoising_strength": 0.7
        },
        "building_modifiers": {
            "standard": "geometric art deco building with strong vertical lines, gold details",
            "landmark": "grand art deco palace with stepped design, golden domes and statues",
            "commercial": "luxury art deco theater with neon lighting and ornate marquee",
            "residential": "art deco apartment building with zigzag patterns and metal details",
            "industrial": "streamlined factory with art deco motifs, chrome and glass"
        },
        "color_palette": {
            "primary": "#1A1A1A",
            "secondary": "#BF9B30",
            "accent": "#C92929",
            "highlight": "#E8D5B7",
            "shadow": "#000000"
        },
        "texture_modifiers": {
            "brick": "black and gold geometric patterns",
            "stone": "polished marble with inlaid metal designs",
            "wood": "ebony and gold lacquered paneling",
            "metal": "polished chrome and brass with geometric patterns",
            "glass": "stained glass with angular art deco patterns"
        },
        "atmosphere_modifiers": {
            "weather": "clear night",
            "time_of_day": "night",
            "ambient_light": "dramatic neon and spotlights",
            "special_effects": "searchlights and reflections"
        }
    },
    "arcanum_weird_west": {
        "name": "Arcanum Weird West",
        "description": "A blend of Wild West and dark fantasy, with supernatural elements and frontier aesthetics.",
        "prompt": "arcanum weird west fantasy architecture, frontier town, supernatural western, dark fantasy, weathered wood and stone, desert tones, occult symbols, fantasy wild west",
        "negative_prompt": "photorealistic, modern, contemporary, bright, clean, urban, futuristic",
        "parameters": {
            "seed": 42,
            "cfg_scale": 7.5,
            "steps": 30,
            "width": 1024,
            "height": 1024,
            "sampler": "euler_a",
            "scheduler": "karras",
            "denoising_strength": 0.7
        },
        "building_modifiers": {
            "standard": "weathered wooden building with frontier styling and occult symbols",
            "landmark": "imposing stone temple with western and supernatural elements",
            "commercial": "frontier saloon with mystical decorations and eerie lighting",
            "residential": "homestead cabin with protective sigils and strange modifications",
            "industrial": "steam-powered mill with otherworldly mechanisms and dark energy"
        },
        "color_palette": {
            "primary": "#5E3C2C",
            "secondary": "#A67153",
            "accent": "#D0A675",
            "highlight": "#BF304B",
            "shadow": "#2B1D1B"
        },
        "texture_modifiers": {
            "brick": "adobe with embedded arcane symbols",
            "stone": "rough-hewn stone with fossil impressions",
            "wood": "weathered barn wood with bullet holes and carvings",
            "metal": "rusted iron with mystical engravings",
            "glass": "dusty bottle glass with strange refractions"
        },
        "atmosphere_modifiers": {
            "weather": "dust storm",
            "time_of_day": "sunset",
            "ambient_light": "warm amber with eerie blue accents",
            "special_effects": "dust motes and strange mist"
        }
    }
}

def init_default_styles() -> None:
    """Initialize default styles if they don't exist."""
    for style_id, style_data in DEFAULT_STYLES.items():
        style_dir = os.path.join(DEFAULT_STYLES_DIR, style_id)
        os.makedirs(style_dir, exist_ok=True)
        
        # Write style definition to file
        style_file = os.path.join(style_dir, "style.json")
        if not os.path.exists(style_file):
            with open(style_file, "w") as f:
                json.dump(style_data, f, indent=2)
                logger.info(f"Created default style: {style_id}")

# Initialize default styles at module load time
init_default_styles()

def get_style(style_id: str) -> Dict:
    """
    Get a style definition by ID.
    
    Args:
        style_id: ID of the style to retrieve
        
    Returns:
        Style definition dictionary
    """
    # Check user styles first
    user_style_file = os.path.join(USER_STYLES_DIR, style_id, "style.json")
    if os.path.exists(user_style_file):
        try:
            with open(user_style_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user style {style_id}: {str(e)}")
    
    # Check default styles
    default_style_file = os.path.join(DEFAULT_STYLES_DIR, style_id, "style.json")
    if os.path.exists(default_style_file):
        try:
            with open(default_style_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading default style {style_id}: {str(e)}")
    
    # Style not found
    logger.warning(f"Style {style_id} not found")
    return None

def list_styles() -> List[Dict]:
    """
    List all available styles.
    
    Returns:
        List of style information dictionaries
    """
    styles = []
    
    # Get default styles
    for style_dir in os.listdir(DEFAULT_STYLES_DIR):
        if os.path.isdir(os.path.join(DEFAULT_STYLES_DIR, style_dir)):
            style_file = os.path.join(DEFAULT_STYLES_DIR, style_dir, "style.json")
            if os.path.exists(style_file):
                try:
                    with open(style_file, "r") as f:
                        style_data = json.load(f)
                        styles.append({
                            "id": style_dir,
                            "name": style_data.get("name", style_dir),
                            "description": style_data.get("description", ""),
                            "type": "default"
                        })
                except Exception as e:
                    logger.error(f"Error loading default style {style_dir}: {str(e)}")
    
    # Get user styles
    for style_dir in os.listdir(USER_STYLES_DIR):
        if os.path.isdir(os.path.join(USER_STYLES_DIR, style_dir)):
            style_file = os.path.join(USER_STYLES_DIR, style_dir, "style.json")
            if os.path.exists(style_file):
                try:
                    with open(style_file, "r") as f:
                        style_data = json.load(f)
                        styles.append({
                            "id": style_dir,
                            "name": style_data.get("name", style_dir),
                            "description": style_data.get("description", ""),
                            "type": "user"
                        })
                except Exception as e:
                    logger.error(f"Error loading user style {style_dir}: {str(e)}")
    
    return styles

def create_style(style_data: Dict, style_id: str = None) -> str:
    """
    Create a new custom style.
    
    Args:
        style_data: Style definition dictionary
        style_id: Optional ID for the style (auto-generated if not provided)
        
    Returns:
        ID of the created style
    """
    # Generate style ID if not provided
    if style_id is None:
        style_id = f"custom_{str(uuid.uuid4())[:8]}"
    
    # Make sure required fields are present
    required_fields = ["name", "prompt", "negative_prompt"]
    for field in required_fields:
        if field not in style_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Create style directory
    style_dir = os.path.join(USER_STYLES_DIR, style_id)
    os.makedirs(style_dir, exist_ok=True)
    
    # Write style definition to file
    style_file = os.path.join(style_dir, "style.json")
    with open(style_file, "w") as f:
        json.dump(style_data, f, indent=2)
    
    logger.info(f"Created custom style: {style_id}")
    return style_id

def update_style(style_id: str, style_data: Dict) -> bool:
    """
    Update an existing style.
    
    Args:
        style_id: ID of the style to update
        style_data: Updated style definition
        
    Returns:
        True if the style was updated, False otherwise
    """
    # Check if it's a default style
    default_style_path = os.path.join(DEFAULT_STYLES_DIR, style_id)
    if os.path.exists(default_style_path):
        # Copy the default style to user styles first
        user_style_path = os.path.join(USER_STYLES_DIR, style_id)
        os.makedirs(user_style_path, exist_ok=True)
        
        # Copy any additional files (e.g., example images)
        for filename in os.listdir(default_style_path):
            if filename != "style.json":  # Don't copy the original definition
                src_path = os.path.join(default_style_path, filename)
                dst_path = os.path.join(user_style_path, filename)
                shutil.copy2(src_path, dst_path)
        
        # Write the updated style definition
        style_file = os.path.join(user_style_path, "style.json")
        with open(style_file, "w") as f:
            json.dump(style_data, f, indent=2)
        
        logger.info(f"Updated style (copied from default): {style_id}")
        return True
    
    # Check if it's a user style
    user_style_path = os.path.join(USER_STYLES_DIR, style_id)
    if os.path.exists(user_style_path):
        # Write the updated style definition
        style_file = os.path.join(user_style_path, "style.json")
        with open(style_file, "w") as f:
            json.dump(style_data, f, indent=2)
        
        logger.info(f"Updated user style: {style_id}")
        return True
    
    # Style not found
    logger.warning(f"Style {style_id} not found for update")
    return False

def delete_style(style_id: str) -> bool:
    """
    Delete a custom style.
    
    Args:
        style_id: ID of the style to delete
        
    Returns:
        True if the style was deleted, False otherwise
    """
    # Only allow deleting user styles
    style_dir = os.path.join(USER_STYLES_DIR, style_id)
    if os.path.exists(style_dir):
        try:
            shutil.rmtree(style_dir)
            logger.info(f"Deleted custom style: {style_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting style {style_id}: {str(e)}")
            return False
    
    # Check if it's a default style (which can't be deleted)
    if os.path.exists(os.path.join(DEFAULT_STYLES_DIR, style_id)):
        logger.warning(f"Cannot delete default style: {style_id}")
        return False
    
    # Style not found
    logger.warning(f"Style {style_id} not found for deletion")
    return False

def clone_style(source_id: str, new_name: str = None) -> str:
    """
    Clone an existing style to create a new one.
    
    Args:
        source_id: ID of the style to clone
        new_name: Optional name for the new style
        
    Returns:
        ID of the cloned style
    """
    # Get the source style
    source_style = get_style(source_id)
    if source_style is None:
        raise ValueError(f"Source style {source_id} not found")
    
    # Create a copy of the style data
    new_style = source_style.copy()
    
    # Update name if provided
    if new_name:
        new_style["name"] = new_name
    else:
        new_style["name"] = f"Copy of {new_style.get('name', source_id)}"
    
    # Create a new style
    new_id = f"custom_{str(uuid.uuid4())[:8]}"
    return create_style(new_style, new_id)

def prepare_style(style_id: str) -> Dict:
    """
    Prepare a style for use with ComfyUI by formatting prompts and parameters.
    
    Args:
        style_id: ID of the style to prepare
        
    Returns:
        Dictionary with prepared style information
    """
    # Get the style definition
    style = get_style(style_id)
    if style is None:
        raise ValueError(f"Style {style_id} not found")
    
    # Extract the base prompt and negative prompt
    base_prompt = style.get("prompt", "")
    negative_prompt = style.get("negative_prompt", "")
    
    # Extract parameters
    parameters = style.get("parameters", {})
    
    # Extract building modifiers
    building_modifiers = style.get("building_modifiers", {})
    
    # Extract texture modifiers
    texture_modifiers = style.get("texture_modifiers", {})
    
    # Extract atmosphere modifiers
    atmosphere_modifiers = style.get("atmosphere_modifiers", {})
    
    # Extract color palette
    color_palette = style.get("color_palette", {})
    
    # Prepare complete data for ComfyUI
    prepared_style = {
        "id": style_id,
        "name": style.get("name", style_id),
        "base_prompt": base_prompt,
        "negative_prompt": negative_prompt,
        "parameters": parameters,
        "building_modifiers": building_modifiers,
        "texture_modifiers": texture_modifiers,
        "atmosphere_modifiers": atmosphere_modifiers,
        "color_palette": color_palette,
        
        # Pre-generated combined prompts for common building types
        "combined_prompts": {
            "standard": f"{base_prompt}, {building_modifiers.get('standard', '')}",
            "landmark": f"{base_prompt}, {building_modifiers.get('landmark', '')}",
            "commercial": f"{base_prompt}, {building_modifiers.get('commercial', '')}",
            "residential": f"{base_prompt}, {building_modifiers.get('residential', '')}",
            "industrial": f"{base_prompt}, {building_modifiers.get('industrial', '')}"
        }
    }
    
    return prepared_style

def generate_style_preview(style_id: str, output_path: str = None) -> str:
    """
    Generate a preview image for a style using ComfyUI.
    
    Args:
        style_id: ID of the style to preview
        output_path: Optional path to save the preview image
        
    Returns:
        Path to the generated preview image
    """
    # Get the style
    style = get_style(style_id)
    if style is None:
        raise ValueError(f"Style {style_id} not found")
    
    # Determine output path if not provided
    if output_path is None:
        style_dir = os.path.join(USER_STYLES_DIR, style_id) 
        if not os.path.exists(style_dir):
            style_dir = os.path.join(DEFAULT_STYLES_DIR, style_id)
        output_path = os.path.join(style_dir, "preview.png")
    
    # Run ComfyUI to generate the preview
    try:
        from modules.comfyui import automation
        
        # Prepare the style
        prepared_style = prepare_style(style_id)
        
        # Generate preview using building prompt for "standard" building
        preview_prompt = prepared_style["combined_prompts"]["standard"]
        
        # Add some extra context for a good preview
        preview_prompt += ", wide angle view, establishing shot, architectural visualization"
        
        # Run ComfyUI generation
        result = automation.generate_image(
            prompt=preview_prompt,
            negative_prompt=style.get("negative_prompt", ""),
            parameters=style.get("parameters", {}),
            output_path=output_path
        )
        
        if result and "image_path" in result:
            logger.info(f"Generated style preview for {style_id}: {result['image_path']}")
            return result["image_path"]
        else:
            logger.error(f"Failed to generate style preview for {style_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error generating style preview for {style_id}: {str(e)}")
        return None

def export_style(style_id: str, output_path: str = None) -> str:
    """
    Export a style to a JSON file for sharing.
    
    Args:
        style_id: ID of the style to export
        output_path: Optional path for the exported file
        
    Returns:
        Path to the exported file
    """
    # Get the style
    style = get_style(style_id)
    if style is None:
        raise ValueError(f"Style {style_id} not found")
    
    # Determine output path if not provided
    if output_path is None:
        output_path = os.path.join(os.getcwd(), f"{style_id}_export.json")
    
    # Add export metadata
    export_data = {
        "arcanum_style_export": True,
        "style_id": style_id,
        "style_data": style
    }
    
    # Write to file
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Exported style {style_id} to {output_path}")
    return output_path

def import_style(import_path: str, new_id: str = None) -> str:
    """
    Import a style from a JSON file.
    
    Args:
        import_path: Path to the exported style JSON file
        new_id: Optional ID for the imported style
        
    Returns:
        ID of the imported style
    """
    try:
        # Load the export file
        with open(import_path, "r") as f:
            import_data = json.load(f)
        
        # Verify it's a valid style export
        if not import_data.get("arcanum_style_export", False):
            raise ValueError("Not a valid Arcanum style export file")
        
        # Extract the style data
        style_data = import_data.get("style_data")
        if not style_data:
            raise ValueError("No style data found in export file")
        
        # Create the style with a new ID if provided
        if new_id:
            style_id = create_style(style_data, new_id)
        else:
            original_id = import_data.get("style_id", "imported")
            style_id = create_style(style_data, f"imported_{original_id}_{str(uuid.uuid4())[:4]}")
        
        logger.info(f"Imported style as {style_id}")
        return style_id
        
    except Exception as e:
        logger.error(f"Error importing style: {str(e)}")
        raise

def get_building_prompt(style_id: str, building_type: str, custom_modifiers: Dict = None) -> str:
    """
    Generate a specific prompt for a building based on the style and building type.
    
    Args:
        style_id: ID of the style to use
        building_type: Type of building (standard, landmark, commercial, etc.)
        custom_modifiers: Optional custom modifiers to override defaults
        
    Returns:
        Generated prompt for the building
    """
    # Prepare the style
    prepared_style = prepare_style(style_id)
    
    # Get the base combined prompt for the building type
    if building_type in prepared_style["combined_prompts"]:
        prompt = prepared_style["combined_prompts"][building_type]
    else:
        # Fall back to standard if building type not found
        prompt = prepared_style["combined_prompts"]["standard"]
    
    # Apply custom modifiers if provided
    if custom_modifiers:
        # Add texture modifiers
        if "texture" in custom_modifiers and custom_modifiers["texture"] in prepared_style["texture_modifiers"]:
            prompt += f", {prepared_style['texture_modifiers'][custom_modifiers['texture']]}"
        
        # Add atmosphere modifiers
        if "atmosphere" in custom_modifiers and custom_modifiers["atmosphere"] in prepared_style["atmosphere_modifiers"]:
            prompt += f", {prepared_style['atmosphere_modifiers'][custom_modifiers['atmosphere']]}"
        
        # Add any direct prompt additions
        if "additional_prompt" in custom_modifiers:
            prompt += f", {custom_modifiers['additional_prompt']}"
    
    return prompt


# Run a demo if this module is executed directly
if __name__ == "__main__":
    import argparse
    
    # Set up logging to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Arcanum Style Manager")
    parser.add_argument("--list", action="store_true", help="List all available styles")
    parser.add_argument("--get", help="Get a style by ID")
    parser.add_argument("--clone", help="Clone a style")
    parser.add_argument("--new-name", help="New name for cloned style")
    parser.add_argument("--delete", help="Delete a style by ID")
    parser.add_argument("--export", help="Export a style by ID")
    parser.add_argument("--import-file", help="Import a style from file")
    parser.add_argument("--generate-preview", help="Generate a preview image for a style")
    args = parser.parse_args()
    
    # Process commands
    if args.list:
        styles = list_styles()
        print(f"Found {len(styles)} styles:")
        for style in styles:
            print(f"  {style['id']} - {style['name']} ({style['type']})")
            print(f"    {style['description']}")
            print()
            
    elif args.get:
        style = get_style(args.get)
        if style:
            print(f"Style: {args.get}")
            print(json.dumps(style, indent=2))
        else:
            print(f"Style {args.get} not found")
            
    elif args.clone:
        try:
            new_id = clone_style(args.clone, args.new_name)
            print(f"Cloned style {args.clone} to {new_id}")
        except Exception as e:
            print(f"Error cloning style: {str(e)}")
            
    elif args.delete:
        if delete_style(args.delete):
            print(f"Deleted style {args.delete}")
        else:
            print(f"Failed to delete style {args.delete}")
            
    elif args.export:
        try:
            output_path = export_style(args.export)
            print(f"Exported style {args.export} to {output_path}")
        except Exception as e:
            print(f"Error exporting style: {str(e)}")
            
    elif args.import_file:
        try:
            style_id = import_style(args.import_file)
            print(f"Imported style as {style_id}")
        except Exception as e:
            print(f"Error importing style: {str(e)}")
            
    elif args.generate_preview:
        try:
            preview_path = generate_style_preview(args.generate_preview)
            if preview_path:
                print(f"Generated preview for {args.generate_preview} at {preview_path}")
            else:
                print(f"Failed to generate preview for {args.generate_preview}")
        except Exception as e:
            print(f"Error generating preview: {str(e)}")
            
    else:
        parser.print_help()