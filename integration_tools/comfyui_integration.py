#!/usr/bin/env python3
"""
ComfyUI Integration Module for Arcanum
--------------------------------------
This module provides integration with FLUX.1-Canny-dev for image style transfer
and other AI-powered operations needed by the Arcanum generator.

This module is a compatibility layer over the modules/comfyui/automation.py
implementation to ensure existing code continues to work.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import the actual implementation from modules
try:
    from modules.comfyui.automation import (
        get_comfyui_status,
        start_comfyui,
        download_flux_models,
        setup_comfyui,
        load_workflow,
        transform_image,
        batch_transform_images
    )
    from modules.comfyui.transformer import ComfyUIStyleTransformer
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"Failed to import from modules/comfyui: {str(e)}")

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Forwards to the module implementation
if MODULES_AVAILABLE:
    # Re-export the imported transformer class to maintain compatibility
    ComfyUIStyleTransformer = ComfyUIStyleTransformer
else:
    # Fallback implementation - log a warning and provide a basic stub
    class ComfyUIStyleTransformer:
        """Stub class for ComfyUIStyleTransformer when modules are not available."""

        def __init__(self, comfyui_path=None, host="127.0.0.1", port=8188,
                   use_diffusers=True, use_blip2=True, device=None, hf_token=None):
            logger.warning("ComfyUIStyleTransformer is not available because modules/comfyui could not be imported")
            self.available = False

        def transform_image(self, image_path, output_path, **kwargs):
            logger.error("Cannot transform image - modules/comfyui is not available")
            return None

        def batch_transform_images(self, image_paths, output_dir, **kwargs):
            logger.error("Cannot transform images - modules/comfyui is not available")
            return []

# Forward module functions to maintain compatibility
def generate_image(prompt, negative_prompt=None, output_path=None, parameters=None):
    """Forward to modules.comfyui.automation.transform_image."""
    if MODULES_AVAILABLE:
        from modules.comfyui.automation import transform_image
        return transform_image(
            image_path=None,  # Will use a blank image
            output_path=output_path,
            prompt=prompt,
            negative_prompt=negative_prompt
        )
    else:
        logger.error("Cannot generate image - modules/comfyui is not available")
        return {"success": False, "error": "modules/comfyui is not available"}

def transform_image_file(image_path, output_path, prompt=None, negative_prompt=None, strength=0.75):
    """Forward to modules.comfyui.automation.transform_image."""
    if MODULES_AVAILABLE:
        return transform_image(
            image_path=image_path,
            output_path=output_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength
        )
    else:
        logger.error("Cannot transform image - modules/comfyui is not available")
        return {"success": False, "error": "modules/comfyui is not available"}

def batch_transform(image_paths, output_dir, prompt=None, negative_prompt=None, strength=0.75):
    """Forward to modules.comfyui.automation.batch_transform_images."""
    if MODULES_AVAILABLE:
        return batch_transform_images(
            image_paths=image_paths,
            output_dir=output_dir,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength
        )
    else:
        logger.error("Cannot batch transform images - modules/comfyui is not available")
        return {"success": False, "error": "modules/comfyui is not available"}