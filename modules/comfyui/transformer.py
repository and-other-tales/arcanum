#!/usr/bin/env python3
"""
Arcanum Style Transformer Class
-----------------------------
This module provides the ArcanumStyleTransformer class for transforming
real-life images into Arcanum style using FLUX.1-dev and ComfyUI.
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

class ArcanumStyleTransformer:
    """
    Class responsible for transforming real-life images into Arcanum style 
    using ComfyUI and X-Labs Flux models.
    """

    def __init__(self, 
                comfyui_path: str = "./ComfyUI", 
                device: str = None,
                hf_token: str = None,
                model_id: str = "black-forest-labs/FLUX.1-dev",
                controlnet_model_id: str = "black-forest-labs/FLUX.1-dev-ControlNet",
                max_batch_size: int = 4):
        """
        Initialize the ArcanumStyleTransformer.

        Args:
            comfyui_path: Path to ComfyUI installation
            device: The device to use ("cuda", "cpu", etc.). If None, will use CUDA if available.
            hf_token: HuggingFace token for accessing gated models
            model_id: HuggingFace model ID for the Flux model to use
            controlnet_model_id: HuggingFace model ID for the ControlNet model
            max_batch_size: Maximum number of images to process in a single batch
        """
        self.comfyui_path = os.path.abspath(comfyui_path)
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
        self.model_id = model_id
        self.controlnet_model_id = controlnet_model_id
        self.max_batch_size = max_batch_size
        
        # Determine device if not provided
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing ArcanumStyleTransformer with device: {self.device}")
        
        # Set initialized flag to False until ComfyUI is properly set up
        self.initialized = False
        
        # Placeholder for the ComfyUI API/client
        self.comfyui_client = None
        
        # Try to initialize immediately if comfyui_path exists
        if os.path.exists(self.comfyui_path):
            self._initialize_comfyui()
        else:
            logger.warning(f"ComfyUI path not found: {self.comfyui_path}")
            
    def _initialize_comfyui(self):
        """Initialize ComfyUI and load required models."""
        try:
            # Import the ComfyUI automation module
            from .automation import setup_comfyui, get_comfyui_status
            
            # Check if ComfyUI is already running
            status = get_comfyui_status()
            if not status["running"]:
                # Start ComfyUI process
                logger.info(f"Starting ComfyUI from {self.comfyui_path}")
                setup_result = setup_comfyui(
                    comfyui_path=self.comfyui_path,
                    hf_token=self.hf_token,
                    model_id=self.model_id,
                    controlnet_model_id=self.controlnet_model_id
                )
                
                if not setup_result["success"]:
                    logger.error(f"Failed to set up ComfyUI: {setup_result['error']}")
                    return False
                    
                # Wait for ComfyUI to start
                max_retries = 10
                for i in range(max_retries):
                    status = get_comfyui_status()
                    if status["running"]:
                        break
                    logger.info(f"Waiting for ComfyUI to start (attempt {i+1}/{max_retries})...")
                    time.sleep(5)
                    
                if not status["running"]:
                    logger.error("Timed out waiting for ComfyUI to start")
                    return False
            
            # If we get here, ComfyUI should be running
            logger.info(f"ComfyUI is running at {status['url']}")
            self.initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import ComfyUI automation module: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error initializing ComfyUI: {str(e)}")
            return False

    def transform_image(self,
                       image_path: str,
                       output_path: str,
                       prompt: str = "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical",
                       negative_prompt: str = "photorealistic, modern, contemporary, bright colors, clear sky",
                       strength: float = 0.75,
                       use_controlnet: bool = True,
                       num_inference_steps: int = 20) -> str:
        """
        Transform a real-life image into Arcanum style.

        Args:
            image_path: Path to the input image.
            output_path: Path to save the transformed image.
            prompt: The prompt to guide the image transformation.
            negative_prompt: Negative prompt to guide what to avoid in the image.
            strength: Strength of the transformation (0.0 to 1.0).
            use_controlnet: Whether to use ControlNet for structure preservation.
            num_inference_steps: Number of denoising steps to perform.

        Returns:
            Path to the transformed image or error message.
        """
        try:
            # Ensure transformer is initialized
            if not self.initialized:
                success = self._initialize_comfyui()
                if not success:
                    return f"Failed to initialize ComfyUI for transformation"
            
            # Import the transform_image function
            from .automation import transform_image as transform_image_func
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Call the transform function
            result = transform_image_func(
                image_path=image_path,
                output_path=output_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                use_controlnet=use_controlnet,
                num_inference_steps=num_inference_steps
            )
            
            if result["success"]:
                logger.info(f"Successfully transformed image: {image_path} -> {output_path}")
                return output_path
            else:
                logger.error(f"Failed to transform image: {result['error']}")
                return f"Failed to transform image: {result['error']}"
                
        except Exception as e:
            logger.error(f"Error transforming image to Arcanum style: {str(e)}")
            return f"Failed to transform image: {str(e)}"

    def batch_transform_images(self,
                              image_paths: List[str],
                              output_dir: str,
                              prompt: str = "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical",
                              negative_prompt: str = "photorealistic, modern, contemporary, bright colors, clear sky",
                              strength: float = 0.75,
                              use_controlnet: bool = True,
                              num_inference_steps: int = 20) -> List[str]:
        """
        Transform multiple images in batches.

        Args:
            image_paths: List of paths to input images.
            output_dir: Directory to save transformed images.
            prompt: The prompt to guide the image transformation.
            negative_prompt: Negative prompt to guide what to avoid in the image.
            strength: Strength of the transformation (0.0 to 1.0).
            use_controlnet: Whether to use ControlNet for structure preservation.
            num_inference_steps: Number of denoising steps to perform.

        Returns:
            List of paths to transformed images.
        """
        try:
            # Import batch transform function
            from .automation import batch_transform_images as batch_transform_func
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Ensure transformer is initialized
            if not self.initialized:
                success = self._initialize_comfyui()
                if not success:
                    logger.error("Failed to initialize ComfyUI for batch transformation")
                    return []
            
            # Call the batch transform function
            result = batch_transform_func(
                image_paths=image_paths,
                output_dir=output_dir,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                use_controlnet=use_controlnet,
                num_inference_steps=num_inference_steps,
                max_batch_size=self.max_batch_size
            )
            
            if result["success"]:
                logger.info(f"Successfully transformed {len(result['output_paths'])} images")
                return result["output_paths"]
            else:
                logger.error(f"Batch transformation failed: {result['error']}")
                return []
                
        except Exception as e:
            logger.error(f"Error in batch transformation: {str(e)}")
            return []

    def __del__(self):
        """Clean up resources when the transformer is deleted."""
        # No need to explicitly stop ComfyUI server here,
        # as it's running in a separate process
        pass