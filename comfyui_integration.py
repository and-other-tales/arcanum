#!/usr/bin/env python3
"""
Arcanum City Gen ComfyUI Integration
-----------------------------------
This script provides integration with ComfyUI and X-Labs Flux ControlNet for transforming
real-life images into Arcanum style for Unity3D city generation.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import subprocess
import random
import shutil
from typing import Dict, List, Tuple, Any, Optional, Union

from PIL import Image

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arcanum_comfyui.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ArcanumComfyUIStyleTransformer:
    """Class responsible for transforming real-life images into Arcanum style using X-Labs Flux with ComfyUI."""
    
    def __init__(self, comfyui_path: str = None, max_batch_size: int = 4):
        """Initialize the ArcanumComfyUIStyleTransformer.
        
        Args:
            comfyui_path: Path to ComfyUI installation. If None, will use the default path.
            max_batch_size: Maximum number of images to process in a single batch.
        """
        # Set paths
        self.comfyui_path = comfyui_path or os.path.expanduser("~/ComfyUI")
        self.x_flux_path = os.path.join(self.comfyui_path, "custom_nodes/x-flux-comfyui")
        
        # Get the path to our local x-flux repository
        arcanum_dir = os.path.dirname(os.path.abspath(__file__))
        self.workflow_path = os.path.join(arcanum_dir, "x-flux-comfyui/workflows/canny_workflow.json")
        
        # Create necessary directories for ComfyUI
        self.models_path = os.path.join(self.comfyui_path, "models")
        self.input_dir = os.path.join(self.comfyui_path, "input")
        self.output_dir = os.path.join(self.comfyui_path, "output")
        
        # Ensure all required directories exist
        for dir_path in [self.models_path, self.input_dir, self.output_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Subdirectories for models
        os.makedirs(os.path.join(self.models_path, "xlabs/controlnets"), exist_ok=True)
        os.makedirs(os.path.join(self.models_path, "clip_vision"), exist_ok=True)
        os.makedirs(os.path.join(self.models_path, "T5Transformer"), exist_ok=True)
        os.makedirs(os.path.join(self.models_path, "vae"), exist_ok=True)
            
        # Check for X-Labs Flux installation
        self._check_x_flux_installation()
        
        # Store configuration
        self.max_batch_size = max_batch_size
        self.initialized = True
        logger.info("ArcanumComfyUIStyleTransformer initialization complete")
    
    def _check_x_flux_installation(self):
        """Check if X-Labs Flux ComfyUI is properly installed and set up."""
        logger.info("Checking X-Labs Flux ComfyUI installation...")
        
        # Check if X-Flux ComfyUI is in the proper location
        if not os.path.exists(self.x_flux_path):
            logger.info(f"X-Labs Flux ComfyUI not found at {self.x_flux_path}, installing...")
            self._install_x_flux_comfyui()
        
        # Check for required models
        required_models = {
            "flux1-dev-fp8.safetensors": "models",
            "flux-canny-controlnet.safetensors": "models/xlabs/controlnets",
            "clip_l.safetensors": "models/clip_vision",
            "t5xxl_fp16.safetensors": "models/T5Transformer",
            "ae.safetensors": "models/vae",
        }
        
        models_missing = False
        for model_file, model_dir in required_models.items():
            model_path = os.path.join(self.comfyui_path, model_dir, model_file)
            if not os.path.exists(model_path):
                models_missing = True
                logger.warning(f"Required model {model_file} not found at {model_path}")
        
        if models_missing:
            logger.warning("====================== IMPORTANT ======================")
            logger.warning("Some required models for X-Labs Flux ComfyUI are missing.")
            logger.warning("You need to download the following models from HuggingFace:")
            logger.warning("1. flux1-dev-fp8.safetensors - Place in: models/")
            logger.warning("2. flux-canny-controlnet.safetensors - Place in: models/xlabs/controlnets/")
            logger.warning("3. clip_l.safetensors - Place in: models/clip_vision/")
            logger.warning("4. t5xxl_fp16.safetensors - Place in: models/T5Transformer/")
            logger.warning("5. ae.safetensors - Place in: models/vae/")
            logger.warning("")
            logger.warning("Model sources:")
            logger.warning("- https://huggingface.co/XLabs-AI/flux-controlnet-collections")
            logger.warning("- https://huggingface.co/black-forest-labs/flux")
            logger.warning("- https://huggingface.co/openai/clip-vit-large-patch14")
            logger.warning("=====================================================")
    
    def _install_x_flux_comfyui(self):
        """Install X-Labs Flux ComfyUI from the git repository."""
        try:
            # Create custom_nodes directory if it doesn't exist
            os.makedirs(os.path.join(self.comfyui_path, "custom_nodes"), exist_ok=True)
            
            # Clone the repository
            logger.info("Cloning X-Labs Flux ComfyUI repository...")
            subprocess.run([
                "git", "clone", "https://github.com/XLabs-AI/x-flux-comfyui",
                os.path.join(self.comfyui_path, "custom_nodes/x-flux-comfyui")
            ], check=True)
            
            # Run setup script
            logger.info("Running setup script...")
            cwd = os.getcwd()
            os.chdir(os.path.join(self.comfyui_path, "custom_nodes/x-flux-comfyui"))
            subprocess.run(["python", "setup.py"], check=True)
            os.chdir(cwd)
            
            logger.info("X-Labs Flux ComfyUI installed successfully!")
        except Exception as e:
            logger.error(f"Error installing X-Labs Flux ComfyUI: {str(e)}")
            raise
            
    def _run_comfyui_workflow(self, 
                             input_image_path: str, 
                             output_dir: str,
                             workflow_path: str,
                             prompt: str,
                             negative_prompt: str = "",
                             strength: float = 0.8,
                             seed: int = None,
                             steps: int = 25) -> str:
        """Run a ComfyUI workflow to transform an image.
        
        Args:
            input_image_path: Path to the input image.
            output_dir: Directory to save the output image.
            workflow_path: Path to the ComfyUI workflow JSON file.
            prompt: The text prompt to guide the transformation.
            negative_prompt: Negative prompt to guide what to avoid.
            strength: ControlNet strength (0.0 to 1.0).
            seed: Random seed for reproducibility.
            steps: Number of denoising steps.
            
        Returns:
            Path to the output image.
        """
        try:
            # Generate a random seed if not provided
            if seed is None:
                seed = random.randint(0, 2**32 - 1)
                
            # Copy input image to ComfyUI input directory
            input_filename = os.path.basename(input_image_path)
            comfyui_input_path = os.path.join(self.input_dir, input_filename)
            shutil.copy2(input_image_path, comfyui_input_path)
            
            # Load the workflow JSON
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)
                
            # Modify the workflow for our specific use case
            # 1. Update the prompt in CLIPTextEncodeFlux nodes
            for node in workflow['nodes']:
                if node['type'] == 'CLIPTextEncodeFlux':
                    # Positive prompt node
                    if node['outputs'][0]['links'] and 18 in node['outputs'][0]['links']:
                        node['widgets_values'][0] = prompt
                        node['widgets_values'][1] = prompt
                    # Negative prompt node
                    elif node['outputs'][0]['links'] and 26 in node['outputs'][0]['links']:
                        node['widgets_values'][0] = negative_prompt
                        node['widgets_values'][1] = negative_prompt
                
                # 2. Update the input image
                elif node['type'] == 'LoadImage':
                    node['widgets_values'][0] = input_filename
                
                # 3. Update the ControlNet strength
                elif node['type'] == 'ApplyFluxControlNet':
                    node['widgets_values'][0] = strength
                
                # 4. Update the sampling parameters (seed and steps)
                elif node['type'] == 'XlabsSampler':
                    node['widgets_values'][0] = seed  # Set the seed
                    node['widgets_values'][2] = steps  # Set the number of steps
            
            # Save the modified workflow
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            modified_workflow_path = os.path.join(self.comfyui_path, f"arcanum_workflow_{timestamp}.json")
            with open(modified_workflow_path, 'w') as f:
                json.dump(workflow, f, indent=2)
            
            # Run ComfyUI with the modified workflow
            logger.info(f"Running ComfyUI workflow to transform {input_image_path}...")
            cmd = [
                "python", os.path.join(self.comfyui_path, "main.py"),
                "--lowvram", "--preview-method", "auto", "--use-split-cross-attention",
                "--workflow", modified_workflow_path
            ]
            subprocess.run(cmd, check=True)
            
            # Get the output image path
            # ComfyUI saves output images in the output directory with a timestamp
            # We need to find the most recent image in the output directory
            output_files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) 
                           if os.path.isfile(os.path.join(self.output_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]
            if output_files:
                # Sort by modification time, newest first
                output_files.sort(key=os.path.getmtime, reverse=True)
                latest_output = output_files[0]
                
                # Copy the output image to the specified output directory
                output_filename = f"arcanum_{os.path.basename(input_image_path)}"
                if output_filename.lower().endswith('.jpg') or output_filename.lower().endswith('.jpeg'):
                    output_filename = output_filename.rsplit('.', 1)[0] + '.png'
                final_output_path = os.path.join(output_dir, output_filename)
                shutil.copy2(latest_output, final_output_path)
                
                logger.info(f"Arcanum-styled image saved to: {final_output_path}")
                return final_output_path
            else:
                raise FileNotFoundError("No output image found from ComfyUI workflow")
                
        except Exception as e:
            logger.error(f"Error running ComfyUI workflow: {str(e)}")
            return f"Failed to transform image: {str(e)}"
    
    def transform_image(self, 
                        image_path: str, 
                        output_path: str,
                        prompt: str = "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical",
                        negative_prompt: str = "photorealistic, modern, contemporary, bright colors, clear sky",
                        strength: float = 0.8,
                        num_inference_steps: int = 25) -> str:
        """Transform a real-life image into Arcanum style using ComfyUI and Flux ControlNet.
        
        Args:
            image_path: Path to the input image.
            output_path: Path to save the transformed image.
            prompt: The prompt to guide the image transformation.
            negative_prompt: Negative prompt to guide what to avoid in the image.
            strength: ControlNet strength (0.0 to 1.0).
            num_inference_steps: Number of denoising steps to perform.
            
        Returns:
            Path to the transformed image.
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Check if the image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Input image not found: {image_path}")
            
            # Generate transformation using ComfyUI workflow
            logger.info(f"Transforming image to Arcanum style: {image_path}")
            return self._run_comfyui_workflow(
                input_image_path=image_path,
                output_dir=os.path.dirname(output_path),
                workflow_path=self.workflow_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                steps=num_inference_steps
            )
            
        except Exception as e:
            logger.error(f"Error transforming image to Arcanum style: {str(e)}")
            return f"Failed to transform image: {str(e)}"
    
    def batch_transform_images(self, 
                              image_paths: List[str], 
                              output_dir: str,
                              prompt: str = "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical",
                              negative_prompt: str = "photorealistic, modern, contemporary, bright colors, clear sky",
                              strength: float = 0.8,
                              num_inference_steps: int = 25) -> List[str]:
        """Transform multiple images in batches using ComfyUI and Flux ControlNet.
        
        Args:
            image_paths: List of paths to input images.
            output_dir: Directory to save transformed images.
            prompt: The prompt to guide the image transformation.
            negative_prompt: Negative prompt to guide what to avoid in the image.
            strength: ControlNet strength (0.0 to 1.0).
            num_inference_steps: Number of denoising steps to perform.
            
        Returns:
            List of paths to transformed images.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        # Process images one by one (ComfyUI doesn't support batch processing directly)
        for i, img_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1} of {len(image_paths)}: {img_path}")
            img_filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"arcanum_{img_filename}")
            result = self.transform_image(
                img_path, 
                output_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=num_inference_steps
            )
            if result and not result.startswith("Failed"):
                output_paths.append(result)
                
        return output_paths

# Main function for testing
def main():
    """Main function for testing."""
    parser = argparse.ArgumentParser(description="Arcanum Style Transformer using ComfyUI")
    parser.add_argument("--input", help="Input image path", required=True)
    parser.add_argument("--output", help="Output image path", required=True)
    parser.add_argument("--prompt", help="Text prompt for transformation", default="arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical")
    parser.add_argument("--negative", help="Negative prompt", default="photorealistic, modern, contemporary, bright colors, clear sky")
    parser.add_argument("--strength", help="ControlNet strength", type=float, default=0.8)
    parser.add_argument("--steps", help="Number of inference steps", type=int, default=25)
    parser.add_argument("--comfyui", help="Path to ComfyUI installation", default=None)
    args = parser.parse_args()
    
    # Create transformer
    transformer = ArcanumComfyUIStyleTransformer(comfyui_path=args.comfyui)
    
    # Transform image
    result = transformer.transform_image(
        image_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        negative_prompt=args.negative,
        strength=args.strength,
        num_inference_steps=args.steps
    )
    
    print(f"Transformation result: {result}")

if __name__ == "__main__":
    main()