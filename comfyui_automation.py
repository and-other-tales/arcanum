#!/usr/bin/env python3
"""
Arcanum ComfyUI Automation
--------------------------
This script automates the setup and usage of ComfyUI with X-Labs Flux for transforming
real-life images into Arcanum style for Unity3D city generation.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import shutil
from pathlib import Path
import time
import re
from typing import Dict, List, Tuple, Any, Optional, Union

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arcanum_comfyui_automation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ArcanumComfyUIAutomation:
    """Class responsible for automating ComfyUI setup and batch processing images for Arcanum."""
    
    def __init__(self, comfyui_path=None, x_labs_path=None, output_dir=None):
        """Initialize the Arcanum ComfyUI Automation tool.
        
        Args:
            comfyui_path: Path to ComfyUI installation or where it should be installed.
            x_labs_path: Path to X-Labs Flux models.
            output_dir: Directory to save transformed images.
        """
        # Set default paths if not provided
        self.comfyui_path = comfyui_path or os.path.expanduser("~/ComfyUI")
        self.x_labs_path = x_labs_path or os.path.expanduser("~/x_labs_models")
        self.output_dir = output_dir or os.path.join(os.getcwd(), "arcanum_output")
        
        # Get the path to our local directories
        self.arcanum_dir = os.path.dirname(os.path.abspath(__file__))
        self.x_flux_local = os.path.join(self.arcanum_dir, "x-flux-comfyui")
        
        # Paths to model directories in ComfyUI
        self.model_paths = {
            "base": os.path.join(self.comfyui_path, "models"),
            "controlnet": os.path.join(self.comfyui_path, "models/xlabs/controlnets"),
            "clip_vision": os.path.join(self.comfyui_path, "models/clip_vision"),
            "t5": os.path.join(self.comfyui_path, "models/T5Transformer"),
            "vae": os.path.join(self.comfyui_path, "models/vae"),
        }
        
        # Required models and their destinations
        self.required_models = {
            "flux1-dev-fp8.safetensors": self.model_paths["base"],
            "flux-canny-controlnet.safetensors": self.model_paths["controlnet"],
            "clip_l.safetensors": self.model_paths["clip_vision"],
            "t5xxl_fp16.safetensors": self.model_paths["t5"],
            "ae.safetensors": self.model_paths["vae"],
        }
        
        # URL for downloading ComfyUI
        self.comfyui_url = "https://github.com/comfyanonymous/ComfyUI.git"
        
        # URL for X-Labs Flux
        self.x_flux_url = "https://github.com/XLabs-AI/x-flux-comfyui.git"
        
        # Default workflow file
        self.workflow_file = os.path.join(self.arcanum_dir, "x-flux-comfyui/workflows/canny_workflow.json")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def check_python_dependencies(self):
        """Check and install required Python dependencies."""
        logger.info("Checking Python dependencies...")
        
        required_packages = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.23.5",
            "pillow>=9.4.0",
            "tqdm>=4.64.1",
            "safetensors>=0.3.1",
            "scipy>=1.10.1",
            "transformers>=4.28.1",
            "aiohttp>=3.8.4",
            "diffusers>=0.16.0",
            "accelerate>=0.19.0"
        ]
        
        for package in required_packages:
            package_name = package.split(">=")[0]
            cmd = [sys.executable, "-m", "pip", "install", package]
            try:
                logger.info(f"Installing/upgrading {package}...")
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {str(e)}")
                return False
        
        logger.info("All Python dependencies installed successfully")
        return True
    
    def install_comfyui(self):
        """Install ComfyUI from GitHub if not already installed."""
        if os.path.exists(os.path.join(self.comfyui_path, "main.py")):
            logger.info(f"ComfyUI already installed at {self.comfyui_path}")
            return True
        
        try:
            logger.info(f"Installing ComfyUI to {self.comfyui_path}...")
            
            # Clone ComfyUI repository
            subprocess.run(["git", "clone", self.comfyui_url, self.comfyui_path], check=True)
            
            # Create necessary directories for models
            for path in self.model_paths.values():
                os.makedirs(path, exist_ok=True)
            
            logger.info("ComfyUI installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install ComfyUI: {str(e)}")
            return False
    
    def install_x_flux(self):
        """Install X-Labs Flux ComfyUI extension."""
        custom_nodes_path = os.path.join(self.comfyui_path, "custom_nodes")
        x_flux_path = os.path.join(custom_nodes_path, "x-flux-comfyui")
        
        # If already installed, return
        if os.path.exists(x_flux_path):
            logger.info(f"X-Labs Flux already installed at {x_flux_path}")
            return True
        
        try:
            logger.info("Installing X-Labs Flux ComfyUI extension...")
            
            # Create custom_nodes directory if it doesn't exist
            os.makedirs(custom_nodes_path, exist_ok=True)
            
            # Check if we have a local copy to use
            if os.path.exists(self.x_flux_local):
                logger.info(f"Using local X-Labs Flux from {self.x_flux_local}")
                shutil.copytree(self.x_flux_local, x_flux_path)
            else:
                # Clone X-Labs Flux repository
                logger.info("Cloning X-Labs Flux repository...")
                subprocess.run(["git", "clone", self.x_flux_url, x_flux_path], check=True)
            
            # Run setup.py in the x-flux-comfyui directory
            cwd = os.getcwd()
            os.chdir(x_flux_path)
            
            # Check if there's a setup.py file
            if os.path.exists("setup.py"):
                logger.info("Running X-Labs Flux setup script...")
                subprocess.run([sys.executable, "setup.py"], check=True)
            
            # Return to original directory
            os.chdir(cwd)
            
            logger.info("X-Labs Flux installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install X-Labs Flux: {str(e)}")
            return False
    
    def check_required_models(self):
        """Check if all required models are available and download them if specified."""
        logger.info("Checking required models...")
        
        missing_models = []
        
        for model_file, model_dir in self.required_models.items():
            model_path = os.path.join(model_dir, model_file)
            if not os.path.exists(model_path):
                missing_models.append((model_file, model_dir))
        
        if missing_models:
            logger.warning(f"Missing {len(missing_models)} required models")
            
            # Check if models are available in the x_labs_path
            if os.path.exists(self.x_labs_path):
                self._copy_models_from_directory(missing_models)
            
            # Check if we still have missing models
            still_missing = []
            for model_file, model_dir in missing_models:
                model_path = os.path.join(model_dir, model_file)
                if not os.path.exists(model_path):
                    still_missing.append((model_file, model_dir))
            
            if still_missing:
                logger.warning("====================== IMPORTANT ======================")
                logger.warning("Some required models for X-Labs Flux ComfyUI are missing.")
                logger.warning("You need to download the following models from HuggingFace:")
                
                for model_file, model_dir in still_missing:
                    logger.warning(f"- {model_file} - Place in: {os.path.relpath(model_dir, self.comfyui_path)}/")
                
                logger.warning("")
                logger.warning("Model sources:")
                logger.warning("- https://huggingface.co/XLabs-AI/flux-controlnet-collections")
                logger.warning("- https://huggingface.co/black-forest-labs/flux")
                logger.warning("- https://huggingface.co/openai/clip-vit-large-patch14")
                logger.warning("=====================================================")
                return False
        
        logger.info("All required models are available")
        return True
    
    def _copy_models_from_directory(self, missing_models):
        """Copy missing models from a local directory if available."""
        logger.info(f"Checking for models in {self.x_labs_path}...")
        
        for model_file, model_dir in missing_models:
            # Look for the model in x_labs_path
            source_path = None
            for root, _, files in os.walk(self.x_labs_path):
                if model_file in files:
                    source_path = os.path.join(root, model_file)
                    break
            
            if source_path:
                # Make sure the destination directory exists
                os.makedirs(model_dir, exist_ok=True)
                
                # Copy the model
                dest_path = os.path.join(model_dir, model_file)
                logger.info(f"Copying {model_file} from {source_path} to {dest_path}")
                shutil.copy2(source_path, dest_path)
    
    def create_default_workflow(self):
        """Create a default workflow file if it doesn't exist."""
        # Check if the workflow directory exists, create it if not
        workflow_dir = os.path.dirname(self.workflow_file)
        os.makedirs(workflow_dir, exist_ok=True)
        
        # If workflow file already exists, don't overwrite it
        if os.path.exists(self.workflow_file):
            logger.info(f"Workflow file already exists at {self.workflow_file}")
            return
        
        logger.info(f"Creating default workflow file at {self.workflow_file}...")
        
        # This is a simplified version of a typical X-Labs Flux workflow for ControlNet
        # The actual workflow would be more complex and should be created using the ComfyUI UI
        # and then saved as a JSON file
        default_workflow = {
            "version": 1,
            "nodes": [
                {
                    "id": 1,
                    "type": "LoadImage",
                    "pos": [200, 200],
                    "inputs": {},
                    "outputs": {"image": [2, 0]},
                    "widgets_values": ["input_image.jpg"]
                },
                {
                    "id": 2,
                    "type": "PreparePluxCannyControlnet",
                    "pos": [500, 200],
                    "inputs": {"image": [1, 0]},
                    "outputs": {"conditioning": [3, 0]},
                    "widgets_values": [100, 200]
                },
                {
                    "id": 3,
                    "type": "CLIPTextEncodeFlux",
                    "pos": [200, 400],
                    "inputs": {},
                    "outputs": {"conditioning": [4, 0]},
                    "widgets_values": ["arcanum gothic victorian fantasy steampunk architecture", "arcanum gothic victorian fantasy steampunk architecture"]
                },
                {
                    "id": 4,
                    "type": "CLIPTextEncodeFlux",
                    "pos": [200, 500],
                    "inputs": {},
                    "outputs": {"conditioning": [4, 1]},
                    "widgets_values": ["photorealistic, modern, contemporary", "photorealistic, modern, contemporary"]
                },
                {
                    "id": 5,
                    "type": "ApplyFluxControlNet",
                    "pos": [800, 300],
                    "inputs": {"conditioning": [3, 0], "controlnet": [2, 0]},
                    "outputs": {"modified_conditioning": [6, 0]},
                    "widgets_values": [0.8]
                },
                {
                    "id": 6,
                    "type": "XlabsSampler",
                    "pos": [1100, 300],
                    "inputs": {"positive": [5, 0], "negative": [4, 0]},
                    "outputs": {"image": [7, 0]},
                    "widgets_values": [42, "euler_ancestral", 25, 7.5, 1, 512, 512]
                },
                {
                    "id": 7,
                    "type": "SaveImage",
                    "pos": [1400, 300],
                    "inputs": {"image": [6, 0]},
                    "outputs": {},
                    "widgets_values": ["arcanum_output", "png"]
                }
            ]
        }
        
        # Save workflow to file
        with open(self.workflow_file, 'w') as f:
            json.dump(default_workflow, f, indent=2)
        
        logger.info("Default workflow created")
    
    def setup_environment(self):
        """Set up the complete environment for ComfyUI with X-Labs Flux."""
        logger.info("Setting up ComfyUI environment for Arcanum...")
        
        # 1. Check Python dependencies
        if not self.check_python_dependencies():
            logger.error("Failed to install required Python dependencies")
            return False
        
        # 2. Install ComfyUI
        if not self.install_comfyui():
            logger.error("Failed to install ComfyUI")
            return False
        
        # 3. Install X-Labs Flux
        if not self.install_x_flux():
            logger.error("Failed to install X-Labs Flux")
            return False
        
        # 4. Check required models
        if not self.check_required_models():
            logger.warning("Some required models are missing. You need to download them manually.")
            # Continue anyway, as the user might add the models later
        
        # 5. Create default workflow
        self.create_default_workflow()
        
        logger.info("ComfyUI environment setup completed successfully")
        return True
    
    def modify_workflow_for_image(self, input_image_path, output_dir, prompt, negative_prompt, strength, steps, seed):
        """Modify the workflow JSON for a specific image transformation."""
        # Load the workflow JSON
        try:
            with open(self.workflow_file, 'r') as f:
                workflow = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load workflow file: {str(e)}")
            return None
        
        # Get input image filename
        input_filename = os.path.basename(input_image_path)
        
        # Modify the workflow for our specific use case
        for node in workflow['nodes']:
            # Update the LoadImage node with our input image
            if node['type'] == 'LoadImage':
                node['widgets_values'][0] = input_filename
            
            # Update CLIPTextEncodeFlux nodes for prompts
            elif node['type'] == 'CLIPTextEncodeFlux':
                # Identify if this is the positive or negative prompt node based on its connections
                if 'outputs' in node and 'conditioning' in node['outputs']:
                    # Simplified check - in a real workflow, we would need a more robust way to identify nodes
                    if node['id'] == 3:  # Assuming id 3 is positive prompt
                        node['widgets_values'][0] = prompt
                        node['widgets_values'][1] = prompt
                    elif node['id'] == 4:  # Assuming id 4 is negative prompt
                        node['widgets_values'][0] = negative_prompt
                        node['widgets_values'][1] = negative_prompt
            
            # Update ControlNet strength
            elif node['type'] == 'ApplyFluxControlNet':
                node['widgets_values'][0] = strength
            
            # Update sampler parameters
            elif node['type'] == 'XlabsSampler':
                node['widgets_values'][0] = seed
                node['widgets_values'][2] = steps
            
            # Update SaveImage node
            elif node['type'] == 'SaveImage':
                output_filename = f"arcanum_{os.path.splitext(input_filename)[0]}"
                node['widgets_values'][0] = output_filename
        
        # Create a temporary workflow file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_workflow_path = os.path.join(self.comfyui_path, f"arcanum_workflow_{timestamp}.json")
        
        try:
            with open(temp_workflow_path, 'w') as f:
                json.dump(workflow, f, indent=2)
            return temp_workflow_path
        except Exception as e:
            logger.error(f"Failed to save modified workflow: {str(e)}")
            return None
    
    def transform_image(self, input_image_path, prompt=None, negative_prompt=None, strength=0.8, steps=25, seed=None):
        """Transform a single image using ComfyUI and X-Labs Flux."""
        if not os.path.exists(input_image_path):
            logger.error(f"Input image not found: {input_image_path}")
            return None
        
        # Set default prompt if not provided
        if prompt is None:
            prompt = "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical"
        
        # Set default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = "photorealistic, modern, contemporary, bright colors, clear sky"
        
        # Set random seed if not provided
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        logger.info(f"Transforming image: {input_image_path}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Seed: {seed}, Steps: {steps}, Strength: {strength}")
        
        # Copy input image to ComfyUI input directory
        input_dir = os.path.join(self.comfyui_path, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        input_filename = os.path.basename(input_image_path)
        comfyui_input_path = os.path.join(input_dir, input_filename)
        shutil.copy2(input_image_path, comfyui_input_path)
        
        # Modify workflow for this specific image
        temp_workflow_path = self.modify_workflow_for_image(
            input_image_path=input_image_path,
            output_dir=self.output_dir,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            seed=seed
        )
        
        if not temp_workflow_path:
            logger.error("Failed to create modified workflow")
            return None
        
        # Run ComfyUI with the modified workflow
        logger.info("Running ComfyUI to transform the image...")
        cmd = [
            sys.executable,
            os.path.join(self.comfyui_path, "main.py"),
            "--lowvram",
            "--preview-method", "auto",
            "--use-split-cross-attention",
            "--workflow", temp_workflow_path
        ]
        
        try:
            # Run ComfyUI
            subprocess.run(cmd, check=True)
            
            # Get the output image path
            # ComfyUI saves output images in the output directory with a timestamp
            output_files = [os.path.join(self.comfyui_path, "output", f) for f in os.listdir(os.path.join(self.comfyui_path, "output")) 
                           if os.path.isfile(os.path.join(self.comfyui_path, "output", f)) and f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if output_files:
                # Sort by modification time, newest first
                output_files.sort(key=os.path.getmtime, reverse=True)
                latest_output = output_files[0]
                
                # Copy the output image to our output directory
                output_filename = f"arcanum_{os.path.basename(input_image_path)}"
                if output_filename.lower().endswith('.jpg') or output_filename.lower().endswith('.jpeg'):
                    output_filename = output_filename.rsplit('.', 1)[0] + '.png'
                final_output_path = os.path.join(self.output_dir, output_filename)
                
                shutil.copy2(latest_output, final_output_path)
                logger.info(f"Transformed image saved to: {final_output_path}")
                
                # Clean up temporary workflow file
                if os.path.exists(temp_workflow_path):
                    os.remove(temp_workflow_path)
                
                return final_output_path
            else:
                logger.error("No output image found after ComfyUI execution")
                return None
            
        except Exception as e:
            logger.error(f"Error running ComfyUI: {str(e)}")
            return None
    
    def batch_transform_images(self, image_paths, prompt=None, negative_prompt=None, strength=0.8, steps=25):
        """Transform multiple images in a batch."""
        if not image_paths:
            logger.error("No input images provided")
            return []
        
        logger.info(f"Batch transforming {len(image_paths)} images...")
        
        output_paths = []
        for i, img_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1} of {len(image_paths)}: {img_path}")
            
            # Process each image with a different random seed for variation
            import random
            seed = random.randint(0, 2**32 - 1)
            
            result = self.transform_image(
                input_image_path=img_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                steps=steps,
                seed=seed
            )
            
            if result:
                output_paths.append(result)
        
        logger.info(f"Batch processing completed. Transformed {len(output_paths)} of {len(image_paths)} images.")
        return output_paths
    
    def run(self, input_path, batch=False, prompt=None, negative_prompt=None, strength=0.8, steps=25, seed=None):
        """Run the ComfyUI automation process."""
        # Setup environment if not already done
        if not os.path.exists(os.path.join(self.comfyui_path, "main.py")):
            logger.info("ComfyUI not found. Setting up environment...")
            if not self.setup_environment():
                logger.error("Failed to set up environment")
                return False
        
        # Process images
        if batch and os.path.isdir(input_path):
            # Process all images in directory
            image_paths = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
                image_paths.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
            
            if not image_paths:
                logger.error(f"No image files found in directory: {input_path}")
                return False
            
            results = self.batch_transform_images(
                image_paths=image_paths,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                steps=steps
            )
            
            return len(results) > 0
            
        elif os.path.isfile(input_path) and input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Process single image
            result = self.transform_image(
                input_image_path=input_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                steps=steps,
                seed=seed
            )
            
            return result is not None
        else:
            logger.error(f"Invalid input path: {input_path}")
            return False


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Arcanum ComfyUI Automation")
    parser.add_argument("--input", help="Input image path or directory", required=True)
    parser.add_argument("--output", help="Output directory for transformed images", default=None)
    parser.add_argument("--batch", help="Process all images in the input directory", action="store_true")
    parser.add_argument("--comfyui-path", help="Path to ComfyUI installation", default=None)
    parser.add_argument("--x-labs-path", help="Path to X-Labs models", default=None)
    parser.add_argument("--prompt", help="Text prompt for transformation", default=None)
    parser.add_argument("--negative", help="Negative prompt", default=None)
    parser.add_argument("--strength", help="ControlNet strength (0.0-1.0)", type=float, default=0.8)
    parser.add_argument("--steps", help="Number of denoising steps", type=int, default=25)
    parser.add_argument("--seed", help="Random seed for generation", type=int, default=None)
    parser.add_argument("--setup-only", help="Only set up the environment, don't process images", action="store_true")
    args = parser.parse_args()
    
    # Create automation instance
    automation = ArcanumComfyUIAutomation(
        comfyui_path=args.comfyui_path,
        x_labs_path=args.x_labs_path,
        output_dir=args.output
    )
    
    # Setup environment only
    if args.setup_only:
        logger.info("Setting up ComfyUI environment...")
        result = automation.setup_environment()
        if result:
            logger.info("ComfyUI environment setup completed successfully")
            return 0
        else:
            logger.error("Failed to set up ComfyUI environment")
            return 1
    
    # Run automation process
    result = automation.run(
        input_path=args.input,
        batch=args.batch,
        prompt=args.prompt,
        negative_prompt=args.negative,
        strength=args.strength,
        steps=args.steps,
        seed=args.seed
    )
    
    if result:
        logger.info("✅ Arcanum ComfyUI automation completed successfully")
        logger.info(f"Transformed images available in: {automation.output_dir}")
        return 0
    else:
        logger.error("❌ Arcanum ComfyUI automation failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    import glob  # Import here to avoid issues if missing
    sys.exit(main())