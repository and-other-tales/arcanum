#!/usr/bin/env python3
"""
Download Flux 1-dev Controlnet Upscaler Model
-------------------------------------------
This script downloads the Flux.1-dev-Controlnet-Upscaler model from HuggingFace
and sets it up correctly in the ComfyUI directory structure.
"""

import os
import sys
import logging
import argparse
import requests
import shutil
from tqdm import tqdm
from pathlib import Path

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Model URLs and destinations
MODEL_CONFIGS = {
    "flux-upscaler-controlnet.safetensors": {
        "url": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/Flux.1-dev-Controlnet-Upscaler.safetensors",
        "path": "models/xlabs/controlnets"
    }
}

def download_file(url, destination):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192 # 8 KB chunks
        
        # Create progress bar
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    t.update(len(chunk))
        t.close()
        
        if total_size != 0 and t.n != total_size:
            logger.error("Error downloading file - size mismatch")
            return False
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return False

def setup_models(comfyui_path):
    """Download and set up the required models."""
    comfyui_path = os.path.expanduser(comfyui_path)
    
    if not os.path.exists(comfyui_path):
        logger.error(f"ComfyUI path does not exist: {comfyui_path}")
        return False
    
    successes = []
    
    for model_file, config in MODEL_CONFIGS.items():
        model_dir = os.path.join(comfyui_path, config["path"])
        model_path = os.path.join(model_dir, model_file)
        
        # Skip if model already exists
        if os.path.exists(model_path):
            logger.info(f"Model already exists: {model_path}")
            successes.append(True)
            continue
        
        # Create the directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download the model
        logger.info(f"Downloading {model_file} from {config['url']}")
        success = download_file(config["url"], model_path)
        successes.append(success)
        
        if success:
            logger.info(f"Successfully downloaded {model_file} to {model_path}")
        else:
            logger.error(f"Failed to download {model_file}")
    
    return all(successes)

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Download Flux.1-dev-Controlnet-Upscaler model")
    parser.add_argument("--comfyui-path", help="Path to ComfyUI installation", default="~/ComfyUI")
    args = parser.parse_args()
    
    logger.info("Starting download of Flux.1-dev-Controlnet-Upscaler model...")
    
    if setup_models(args.comfyui_path):
        logger.info("✅ All models downloaded successfully")
        return 0
    else:
        logger.error("❌ Failed to download some models")
        return 1

if __name__ == "__main__":
    sys.exit(main())