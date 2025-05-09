#!/usr/bin/env python3
"""
Download X-Labs Flux Models for ComfyUI
---------------------------------------
This script downloads all the required models for X-Labs Flux to work with ComfyUI
and sets them up correctly in the ComfyUI directory structure.
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
    "flux1-dev-fp8.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/FLUX.1-dev-fp8-pruned-q.safetensors",
        "path": "models"
    },
    "flux-canny-controlnet.safetensors": {
        "url": "https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/ControlNet/flux-canny/flux-canny-controlnet.safetensors",
        "path": "models/xlabs/controlnets"
    },
    "flux-upscaler-controlnet.safetensors": {
        "url": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/Flux.1-dev-Controlnet-Upscaler.safetensors",
        "path": "models/xlabs/controlnets"
    },
    "clip_l.safetensors": {
        "url": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors",
        "path": "models/clip_vision"
    },
    "t5xxl_fp16.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/T5XXL-interleaved/resolve/main/T5XXL-interleaved-fp16/t5xxl_fp16.safetensors",
        "path": "models/T5Transformer"
    },
    "ae.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        "path": "models/vae"
    }
}

def download_file(url, destination, token=None):
    """Download a file with progress bar.

    Args:
        url: URL to download from
        destination: Local path to save the file
        token: Hugging Face API token for accessing gated models
    """
    try:
        # Set up headers with token if provided
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.get(url, stream=True, headers=headers)
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

def setup_models(comfyui_path, hf_token=None):
    """Download and set up the required models.

    Args:
        comfyui_path: Path to ComfyUI installation
        hf_token: Hugging Face API token for accessing gated models
    """
    comfyui_path = os.path.expanduser(comfyui_path)

    if not os.path.exists(comfyui_path):
        logger.error(f"ComfyUI path does not exist: {comfyui_path}")
        return False

    # Check if token is available from environment if not provided
    if hf_token is None:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            logger.info("Using Hugging Face token from environment variable")
        else:
            logger.warning("No Hugging Face token provided. Some models may fail to download.")

    successes = []
    skipped = []

    for model_file, config in MODEL_CONFIGS.items():
        model_dir = os.path.join(comfyui_path, config["path"])
        model_path = os.path.join(model_dir, model_file)

        # Skip if model already exists
        if os.path.exists(model_path):
            logger.info(f"Model already exists: {model_path}")
            skipped.append(model_file)
            successes.append(True)
            continue

        # Create the directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Download the model
        logger.info(f"Downloading {model_file} from {config['url']}")
        success = download_file(config["url"], model_path, token=hf_token)
        successes.append(success)

        if success:
            logger.info(f"Successfully downloaded {model_file} to {model_path}")
        else:
            logger.error(f"Failed to download {model_file}")

    if skipped:
        logger.info(f"Skipped {len(skipped)} already existing models: {', '.join(skipped)}")

    return all(successes)

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Download X-Labs Flux models for ComfyUI")
    parser.add_argument("--comfyui-path", help="Path to ComfyUI installation", default="~/ComfyUI")
    parser.add_argument("--token", help="Hugging Face API token for accessing gated models")
    args = parser.parse_args()

    logger.info("Starting download of X-Labs Flux models...")

    # Get path to ComfyUI and convert to absolute path if needed
    comfyui_path = os.path.abspath(os.path.expanduser(args.comfyui_path))

    # Check if the path exists first
    if not os.path.exists(comfyui_path):
        logger.error(f"ComfyUI path does not exist: {comfyui_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Attempting to create directory: {comfyui_path}")
        try:
            os.makedirs(comfyui_path, exist_ok=True)
            logger.info(f"Successfully created directory: {comfyui_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {comfyui_path}: {str(e)}")
            return 1

    if setup_models(comfyui_path, args.token):
        logger.info("✅ All models downloaded successfully")
        return 0
    else:
        logger.error("❌ Failed to download some models")
        return 1

if __name__ == "__main__":
    sys.exit(main())