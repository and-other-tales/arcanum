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
import time
import json
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

# Google API related constants
GOOGLE_API_CHECK_INTERVAL = 10  # seconds
GOOGLE_API_TIMEOUT = 300  # seconds

# Model URLs and destinations
MODEL_CONFIGS = {
    "flux1-dev.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
        "path": "models"
    },
    "flux-canny-controlnet-v3.safetensors": {
        "url": "https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-canny-controlnet-v3.safetensors",
        "path": "models/xlabs/controlnets"
    },
    "flux-upscaler-controlnet.safetensors": {
        "url": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/diffusion_pytorch_model.safetensors",
        "path": "models/xlabs/controlnets"
    },
    "clip_l.safetensors": {
        "url": "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors",
        "path": "models/clip_vision"
    },
    "ae.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        "path": "models/vae"
    }
}

def download_file(url, destination, token=None, force_redownload=False):
    """Download a file with progress bar and support for resuming downloads.

    Args:
        url: URL to download from
        destination: Local path to save the file
        token: Hugging Face API token for accessing gated models
        force_redownload: If True, delete existing file and redownload
    """
    try:
        # Set up headers with token if provided
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            # Check if URL is from a repository that likely requires authentication
            auth_required_domains = [
                "black-forest-labs/FLUX.1-dev",
                "black-forest-labs/T5XXL-interleaved"
            ]
            if any(domain in url for domain in auth_required_domains):
                logger.warning(f"This model requires authentication. Please provide a valid Hugging Face token.")
                logger.warning(f"You can set it with --token or HUGGINGFACE_TOKEN environment variable")
                logger.warning(f"Get your token at: https://huggingface.co/settings/tokens")
                return False

        # Check if file already exists
        if os.path.exists(destination):
            if force_redownload:
                logger.info(f"Removing existing file for redownload: {destination}")
                os.remove(destination)
            else:
                # Check if the file is complete by making a HEAD request
                head_response = requests.head(url, headers=headers)
                if head_response.status_code == 401:
                    logger.error(f"Authentication failed for {url}")
                    logger.error(f"This model requires you to:")
                    logger.error(f"1. Login to Hugging Face: https://huggingface.co/login")
                    logger.error(f"2. Accept the model license agreement on the model page")
                    logger.error(f"3. Create an access token: https://huggingface.co/settings/tokens")
                    logger.error(f"4. Provide the token with --token or HUGGINGFACE_TOKEN environment variable")
                    return False

                try:
                    head_response.raise_for_status()
                    expected_size = int(head_response.headers.get('content-length', 0))
                    current_size = os.path.getsize(destination)

                    # If file size matches, consider it complete
                    if expected_size == current_size and expected_size > 0:
                        logger.info(f"File already exists and is complete: {destination}")
                        return True

                    # If file exists but is incomplete, resume download
                    if expected_size > current_size and current_size > 0:
                        logger.info(f"Resuming download for {destination} ({current_size}/{expected_size} bytes)")
                        headers['Range'] = f'bytes={current_size}-'
                        mode = 'ab'  # Append binary mode for resuming
                        initial_pos = current_size
                    else:
                        # File is larger than expected or corrupted
                        logger.warning(f"Existing file may be corrupted, redownloading: {destination}")
                        os.remove(destination)
                        mode = 'wb'  # Write binary mode for fresh download
                        initial_pos = 0
                except Exception as e:
                    logger.warning(f"Error checking file status: {str(e)}")
                    logger.warning(f"Will attempt to download file anyway")
                    mode = 'wb'
                    initial_pos = 0
        else:
            mode = 'wb'  # Write binary mode for fresh download
            initial_pos = 0

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)

        # Make request
        if 'Range' in headers:
            logger.info(f"Requesting file range starting at byte {initial_pos}")

        response = requests.get(url, stream=True, headers=headers)

        # Handle authentication failures specifically
        if response.status_code == 401:
            if any(domain in url for domain in ["huggingface.co"]):
                logger.error(f"Authentication failed for {url}")
                logger.error(f"This model requires you to:")
                logger.error(f"1. Login to Hugging Face: https://huggingface.co/login")
                logger.error(f"2. Accept the model license agreement on the model page")
                logger.error(f"3. Create an access token: https://huggingface.co/settings/tokens")
                logger.error(f"4. Provide the token with --token or HUGGINGFACE_TOKEN environment variable")
                return False

        response.raise_for_status()

        # Get file size for progress bar
        if 'content-length' in response.headers:
            file_size = int(response.headers.get('content-length', 0))
            total_size = file_size + initial_pos if 'Range' in headers else file_size
        else:
            total_size = 0  # Unknown size

        block_size = 8192 # 8 KB chunks

        # Create progress bar
        t = tqdm(total=total_size, initial=initial_pos, unit='iB', unit_scale=True)

        with open(destination, mode) as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    t.update(len(chunk))
        t.close()

        # Verify download if we know the expected size
        if total_size != 0 and t.n != total_size:
            logger.warning("Downloaded file size doesn't match expected size - may be incomplete")

            # Ask user if they want to delete and try again
            if not os.environ.get("SKIP_DOWNLOAD_PROMPTS") and not os.environ.get("SKIP_AUTH_CHECK"):
                retry = input(f"Downloaded file may be incomplete. Retry download? (y/n): ")
                if retry.lower() == 'y':
                    logger.info("Retrying download...")
                    return download_file(url, destination, token, force_redownload=True)
            return False
        return True
    except requests.exceptions.HTTPError as e:
        if hasattr(e, 'response') and e.response.status_code == 404:
            logger.error(f"Error downloading file: File not found (404) at URL: {url}")
        else:
            logger.error(f"Error downloading file: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")

        # If download was interrupted, don't delete the partial file
        # so it can be resumed later
        if os.path.exists(destination):
            logger.info(f"Partial download saved at {destination}")
            logger.info(f"Run the script again to resume downloading")
        return False

def setup_models(comfyui_path, hf_token=None, force_redownload=False):
    """Download and set up the required models.

    Args:
        comfyui_path: Path to ComfyUI installation
        hf_token: Hugging Face API token for accessing gated models
        force_redownload: If True, redownload all models even if they exist
    """
    comfyui_path = os.path.expanduser(comfyui_path)

    # Create comfyui_path if it doesn't exist
    if not os.path.exists(comfyui_path):
        logger.info(f"Creating ComfyUI directory: {comfyui_path}")
        try:
            os.makedirs(comfyui_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create ComfyUI directory: {str(e)}")
            return False

    # Check if token is available from environment if not provided
    if hf_token is None:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            logger.info("Using Hugging Face token from environment variable")
        else:
            logger.warning("No Hugging Face token provided. Models from black-forest-labs will fail to download.")
            logger.warning("Get your token at: https://huggingface.co/settings/tokens")
            logger.warning("Set it with --token or HUGGINGFACE_TOKEN environment variable")
            logger.warning("You must also accept the model terms at https://huggingface.co/black-forest-labs/FLUX.1-dev")

    successes = []
    skipped = []

    for model_file, config in MODEL_CONFIGS.items():
        model_dir = os.path.join(comfyui_path, config["path"])
        model_path = os.path.join(model_dir, model_file)

        # Create the directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Skip if model already exists and we're not forcing a redownload
        if os.path.exists(model_path) and not force_redownload:
            # Get file size to check if it's potentially complete
            if os.path.getsize(model_path) > 0:
                logger.info(f"Model already exists: {model_path}")
                skipped.append(model_file)
                successes.append(True)
                continue
            else:
                # Empty file, should be removed
                logger.warning(f"Found empty file at {model_path}, will redownload")
                os.remove(model_path)

        # Download the model
        logger.info(f"Downloading {model_file} from {config['url']}")
        success = download_file(config["url"], model_path, token=hf_token, force_redownload=force_redownload)
        successes.append(success)

        if success:
            logger.info(f"Successfully downloaded {model_file} to {model_path}")
        else:
            logger.error(f"Failed to download {model_file}")

            # Check if we have a partial download
            if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                logger.info(f"Partial download exists at {model_path}")
                logger.info(f"Run the script again to resume downloading")

    if skipped:
        logger.info(f"Skipped {len(skipped)} already existing models: {', '.join(skipped)}")

    return all(successes)

def check_google_api_access(api_name, credentials_path=None):
    """
    Check if a Google API is enabled and prompt to enable if needed.

    Args:
        api_name: Name of the Google API to check (e.g., "vision", "translate")
        credentials_path: Path to Google API credentials JSON file

    Returns:
        bool: True if API is available/enabled, False otherwise
    """
    try:
        # Try to import the google-auth package
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError:
            logger.warning(f"Google Cloud libraries not installed. Run: pip install google-auth google-api-python-client")

            if not os.environ.get("SKIP_AUTH_CHECK"):
                install = input("Install Google Cloud libraries now? (y/n): ")
                if install.lower() == 'y':
                    logger.info("Installing Google Cloud libraries...")
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-auth", "google-api-python-client"])
                    # Re-import after installation
                    import google.auth
                    from google.auth.transport.requests import Request
                else:
                    return False

        # Find credentials file if not provided
        if not credentials_path:
            # Check common locations
            common_locations = [
                os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
                os.path.expanduser("~/gcloud-credentials.json"),
                "./google-credentials.json",
                "./credentials.json"
            ]

            for loc in common_locations:
                if os.path.exists(loc):
                    credentials_path = loc
                    logger.info(f"Found Google credentials at: {credentials_path}")
                    break

            if not credentials_path:
                logger.warning(f"No Google credentials file found.")
                if not os.environ.get("SKIP_AUTH_CHECK"):
                    cred_path = input("Enter path to Google credentials file or press Enter to skip: ")
                    if cred_path and os.path.exists(cred_path):
                        credentials_path = cred_path
                    else:
                        return False
                else:
                    return False

        # Load credentials
        with open(credentials_path, 'r') as f:
            creds_data = json.load(f)

        # Extract project ID
        project_id = creds_data.get('project_id')
        if not project_id:
            logger.warning("No project_id found in credentials file")
            return False

        # Try to use the credentials to check API status
        try:
            from googleapiclient import discovery

            # Create a service to check API availability
            service = discovery.build('serviceusage', 'v1', credentials=None)

            # Format the API name correctly
            if not api_name.startswith('services/'):
                api_name = f"services/{api_name}.googleapis.com"

            # Check if API is enabled
            request = service.services().get(name=f"projects/{project_id}/{api_name}")
            response = request.execute()

            if response.get('state') != 'ENABLED':
                logger.warning(f"{api_name} is not enabled for this project")

                if not os.environ.get("SKIP_AUTH_CHECK"):
                    enable = input(f"Enable {api_name} API now? (y/n): ")
                    if enable.lower() == 'y':
                        # Enable the API
                        enable_request = service.services().enable(name=f"projects/{project_id}/{api_name}")
                        enable_response = enable_request.execute()

                        # Operation is async, so we need to wait for it to complete
                        op_name = enable_response.get('name')
                        if op_name:
                            logger.info(f"Enabling {api_name}...")

                            start_time = time.time()
                            while time.time() - start_time < GOOGLE_API_TIMEOUT:
                                check_request = service.operations().get(name=op_name)
                                check_response = check_request.execute()

                                if check_response.get('done'):
                                    logger.info(f"{api_name} API enabled successfully")
                                    return True

                                time.sleep(GOOGLE_API_CHECK_INTERVAL)
                                logger.info(f"Still waiting for {api_name} API to be enabled...")

                            logger.error(f"Timed out waiting for {api_name} API to be enabled")
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                logger.info(f"{api_name} API is enabled and ready to use")
                return True

        except Exception as e:
            logger.error(f"Error checking Google API status: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Error in Google API check: {str(e)}")
        return False

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Download X-Labs Flux models for ComfyUI")
    parser.add_argument("--comfyui-path", help="Path to ComfyUI installation", default="~/ComfyUI")
    parser.add_argument("--token", help="Hugging Face API token for accessing gated models")
    parser.add_argument("--skip-auth-check", action="store_true",
                       help="Skip authentication requirement check for models")
    parser.add_argument("--force-redownload", action="store_true",
                       help="Force redownload of models even if they exist")
    parser.add_argument("--no-interactive", action="store_true",
                       help="Non-interactive mode, don't prompt for confirmation")
    parser.add_argument("--google-credentials", help="Path to Google API credentials JSON file")
    parser.add_argument("--check-google-api", help="Check if a specific Google API is enabled (e.g., vision)")
    args = parser.parse_args()

    # Set environment variable for non-interactive mode
    if args.no_interactive or args.skip_auth_check:
        os.environ["SKIP_DOWNLOAD_PROMPTS"] = "1"
        os.environ["SKIP_AUTH_CHECK"] = "1"

    # Check Google API access if requested
    if args.check_google_api:
        api_enabled = check_google_api_access(args.check_google_api, args.google_credentials)
        if not api_enabled:
            logger.error(f"Google API '{args.check_google_api}' is not enabled or accessible")
            if not os.environ.get("SKIP_AUTH_CHECK"):
                cont = input("Continue without Google API access? (y/n): ")
                if cont.lower() != 'y':
                    logger.info("Download cancelled by user")
                    return 1
            else:
                logger.warning("Continuing without Google API access")
        else:
            logger.info(f"Google API '{args.check_google_api}' is enabled and ready to use")

    logger.info("Starting download of X-Labs Flux models...")

    # Print token status
    if args.token:
        logger.info("Hugging Face token provided via command line")
    elif os.environ.get("HUGGINGFACE_TOKEN"):
        logger.info("Hugging Face token found in environment variable")
    else:
        logger.warning("⚠️ No Hugging Face token provided - gated models will not download")
        logger.warning("Instructions to get models working:")
        logger.warning("1. Create a Hugging Face account: https://huggingface.co/join")
        logger.warning("2. Accept the model license: https://huggingface.co/black-forest-labs/FLUX.1-dev")
        logger.warning("3. Create a token: https://huggingface.co/settings/tokens")
        logger.warning("4. Run this script with --token=YOUR_TOKEN")
        if not args.skip_auth_check and input("Continue without token? [y/N]: ").lower() != 'y':
            logger.info("Download cancelled by user")
            return 1

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

    if setup_models(comfyui_path, args.token, force_redownload=args.force_redownload):
        logger.info("✅ All models downloaded successfully")
        return 0
    else:
        logger.error("❌ Failed to download some models")
        # Provide helpful summary if no token was provided
        if not args.token and not os.environ.get("HUGGINGFACE_TOKEN"):
            logger.info("\nTo fix authentication errors:")
            logger.info("1. Get a token: https://huggingface.co/settings/tokens")
            logger.info("2. Run: python download_flux_models.py --token=YOUR_TOKEN")
        logger.info("\nTo resume partial downloads, simply run the script again.")
        logger.info("To force redownload of all models: --force-redownload")
        logger.info("For non-interactive mode: --no-interactive")
        return 1

if __name__ == "__main__":
    sys.exit(main())