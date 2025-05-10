#!/bin/bash

# Arcanum Setup and Startup Script
# This script checks dependencies, installs ComfyUI (GPU or CPU version based on hardware),
# and runs the Arcanum generator.

# Set color codes for prettier output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define paths
ARCANUM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$ARCANUM_DIR/ComfyUI"
COMFYUI_CPU_DIR="$ARCANUM_DIR/ComfyUI-cpu"

# Define storage defaults
GCS_BUCKET="arcanum-maps"
CDN_URL="https://arcanum.fortunestold.co"
CLEANUP_ORIGINALS=true

# Print banner
echo -e "${BLUE}"
echo "    _                                     "
echo "   / \   _ __ ___ __ _ _ __  _   _ _ __ ___  "
echo "  / _ \ | '__/ __/ _' | '_ \| | | | '_ ' _ \ "
echo " / ___ \| | | (_| (_| | | | | |_| | | | | | |"
echo "/_/   \_\_|  \___\__,_|_| |_|\__,_|_| |_| |_|"
echo "          Arcanum Map Builder [Beta]"
echo -e "${NC}"
echo -e "${YELLOW}Arcanum Setup and Startup Script${NC}"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Python package is installed
package_installed() {
    python3 -c "import $1" >/dev/null 2>&1
}

# Function to check for GPU availability
check_gpu() {
    echo -e "${BLUE}Checking for GPU availability...${NC}"

    # Create temporary Python script to check GPU
    cat > /tmp/gpu_check.py << 'EOF'
import os
import sys
import subprocess

# Check environment variables
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
torch_device = os.environ.get('TORCH_DEVICE')

# First try torch if available
try:
    import torch
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print(f"CUDA available: Yes")
        print(f"CUDA version: {cuda_version}")
        print(f"GPU count: {device_count}")
        print(f"GPU name: {device_name}")
        sys.exit(0)  # GPU available
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) available: Yes")
        sys.exit(0)  # GPU available (Apple Silicon)
except ImportError:
    print("PyTorch not installed, checking alternative methods")

# Try to run nvidia-smi as fallback
try:
    nvidia_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                           stderr=subprocess.STDOUT,
                                           universal_newlines=True)
    print("NVIDIA-SMI detected GPU:")
    print(nvidia_output)
    sys.exit(0)  # GPU detected
except (subprocess.CalledProcessError, FileNotFoundError):
    # Try environment variables as last resort
    if (cuda_visible is not None and cuda_visible != '-1') or (torch_device is not None and 'cuda' in torch_device):
        print("GPU detected via environment variables")
        sys.exit(0)  # GPU detected via env vars
    else:
        print("No GPU acceleration available")
        sys.exit(1)  # No GPU
EOF

    # Run the GPU check script
    python3 /tmp/gpu_check.py
    GPU_STATUS=$?

    rm /tmp/gpu_check.py

    if [ $GPU_STATUS -eq 0 ]; then
        return 0  # GPU available
    else
        return 1  # No GPU
    fi
}

# Function to install dependencies
install_dependencies() {
    echo -e "${BLUE}Checking required dependencies...${NC}"
    
    # Check for Python 3
    if ! command_exists python3; then
        echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Python 3 is installed.${NC}"
    
    # Check for pip
    if ! command_exists pip3; then
        echo -e "${YELLOW}pip3 not found. Installing pip...${NC}"
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y python3-pip
        elif command_exists yum; then
            sudo yum install -y python3-pip
        elif command_exists brew; then
            brew install python3-pip
        else
            echo -e "${RED}Could not install pip. Please install pip manually.${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}pip3 is installed.${NC}"
    
    # Check for git
    if ! command_exists git; then
        echo -e "${YELLOW}git not found. Installing git...${NC}"
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y git
        elif command_exists yum; then
            sudo yum install -y git
        elif command_exists brew; then
            brew install git
        else
            echo -e "${RED}Could not install git. Please install git manually.${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}git is installed.${NC}"
    
    # Install required Python packages
    echo -e "${BLUE}Installing required Python packages...${NC}"
    
    # Install PyTorch based on GPU availability
    if check_gpu; then
        echo -e "${YELLOW}Installing PyTorch (GPU version)...${NC}"
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu113
    else
        echo -e "${YELLOW}Installing PyTorch (CPU version)...${NC}"
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other required packages from requirements.txt
    echo -e "${YELLOW}Installing dependencies from requirements.txt...${NC}"
    pip3 install -r "$ARCANUM_DIR/requirements.txt"
    
    # Install diffusers and controlnet_aux for FLUX.1-Canny-dev support
    echo -e "${YELLOW}Installing diffusers and controlnet_aux for FLUX.1-Canny-dev support...${NC}"
    pip3 install -U diffusers controlnet_aux
    
    # Install transformers for BLIP-2 image captioning
    echo -e "${YELLOW}Installing transformers for BLIP-2 image captioning...${NC}"
    pip3 install -U transformers accelerate

    # Install Google Cloud Storage for GCS integration
    echo -e "${YELLOW}Installing Google Cloud Storage client library...${NC}"
    pip3 install -U google-cloud-storage
    
    echo -e "${GREEN}Dependencies installed successfully.${NC}"
}

# Function to install ComfyUI (GPU version)
install_comfyui_gpu() {
    echo -e "${BLUE}Installing ComfyUI (GPU version)...${NC}"
    
    if [ -d "$COMFYUI_DIR" ]; then
        echo -e "${YELLOW}ComfyUI directory already exists. Updating...${NC}"
        cd "$COMFYUI_DIR" && git pull
    else
        echo -e "${YELLOW}Cloning ComfyUI repository...${NC}"
        git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    fi
    
    echo -e "${GREEN}ComfyUI (GPU version) installed successfully.${NC}"
}

# Function to install ComfyUI-cpu (CPU version)
install_comfyui_cpu() {
    echo -e "${BLUE}Installing ComfyUI-cpu (CPU optimized version)...${NC}"
    
    if [ -d "$COMFYUI_CPU_DIR" ]; then
        echo -e "${YELLOW}ComfyUI-cpu directory already exists. Updating...${NC}"
        cd "$COMFYUI_CPU_DIR" && git pull
    else
        echo -e "${YELLOW}Cloning ComfyUI-cpu repository...${NC}"
        git clone https://github.com/ArdeniusAI/ComfyUI-cpu.git "$COMFYUI_CPU_DIR"
    fi
    
    echo -e "${GREEN}ComfyUI-cpu installed successfully.${NC}"
}

# Function to set up ComfyUI and required models
setup_comfyui() {
    local comfyui_path=$1

    echo -e "${BLUE}Setting up ComfyUI environment...${NC}"

    # Ensure ComfyUI directory exists
    if [ ! -d "$comfyui_path" ]; then
        echo -e "${YELLOW}ComfyUI directory doesn't exist. Creating it...${NC}"
        mkdir -p "$comfyui_path"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to create directory. Trying with sudo...${NC}"
            sudo mkdir -p "$comfyui_path"
            sudo chown -R $(whoami) "$comfyui_path"
        fi
    fi

    # Check if we have write permissions to the directory
    if [ ! -w "$comfyui_path" ]; then
        echo -e "${YELLOW}No write permission to ComfyUI directory. Fixing permissions...${NC}"
        sudo chown -R $(whoami) "$comfyui_path"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to fix permissions. Please run the script with sudo or fix permissions manually.${NC}"
            return 1
        fi
    fi

    # Create required directories
    mkdir -p "$comfyui_path/models/x-labs/Flux"
    mkdir -p "$comfyui_path/models/clip_vision"
    mkdir -p "$comfyui_path/models/T5Transformer"
    mkdir -p "$comfyui_path/models/vae"
    mkdir -p "$comfyui_path/models/xlabs/controlnets"

    # Install additional requirements if needed
    if [ -f "$comfyui_path/requirements.txt" ]; then
        echo -e "${YELLOW}Installing ComfyUI requirements...${NC}"
        pip3 install -r "$comfyui_path/requirements.txt"
    fi

    echo -e "${GREEN}ComfyUI setup completed.${NC}"
}

# Function to run generator.py with appropriate ComfyUI path
run_generator() {
    local comfyui_path=$1

    echo -e "${BLUE}Starting Arcanum generator...${NC}"
    echo -e "${YELLOW}Using ComfyUI at: $comfyui_path${NC}"

    # Verify if the COMFYUI_PATH is valid
    if [ ! -d "$comfyui_path" ]; then
        echo -e "${RED}Invalid COMFYUI_PATH: $comfyui_path${NC}"
        exit 1
    fi

    # Set environment variable for generator.py to use the right ComfyUI path
    export COMFYUI_PATH="$comfyui_path"

    # Run the generator.py with full ComfyUI integration
    python3 "$ARCANUM_DIR/generator.py" --comfyui-path="$comfyui_path"
}

# Function to transform and upload generated content to GCS
transform_and_upload() {
    local comfyui_path=$1
    local bucket=$2
    local cdn_url=$3
    local cleanup=$4

    echo -e "${BLUE}Starting Arcanum transformation and upload...${NC}"

    # Check if output directories exist
    local satellite_dir="$ARCANUM_DIR/arcanum_3d_output/raw_data/satellite"
    local street_view_dir="$ARCANUM_DIR/arcanum_3d_output/raw_data/street_view"
    local textures_dir="$ARCANUM_DIR/arcanum_3d_output/processed_data/textures"

    # Process satellite imagery if available
    if [ -d "$satellite_dir" ] && [ "$(ls -A $satellite_dir)" ]; then
        echo -e "${YELLOW}Processing satellite imagery...${NC}"
        python3 "$ARCANUM_DIR/transform_and_upload.py" \
            --input "$satellite_dir" \
            --mode satellite \
            --comfyui-path="$comfyui_path" \
            --gcs-bucket="$bucket" \
            --cdn-url="$cdn_url" \
            $([ "$cleanup" == true ] || echo "--keep-originals")
    fi

    # Process street view imagery if available
    if [ -d "$street_view_dir" ] && [ "$(ls -A $street_view_dir)" ]; then
        echo -e "${YELLOW}Processing street view imagery...${NC}"
        python3 "$ARCANUM_DIR/transform_and_upload.py" \
            --input "$street_view_dir" \
            --mode street_view \
            --comfyui-path="$comfyui_path" \
            --gcs-bucket="$bucket" \
            --cdn-url="$cdn_url" \
            $([ "$cleanup" == true ] || echo "--keep-originals")
    fi

    # Process textures if available
    if [ -d "$textures_dir" ] && [ "$(ls -A $textures_dir)" ]; then
        echo -e "${YELLOW}Processing texture files...${NC}"
        python3 "$ARCANUM_DIR/transform_and_upload.py" \
            --input "$textures_dir" \
            --comfyui-path="$comfyui_path" \
            --gcs-bucket="$bucket" \
            --cdn-url="$cdn_url" \
            $([ "$cleanup" == true ] || echo "--keep-originals")
    fi

    echo -e "${GREEN}Transformation and upload process completed.${NC}"
    echo -e "${GREEN}Content should be available at: ${cdn_url}/tilesets/arcanum/tileset.json${NC}"
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --run)
                RUN_ONLY=true
                shift
                ;;
            --gcs-bucket=*)
                GCS_BUCKET="${1#*=}"
                shift
                ;;
            --cdn-url=*)
                CDN_URL="${1#*=}"
                shift
                ;;
            --keep-originals)
                CLEANUP_ORIGINALS=false
                shift
                ;;
            --skip-upload)
                SKIP_UPLOAD=true
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                exit 1
                ;;
        esac
    done
}

# Main workflow
main() {
    # Parse arguments
    parse_args "$@"
    
    # Install dependencies
    install_dependencies
    
    # Check for GPU
    check_gpu
    HAS_GPU=$?
    
    if [ $HAS_GPU -eq 0 ]; then
        echo -e "${GREEN}GPU detected! Will use GPU-accelerated version.${NC}"
        GPU_AVAILABLE=true
    else
        echo -e "${YELLOW}No GPU detected. Will use CPU-optimized version.${NC}"
        GPU_AVAILABLE=false
    fi
    
    # Ask user for confirmation
    echo ""
    echo -e "${BLUE}Arcanum Setup${NC}"
    echo "This script will:"
    echo "1. Check and install required dependencies"
    echo "2. Install ComfyUI (either GPU or CPU version)"
    echo "3. Set up the required models and configurations"
    echo "4. Run the Arcanum generator"
    if [ "$SKIP_UPLOAD" != true ]; then
        echo "5. Transform and upload content to Google Cloud Storage"
    fi
    echo ""
    
    if [ "$GPU_AVAILABLE" = true ]; then
        echo -e "${GREEN}GPU support is available and will be used.${NC}"
    else
        echo -e "${YELLOW}No GPU detected. The CPU version will be used, which may be slower.${NC}"
    fi
    
    read -p "Do you want to continue? (y/n): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Setup aborted.${NC}"
        exit 1
    fi
    
    # Install appropriate ComfyUI version
    if [ "$GPU_AVAILABLE" = true ]; then
        install_comfyui_gpu
        COMFYUI_PATH="$COMFYUI_DIR"
    else
        # Ask if they want to try GPU version anyway or use CPU version
        echo ""
        echo -e "${YELLOW}Options:${NC}"
        echo "1. Install CPU-optimized version (recommended for systems without GPU)"
        echo "2. Try standard ComfyUI anyway (might be slow without GPU)"
        read -p "Choose an option (1/2): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[1]$ ]]; then
            install_comfyui_cpu
            COMFYUI_PATH="$COMFYUI_CPU_DIR"
        else
            install_comfyui_gpu
            COMFYUI_PATH="$COMFYUI_DIR"
        fi
    fi
    
    # Set up ComfyUI
    setup_comfyui "$COMFYUI_PATH"

    # Check for Hugging Face token
    if [ -z "${HUGGINGFACE_TOKEN}" ]; then
        echo -e "${YELLOW}No Hugging Face token found in environment.${NC}"
        echo -e "${YELLOW}Some models may fail to download without authentication.${NC}"
        echo -e "${YELLOW}You can provide a token now or press Enter to continue without one.${NC}"
        read -p "Enter Hugging Face token (or press Enter to skip): " HF_TOKEN

        if [ ! -z "$HF_TOKEN" ]; then
            export HUGGINGFACE_TOKEN="$HF_TOKEN"
            echo -e "${GREEN}Token set for this session.${NC}"
        else
            echo -e "${YELLOW}Continuing without token. Some downloads may fail.${NC}"
        fi
    else
        echo -e "${GREEN}Using Hugging Face token from environment.${NC}"
    fi

    # Download required Flux models if needed
    echo -e "${BLUE}Checking for required X-Labs Flux models...${NC}"
    python3 "$ARCANUM_DIR/download_flux_models.py" --comfyui-path="$COMFYUI_PATH" --token="$HUGGINGFACE_TOKEN"

    # Configure Google Cloud Storage if uploading is enabled
    if [ "$SKIP_UPLOAD" != true ]; then
        echo -e "${BLUE}Checking Google Cloud Storage configuration...${NC}"
        
        # Ask for bucket name if not provided
        if [ "$GCS_BUCKET" == "arcanum-maps" ]; then
            echo -e "${YELLOW}Default GCS bucket: ${GCS_BUCKET}${NC}"
            read -p "Enter a different bucket name or press Enter to use default: " NEW_BUCKET
            if [ ! -z "$NEW_BUCKET" ]; then
                GCS_BUCKET="$NEW_BUCKET"
            fi
        fi
        
        # Ask for CDN URL if default
        if [ "$CDN_URL" == "https://arcanum.fortunestold.co" ]; then
            echo -e "${YELLOW}Default CDN URL: ${CDN_URL}${NC}"
            read -p "Enter a different CDN URL or press Enter to use default: " NEW_CDN
            if [ ! -z "$NEW_CDN" ]; then
                CDN_URL="$NEW_CDN"
            fi
        fi
        
        echo -e "${GREEN}Using GCS bucket: ${GCS_BUCKET}${NC}"
        echo -e "${GREEN}Using CDN URL: ${CDN_URL}${NC}"
    fi

    # Run generator
    echo ""
    echo -e "${BLUE}Setup completed!${NC}"
    echo -e "${GREEN}Do you want to run the Arcanum generator now?${NC}"
    read -p "Run generator? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Run the generator with the configured ComfyUI path
        run_generator "$COMFYUI_PATH"
        
        # Transform and upload content if enabled
        if [ "$SKIP_UPLOAD" != true ]; then
            echo -e "${GREEN}Generator completed. Do you want to transform and upload to GCS?${NC}"
            read -p "Transform and upload? (y/n): " -n 1 -r
            echo ""
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                transform_and_upload "$COMFYUI_PATH" "$GCS_BUCKET" "$CDN_URL" "$CLEANUP_ORIGINALS"
            else
                echo -e "${YELLOW}Skipping transformation and upload.${NC}"
                echo -e "${YELLOW}You can run it later with:${NC}"
                echo "./start.sh --run --skip-generator"
            fi
        fi
    else
        echo -e "${YELLOW}You can run the generator later with:${NC}"
        echo "./start.sh --run"
    fi
}

# Check if the script is called with --run flag
if [ "$1" = "--run" ]; then
    # Parse all arguments
    parse_args "$@"
    
    # Find the appropriate ComfyUI directory to use
    if [ -d "$COMFYUI_CPU_DIR" ]; then
        COMFYUI_PATH="$COMFYUI_CPU_DIR"
    elif [ -d "$COMFYUI_DIR" ]; then
        COMFYUI_PATH="$COMFYUI_DIR"
    else
        echo -e "${RED}Error: ComfyUI not installed. Please run ./start.sh without arguments first.${NC}"
        exit 1
    fi

    # Check for Hugging Face token
    if [ -z "${HUGGINGFACE_TOKEN}" ]; then
        echo -e "${YELLOW}No Hugging Face token found in environment.${NC}"
        echo -e "${YELLOW}Some models may fail to download without authentication.${NC}"
        echo -e "${YELLOW}You can provide a token now or press Enter to continue without one.${NC}"
        read -p "Enter Hugging Face token (or press Enter to skip): " HF_TOKEN

        if [ ! -z "$HF_TOKEN" ]; then
            export HUGGINGFACE_TOKEN="$HF_TOKEN"
            echo -e "${GREEN}Token set for this session.${NC}"
        else
            echo -e "${YELLOW}Continuing without token. Some downloads may fail.${NC}"
        fi
    else
        echo -e "${GREEN}Using Hugging Face token from environment.${NC}"
    fi

    # Download required Flux models if needed
    echo -e "${BLUE}Checking for required X-Labs Flux models...${NC}"
    python3 "$ARCANUM_DIR/download_flux_models.py" --comfyui-path="$COMFYUI_PATH" --token="$HUGGINGFACE_TOKEN"

    # Run generator
    if [ "$SKIP_GENERATOR" != true ]; then
        run_generator "$COMFYUI_PATH"
    fi
    
    # Transform and upload if not skipped
    if [ "$SKIP_UPLOAD" != true ]; then
        transform_and_upload "$COMFYUI_PATH" "$GCS_BUCKET" "$CDN_URL" "$CLEANUP_ORIGINALS"
    fi
else
    # Run main setup workflow
    main "$@"
fi
