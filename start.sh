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
    
    # Install PyTorch first (this is a large dependency, so we handle it separately)
    if ! package_installed torch; then
        echo -e "${YELLOW}Installing PyTorch...${NC}"
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other required packages from requirements_no_gdal.txt
    echo -e "${YELLOW}Installing dependencies from requirements_no_gdal.txt...${NC}"
    pip3 install -r "$ARCANUM_DIR/requirements_no_gdal.txt"
    
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
    
    # Create required directories
    mkdir -p "$comfyui_path/models/x-labs/Flux"
    
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

    # Set environment variable for generator.py to use the right ComfyUI path
    export COMFYUI_PATH="$comfyui_path"

    # Run generator.py - fixed to use the comfyui path with a proper flag
    python3 "$ARCANUM_DIR/generator.py" --comfyui-path="$comfyui_path"
}

# Main workflow
main() {
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
    
    # Run generator
    echo ""
    echo -e "${BLUE}Setup completed!${NC}"
    echo -e "${GREEN}Do you want to run the Arcanum generator now?${NC}"
    read -p "Run generator? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Run the generator with the configured ComfyUI path
        run_generator "$COMFYUI_PATH"
    else
        echo -e "${YELLOW}You can run the generator later with:${NC}"
        echo "./start.sh --run"
    fi
}

# Check if the script is called with --run flag
if [ "$1" = "--run" ]; then
    # Find the appropriate ComfyUI directory to use
    if [ -d "$COMFYUI_CPU_DIR" ]; then
        COMFYUI_PATH="$COMFYUI_CPU_DIR"
    elif [ -d "$COMFYUI_DIR" ]; then
        COMFYUI_PATH="$COMFYUI_DIR"
    else
        echo -e "${RED}Error: ComfyUI not installed. Please run ./start.sh without arguments first.${NC}"
        exit 1
    fi
    
    # Run generator with arguments except --run
    shift  # Remove --run from arguments
    run_generator "$COMFYUI_PATH"
else
    # Run main setup workflow
    main "$@"
fi