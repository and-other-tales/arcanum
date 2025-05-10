# Arcanum Installation Guide

This guide provides detailed instructions for setting up the Arcanum 3D City Generation Framework, including all dependencies and environment configuration.

## System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS 12+, or Windows 10/11
- **CPU**: 4+ cores, 2.5GHz+ (8+ cores recommended for large areas)
- **RAM**: 16GB minimum, 32GB+ recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM for model inference (16GB+ recommended)
- **Storage**: 50GB+ free space (100GB+ recommended for large projects)
- **Internet**: High-speed connection for data downloads

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/arcanum.git
cd arcanum
```

## Step 2: Set Up Python Environment

### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
# venv\Scripts\activate
```

### Option B: Using conda

```bash
# Create conda environment
conda create -n arcanum python=3.10
conda activate arcanum
```

## Step 3: Install Dependencies

```bash
# Install basic requirements
pip install -r requirements.txt
```

### GDAL Installation

GDAL is required for geospatial data processing. Installation depends on your operating system:

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y libgdal-dev gdal-bin
pip install pygdal==$(gdal-config --version).*
```

#### macOS

```bash
brew install gdal
pip install gdal==$(gdal-config --version)
```

#### Windows

Install OSGeo4W from https://trac.osgeo.org/osgeo4w/, then:

```bash
# Make sure to use the same version as your installed GDAL
pip install GDAL==X.X.X
```

## Step 4: ComfyUI and Model Setup

### ComfyUI Installation

```bash
# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install ComfyUI dependencies
pip install -r requirements.txt

# Return to arcanum directory
cd ..
```

### Download Flux Models

```bash
# Download required models for style transfer
python download_flux_models.py --comfyui-path ./ComfyUI
```

## Step 5: Google Cloud Setup (Optional)

For Google 3D Tiles integration and cloud storage:

1. Create a Google Cloud account if you don't have one
2. Create a new project in Google Cloud Console
3. Enable the following APIs:
   - Google Maps API
   - Earth Engine API
   - Google Cloud Storage API
4. Create a service account with appropriate permissions
5. Download the service account key as `key.json` and place it in the root directory

## Step 6: Environment Configuration

Create and configure environment variables:

```bash
# Copy example environment file
cp env.example .env

# Edit .env file with your settings
nano .env
```

Required environment variables:

```
# API Keys (if using cloud features)
GOOGLE_APPLICATION_CREDENTIALS=./key.json
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
GOOGLE_EARTH_API_KEY=your_earth_engine_api_key
HUGGINGFACE_TOKEN=your_huggingface_token

# Paths
COMFYUI_PATH=./ComfyUI

# System Configuration
ARCANUM_OUTPUT_DIR=./arcanum_output
CUDA_VISIBLE_DEVICES=0  # GPU device ID to use
```

## Step 7: Verify Installation

Test if your installation is working:

```bash
# Verify GPU setup
python gpu_check.py

# Test Earth Engine authentication (if using)
python authenticate_ee.py
```

## Step 8: Start Arcanum

Now you can use Arcanum to generate 3D cities:

```bash
python arcanum.py --help
```

## Docker Setup (Alternative)

For a containerized setup:

```bash
# Build the Arcanum Docker image
docker build -t arcanum:latest .

# Run with appropriate volume mounts for output
docker run -v $(pwd)/output:/app/output -v $(pwd)/key.json:/app/key.json arcanum:latest
```

## Troubleshooting

### Common Issues

1. **GDAL Installation Errors**:
   - Ensure system libraries are installed before Python packages
   - Make sure versions match between system GDAL and pygdal

2. **GPU Not Detected**:
   - Check CUDA installation with `nvidia-smi`
   - Ensure PyTorch was installed with CUDA support

3. **Memory Errors During Processing**:
   - Reduce batch size in configuration
   - Process smaller geographic areas
   - Increase swap space if necessary

4. **Model Download Failures**:
   - Check your HuggingFace token
   - Accept model licenses on the HuggingFace website
   - Try with `--force-redownload` flag

### Support

If you encounter issues not covered here:

1. Check the logs in `.arcanum/logs/`
2. Open an issue on the GitHub repository
3. Include your system information and relevant log files

## Next Steps

- See [README.md](README.md) for basic usage instructions
- See [technical_documentation.md](technical_documentation.md) for system architecture details
- See [unity_integration.md](unity_integration.md) for importing generated models into Unity