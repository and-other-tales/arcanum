# Arcanum City Generator

A generator script for creating a 3D model of an alternate universe, fantasy/steampunk version of London called 'Arcanum'. This model is suitable for exploring in 1:1 scale in Unity3D for level creation in games.

## Features

- **Non-photorealistic Styling**: Transforms real-life London images into Arcanum-styled versions using X-Labs Flux ControlNet
- **Comprehensive City Coverage**: Processes buildings, streets, landmarks, and terrain
- **1:1 Scale Accuracy**: Maintains the original geographic layout and proportions
- **Unity3D Integration**: Prepares assets for direct import into Unity
- **Geographic Data Collection**: Automated extraction of building footprints, road networks, and terrain data from OpenStreetMap and other open data sources
- **Satellite Imagery Integration**: Processing of satellite imagery for texturing and reference
- **LiDAR Processing**: Creation of accurate Digital Terrain Models from LiDAR point clouds
- **Procedural Building Generation**: Automatic creation of 3D buildings from footprints and height data
- **Landmark Modeling**: Special handling for important Arcanum landmarks
- **Streaming System**: Cell-based streaming for efficient exploration of the large environment

## Overview

Arcanum is an alternative universe version of London featuring a gothic victorian fantasy steampunk aesthetic. This generator creates a 3D model by:

1. Collecting geographic data from various sources (OpenStreetMap, LiDAR, satellite imagery)
2. Applying Arcanum styling to all visual assets using the X-Labs Flux ComfyUI ControlNet
3. Generating 3D models for buildings, landmarks, and terrain
4. Preparing assets for Unity3D import

## Installation

### Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arcanum.git
   cd arcanum
   ```

2. Copy the example environment file and configure it:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. Build and run using Docker Compose:
   ```bash
   docker-compose up -d
   ```

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arcanum.git
   cd arcanum
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up ComfyUI and required models:
   ```bash
   # Clone ComfyUI
   git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI
   
   # Clone X-Labs Flux ComfyUI
   git clone https://github.com/XLabs-AI/x-flux-comfyui.git ~/ComfyUI/custom_nodes/x-flux-comfyui
   
   # Setup X-Labs Flux
   cd ~/ComfyUI/custom_nodes/x-flux-comfyui
   python setup.py
   ```

4. Download required models:
   - [flux1-dev-fp8.safetensors](https://huggingface.co/black-forest-labs/flux/resolve/main/flux1-dev-fp8.safetensors) → `~/ComfyUI/models/`
   - [flux-canny-controlnet.safetensors](https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-canny-controlnet.safetensors) → `~/ComfyUI/models/xlabs/controlnets/`
   - [clip_l.safetensors](https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors) → `~/ComfyUI/models/clip_vision/`
   - [t5xxl_fp16.safetensors](https://huggingface.co/black-forest-labs/flux/resolve/main/t5xxl_fp16.safetensors) → `~/ComfyUI/models/T5Transformer/`
   - [ae.safetensors](https://huggingface.co/black-forest-labs/flux/resolve/main/ae.safetensors) → `~/ComfyUI/models/vae/`

## Usage

### Running the Generator

```bash
# With Docker:
docker-compose run arcanum-generator --output ./output/arcanum_3d_output

# Manual:
python generator.py --output ./arcanum_3d_output
```

### Optional Arguments

- `--output`: Output directory path (default: ./arcanum_3d_output)
- `--bounds`: Area bounds in format "north,south,east,west" (default: "560000,500000,560000,500000")

### Working with the ComfyUI Interface

For development and fine-tuning of the Arcanum style:

```bash
# Start the ComfyUI interface
docker-compose --profile dev up comfyui

# Then access the interface at: http://localhost:8188
```

## Customizing the Arcanum Style

The Arcanum styling uses prompts that can be customized in `.env`:

```
ARCANUM_PROMPT="arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical"
ARCANUM_NEGATIVE_PROMPT="photorealistic, modern, contemporary, bright colors, clear sky"
```

## Unity Import

After running the generator, import the resulting assets into Unity:

1. Create a new Unity project using the HDRP template
2. Import the generated terrain data
3. Import building models and other assets
4. Configure the streaming system using the generated configuration
5. Set up the first-person controller

## Workflow Overview

1. **Data Collection**: Gathering of all necessary geographical data
2. **Styling**: Transformation of photorealistic images to Arcanum style
3. **Terrain Processing**: Creation of accurate terrain model from LiDAR data
4. **Building Generation**: Procedural creation of 3D building models
5. **Texturing**: Application of Arcanum-styled materials and textures
6. **Unity Integration**: Preparation of assets for Unity import
7. **Optimization**: Setting up LOD systems and streaming

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- X-Labs for the Flux ControlNet models
- Black Forest Labs for the FLUX model
- OpenStreetMap contributors for geographic data
- UK Environment Agency for LiDAR data
- Google for satellite imagery and Street View access