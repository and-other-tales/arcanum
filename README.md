# Arcanum City Generator

Generate a stylized, non-photorealistic 1:1 scale model of Arcanum (an alternative London) for exploration in Unity3D.

## Overview

Arcanum City Generator combines OpenStreetMap data, image stylization using X-Labs FLUX.1 AI models, and procedural generation to create a cohesive, atmospheric urban environment. It transforms real-world geographic data into a stylized, gothic-steampunk aesthetic.

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

## Code Structure

The codebase has been reorganized into a modular structure for better maintainability:

```
arcanum/
├── arcanum.py             # New main entry point
├── modules/               # Modular components
│   ├── osm/               # OpenStreetMap functionality
│   │   ├── __init__.py    # Module exports
│   │   ├── config.py      # OSMnx configuration
│   │   ├── bbox_downloader.py  # Direct area download
│   │   └── grid_downloader.py  # Grid-based download
│   ├── comfyui/           # ComfyUI integration
│   │   ├── __init__.py    # Module exports
│   │   ├── automation.py  # ComfyUI automation
│   │   ├── transformer.py # Style transformation
│   │   └── workflows/     # ComfyUI workflow definitions
│   └── ...                # Other modules
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arcanum.git
   cd arcanum
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up ComfyUI (if using style transformation):
   ```bash
   # Clone ComfyUI repository
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   
   # Install ComfyUI dependencies
   pip install -r requirements.txt
   
   # Return to Arcanum directory
   cd ..
   
   # Set environment variable
   export COMFYUI_PATH=/path/to/ComfyUI
   ```

## Usage

### Basic Usage

```bash
# Run the full generation workflow
python arcanum.py generate --bounds 51.5084,51.5064,-0.1258,-0.1298 --output ./arcanum_output

# Download OSM data only
python arcanum.py osm --bounds 51.5084,51.5064,-0.1258,-0.1298 --cell-size 100

# Transform images to Arcanum style
python arcanum.py transform --input ./input_images --output ./arcanum_output/final
```

### Command Line Arguments

#### Generate Mode
- `--output`: Output directory (default: ./arcanum_3d_output)
- `--bounds`: Area bounds in WGS84 coordinates as north,south,east,west
- `--cell-size`: Grid cell size in meters (default: 100)

#### OSM Mode
- `--output`: Output directory (default: ./arcanum_3d_output)
- `--bounds`: Area bounds in WGS84 coordinates as north,south,east,west
- `--cell-size`: Grid cell size in meters (0 for direct download)

#### Transform Mode
- `--input`: Input directory containing images
- `--output`: Output directory for transformed images
- `--prompt`: Text prompt for image generation
- `--no-controlnet`: Disable ControlNet for faster processing

## OpenStreetMap Download Options

The project offers two methods for downloading OpenStreetMap data:

1. **Direct Download**: Best for small areas, uses OSMnx directly to download the entire area at once.
2. **Grid-Based Download**: Best for large areas, divides the region into smaller grid cells and downloads each individually, then merges the results.

The grid-based approach helps avoid Overpass API limitations and timeouts when working with large regions.

## Image Stylization

The style transformation uses X-Labs FLUX.1-dev with ComfyUI to transform real-world images into Arcanum's gothic-steampunk style. Two modes are available:

1. **Standard Mode**: Basic img2img style transfer.
2. **Enhanced Mode**: Uses ControlNet-Canny for better structure preservation and adds upscaling.

## Environment Variables

- `COMFYUI_PATH`: Path to ComfyUI installation
- `HUGGINGFACE_TOKEN`: HuggingFace token for accessing gated models

## Notes on Deprecated Files

Several files in the root directory are deprecated and maintained only for backward compatibility:

- `fixed_generator.py`: Use `arcanum.py generate` instead
- `fixed_graph_from_bbox.py`: Integrated into `modules/osm`
- `fixed_graph_from_bbox_large.py`: Integrated into `modules/osm`
- `osmnx_config.py`: Moved to `modules/osm/config.py`
- `osm_grid_downloader.py`: Moved to `modules/osm/grid_downloader.py`
- `comfyui_automation.py`: Moved to `modules/comfyui/automation.py`
- `comfyui_automation_enhanced.py`: Functionality integrated into `modules/comfyui`

## Troubleshooting

### OpenStreetMap Download Issues

If you encounter issues with OSM downloads:

1. Try using a smaller area or reducing the `--cell-size` value
2. Ensure you're using the correct Overpass API endpoint (https://overpass-api.de/api/interpreter)
3. Check network connectivity and firewall settings

### ComfyUI Integration Issues

1. Verify ComfyUI is installed correctly and the path is set
2. Check that you have the required models installed
3. For X-Labs FLUX models, ensure you have a valid HuggingFace token

## Customizing the Arcanum Style

The Arcanum styling uses prompts that can be customized:

```
ARCANUM_PROMPT="arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical"
ARCANUM_NEGATIVE_PROMPT="photorealistic, modern, contemporary, bright colors, clear sky"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- X-Labs for the Flux ControlNet models
- Black Forest Labs for the FLUX model
- OpenStreetMap contributors for geographic data
- UK Environment Agency for LiDAR data
- Google for satellite imagery and Street View access