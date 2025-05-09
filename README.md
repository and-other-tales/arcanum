# Arcanum 3D City Generator

A comprehensive system for generating a photorealistic 1:1 scale model of Arcanum for exploration in Unity3D.

## Overview

This project orchestrates the generation of a highly detailed 3D model of Arcanum by automating the collection and processing of geographical data, generating 3D models, and integrating everything into a Unity3D environment for exploration.

## Features

- **Geographic Data Collection**: Automated extraction of building footprints, road networks, and terrain data from OpenStreetMap and other open data sources
- **Satellite Imagery Integration**: Processing of satellite imagery for texturing and reference
- **LiDAR Processing**: Creation of accurate Digital Terrain Models from LiDAR point clouds
- **Procedural Building Generation**: Automatic creation of 3D buildings from footprints and height data
- **Landmark Modeling**: Special handling for important Arcanum landmarks
- **Texturing System**: Creation of appropriate facade textures based on building type and era
- **Unity3D Integration**: Full workflow for importing into Unity with proper LOD setup
- **Streaming System**: Cell-based streaming for efficient exploration of the large environment

## Requirements

- Python 3.8+
- Required Python packages (see `requirements.txt`)
- Google Cloud API credentials (for satellite imagery and Street View)
- Unity3D 2021.3 or later with HDRP

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arcanum-3d-generator.git
cd arcanum-3d-generator

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for API keys
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

## Usage

```bash
# Run the complete workflow
python generator.py --output ./arcanum_output

# Specify custom bounds (in British National Grid coordinates)
python generator.py --bounds 560000,500000,560000,500000
```

## Workflow Overview

1. **Data Collection**: Gathering of all necessary geographical data
2. **Terrain Processing**: Creation of accurate terrain model from LiDAR data
3. **Building Generation**: Procedural creation of 3D building models
4. **Texturing**: Application of appropriate materials and textures
5. **Unity Integration**: Preparation of assets for Unity import
6. **Optimization**: Setting up LOD systems and streaming

## Unity Import

After running the generator, import the resulting assets into Unity:

1. Create a new Unity project using the HDRP template
2. Import the generated terrain data
3. Import building models and other assets
4. Configure the streaming system using the generated configuration
5. Set up the first-person controller

## Customization

The generator can be customized by modifying the `PROJECT_CONFIG` dictionary in the main script. Key parameters include:

- Coordinate system and bounds
- Cell size for streaming
- LOD distance thresholds
- Output directory structure

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenStreetMap contributors for geographical data
- UK Environment Agency for LiDAR data
- Google for satellite imagery and Street View access