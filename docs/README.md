# Arcanum: 3D City Generation Framework

## Overview

Arcanum is a framework for generating stylized 3D cities by combining data from various sources:
- OpenStreetMap for building and road layouts
- Satellite imagery for textures
- ComfyUI with Flux models for style transfer and image generation
- Google 3D Tiles for reference geometry

The system transforms real-world geographic data into a stylized "Arcanum" aesthetic characterized by Victorian-era steampunk and gothic fantasy elements.

## Features

- **Geographic Data Collection**: Downloads and processes OpenStreetMap data, LiDAR point clouds, and satellite imagery
- **Style Transformation**: Uses X-Labs Flux models to transform photorealistic imagery into the Arcanum aesthetic
- **3D Model Generation**: Creates building models, terrain, and street elements from collected data
- **Unity Integration**: Prepares assets for import into Unity3D with streaming and LOD support
- **Google Cloud Integration**: Supports streaming and storage of 3D tiles and assets

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arcanum.git
   cd arcanum
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Download ComfyUI and Flux models:
   ```bash
   python download_flux_models.py
   ```

5. Set up environment variables:
   ```bash
   cp env.example .env
   # Edit .env with your API keys and paths
   ```

## Usage

Arcanum provides several commands for different operations:

### Generate a Complete City

```bash
python arcanum.py generate --output ./my_city --bounds 51.5084,51.5064,-0.1258,-0.1298 --cell-size 100
```

### Download OpenStreetMap Data

```bash
python arcanum.py osm --output ./my_osm_data --bounds 51.5084,51.5064,-0.1258,-0.1298 --cell-size 100
```

### Transform Images with Arcanum Style

```bash
python arcanum.py transform --input ./my_images --output ./styled_images --prompt "arcanum gothic fantasy steampunk"
```

## Architecture

Arcanum is organized into several modules:

### OSM Module

The OSM module handles downloading and processing OpenStreetMap data:

```python
from modules import osm
result = osm.download_osm_data(bounds, output_dir)
```

### ComfyUI Module

The ComfyUI module provides integration with ComfyUI and X-Labs Flux models for image transformation:

```python
from modules import comfyui
transformer = comfyui.ArcanumStyleTransformer()
result_paths = transformer.batch_transform_images(image_paths, output_dir, prompt)
```

### Integration Tools

Integration tools connect Arcanum to external services:

```python
from integration_tools import google_3d_tiles_integration
from integration_tools import storage_integration
from integration_tools import unity_integration
```

## Documentation

For more detailed information, see the following documents:

- [Installation Guide](installation.md) - Detailed setup instructions
- [Technical Documentation](technical_documentation.md) - System architecture and implementation details
- [Google 3D Tiles Integration](google_3d_tiles_integration.md) - Working with Google Maps 3D Tiles
- [Storage Integration](storage_integration.md) - Cloud storage integration
- [Unity Integration](unity_integration.md) - Importing models into Unity3D

## Troubleshooting

### Logging

All logs are stored in the `.arcanum/logs` directory:
- `arcanum.log` - Main application log
- `osm.log` - OpenStreetMap module log
- `comfyui.log` - ComfyUI integration log

### GPU Checks

If you encounter GPU memory errors:
```bash
python gpu_check.py  # Verify GPU availability
```

## Contributing

Contributions to Arcanum are welcome! Please follow the standard GitHub workflow:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

Arcanum is licensed under the MIT License. See LICENSE file for details.