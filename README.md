# 🏙️ Arcanum: 3D City Generation Framework

## Overview

Arcanum is a comprehensive framework for generating stylized 3D cities by combining data from various sources:
- 🗺️ OpenStreetMap for building and road layouts
- 📸 Google Street View for street-level perspectives
- 🌐 Google 3D Tiles for reference geometry
- 🖼️ ComfyUI with Flux models for style transfer and image generation

The system transforms real-world geographic data into a stylized "Arcanum" aesthetic characterized by Victorian-era steampunk and gothic fantasy elements.

## ✨ Features

- **Complete Geographic Coverage**: Enhanced systems ensure comprehensive coverage of target areas
  - City-scale 3D tiles download with automatic grid subdivision
  - Road network-based Street View collection following actual streets
  - Coverage verification tools to identify and address gaps

- **Geographic Data Collection**: Downloads and processes OpenStreetMap data, Google 3D Tiles, and Street View imagery
  - Intelligent nearest imagery finder for robust Street View coverage
  - Grid-based downloads for handling large geographic areas
  - Spatial bounds utilities for precise geographic targeting

- **Style Transformation**: Uses X-Labs Flux models to transform photorealistic imagery into the Arcanum aesthetic
  - Structure-preserving style transfer with ControlNet
  - Consistent aesthetic application across diverse urban elements
  - Batch processing for efficiency

- **3D Model Generation**: Creates building models, terrain, and street elements from collected data
  - Footprint extrusion with appropriate height information
  - Roof type detection and generation
  - LOD generation for performance optimization

- **Unity Integration**: Prepares assets for import into Unity3D with streaming and LOD support
  - Material and texture setup for HDRP
  - Prefab generation for entire city blocks
  - Asset organization for large-scale scenes

## 🔧 Installation

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

## 🚀 Usage

Arcanum provides several commands for different operations:

### Generate a Complete City

```bash
python arcanum.py generate --city "London" --output ./london_city --style "arcanum_gothic"
```

### Download OpenStreetMap Data

```bash
python arcanum.py osm --output ./my_osm_data --bounds 51.5084,51.5064,-0.1258,-0.1298 --cell-size 100
```

### Download 3D Tiles for a City

```bash
python fetch_city_3d_tiles.py --city "London" --output ./london_3d_tiles --depth 4
```

### Collect Street View Along Roads

```bash
python fetch_street_view_along_roads.py --osm ./data/london.gpkg --output ./london_street_view --interval 50
```

### Verify Coverage

```bash
python verify_coverage.py --mode both --city "London" --osm ./data/london.gpkg --tiles-dir ./london_3d_tiles --street-view-dir ./london_street_view
```

### Transform Images with Arcanum Style

```bash
python arcanum.py transform --input ./my_images --output ./styled_images --prompt "arcanum gothic fantasy steampunk"
```

## 🏗️ Architecture

Arcanum is organized into several modules:

### Data Collection Modules

- **OSM Module**: Handles downloading and processing OpenStreetMap data
  ```python
  from modules import osm
  result = osm.download_osm_data(bounds, output_dir)
  ```

- **Google 3D Tiles**: Manages photorealistic 3D city model download
  ```python
  from integration_tools import google_3d_tiles_integration
  result = google_3d_tiles_integration.fetch_city_tiles("London", output_dir)
  ```

- **Street View Integration**: Collects street-level imagery along road networks
  ```python
  from integration_tools import street_view_integration
  from integration_tools import road_network
  result = road_network.fetch_street_view_along_roads(osm_path, output_dir)
  ```

### Styling and Processing

- **ComfyUI Module**: Integrates with ComfyUI and X-Labs Flux models for image transformation
  ```python
  from modules import comfyui
  transformer = comfyui.ArcanumStyleTransformer()
  result_paths = transformer.batch_transform_images(image_paths, output_dir, prompt)
  ```

### Output and Verification

- **Coverage Verification**: Ensures comprehensive coverage of target areas
  ```python
  from integration_tools import coverage_verification
  verifier = coverage_verification.CoverageVerifier(bounds=city_bounds)
  result = verifier.verify_3d_tiles_coverage(tiles_dir)
  ```

- **Unity Integration**: Prepares assets for Unity3D import
  ```python
  from integration_tools import unity_integration
  unity_integration.prepare_unity_assets(models_dir, textures_dir, output_dir)
  ```

## 📚 Documentation

For more detailed information, see the following documents:

- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [Technical Documentation](docs/technical_documentation.md) - System architecture and implementation details
- [Google 3D Tiles Integration](docs/google_3d_tiles_integration.md) - Working with Google Maps 3D Tiles
- [Street View Integration](docs/street_view_integration.md) - Road-based Street View collection
- [Coverage Verification](docs/coverage_verification.md) - Ensuring comprehensive coverage
- [Storage Integration](docs/storage_integration.md) - Cloud storage integration
- [Unity Integration](docs/unity_integration.md) - Importing models into Unity3D
- [API Keys Configuration](docs/API_KEYS.md) - Setting up required API keys

## 🔍 Troubleshooting

### Logging

All logs are stored in the `.arcanum/logs` directory:
- `arcanum.log` - Main application log
- `osm.log` - OpenStreetMap module log
- `comfyui.log` - ComfyUI integration log
- `3d_tiles.log` - Google 3D Tiles download log
- `street_view.log` - Street View collection log

### GPU Checks

If you encounter GPU memory errors:
```bash
python gpu_check.py  # Verify GPU availability
```

### Common Issues

- **API Key Errors**: Ensure your Google Maps API keys are correctly set in environment variables
- **Memory Errors**: Reduce batch sizes or processing area size
- **Coverage Gaps**: Use the coverage verification tools to identify and address specific areas

## 🤝 Contributing

Contributions to Arcanum are welcome! Please follow the standard GitHub workflow:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

Arcanum is licensed under the MIT License. See LICENSE file for details.