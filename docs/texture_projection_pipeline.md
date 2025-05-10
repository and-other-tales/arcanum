# Texture Projection Pipeline

## Overview

The Texture Projection Pipeline is a key component of the Arcanum system that handles the process of applying textures to 3D building models. It provides a comprehensive workflow for:

1. Acquiring textures from various sources (ComfyUI generation, Street View imagery, user-provided textures)
2. Managing texture atlases to optimize rendering performance
3. Mapping textures to building geometries with correct UV coordinates
4. Integrating with Unity and other 3D environments

This document describes the architecture, components, and usage of the texture projection pipeline.

## Architecture

The texture projection pipeline consists of several integrated modules:

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  OSM Building   │◄────►│  Texture Atlas  │◄────►│  Visualization  │
│   Processor     │      │    Manager      │      │     Module      │
│                 │      │                 │      │                 │
└────────┬────────┘      └────────┬────────┘      └─────────────────┘
         │                        │                       ▲
         │                        │                       │
         │                        ▼                       │
         │               ┌─────────────────┐      ┌───────┴───────┐
         │               │                 │      │               │
         └──────────────►│ Texture Pipeline├─────►│ Web Interface │
                         │   Integration   │      │               │
                         │                 │      └───────────────┘
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │                 │
                         │ Unity Material  │
                         │    Pipeline     │
                         │                 │
                         └─────────────────┘
```

### Key Components

1. **OSM Building Processor**: Processes building data from OpenStreetMap, creating building geometries with appropriate metadata.

2. **Texture Atlas Manager**: The core component that manages texture atlases, UV mapping, and texture assignments.

3. **Visualization Module**: Provides preview visualization for textures, atlases, and building models.

4. **Texture Pipeline Integration**: Connects all components into a coherent workflow.

5. **Unity Material Pipeline**: Creates Unity materials and prefabs based on the texture assignments.

6. **Web Interface**: Provides a user-friendly way to manage textures and building assignments.

## Texture Atlas Manager

The Texture Atlas Manager is responsible for efficiently packing multiple building textures into optimized texture atlases.

### Key Features

- **Texture Assignment**: Maps specific textures to buildings
- **Atlas Generation**: Packs multiple textures into optimized atlas textures
- **UV Mapping**: Calculates UV coordinates for building models
- **Texture Caching**: Maintains a cache of processed textures for performance

### Usage

```python
from modules.texture_atlas_manager import TextureAtlasManager

# Create a texture atlas manager
atlas_manager = TextureAtlasManager(
    output_dir="/path/to/atlases",
    atlas_size=(4096, 4096),
    building_texture_size=(1024, 1024)
)

# Assign a texture to a building
result = atlas_manager.assign_texture_to_building(
    building_id="building_123",
    texture_path="/path/to/texture.jpg",
    building_metadata={}  # Optional metadata
)

# Generate atlases
atlas_result = atlas_manager.generate_atlases()
```

## Street View Texture Projection

The system can project textures from Google Street View images onto building models:

### Process

1. **Image Acquisition**: Download Street View images for the target area
2. **Camera Position Estimation**: Compute camera positions relative to buildings
3. **Ray Casting**: Cast rays from camera through image pixels to building surfaces
4. **Texture Projection**: Project image pixels onto building faces
5. **Texture Blending**: Blend multiple images for complete coverage
6. **Atlas Integration**: Integrate projected textures into the atlas system

### Usage

```python
from integration_tools import texture_projection
from modules.integration.texture_pipeline import TexturePipeline

# Create a texture pipeline
pipeline = TexturePipeline(output_dir="/path/to/output")

# Project Street View textures
result = pipeline.project_street_view_textures(
    street_view_dir="/path/to/street_view_images"
)
```

## Texture Pipeline Integration

The Texture Pipeline Integration module combines all components into a unified workflow:

### Features

- **Building Processing**: Process building geometries from OSM data
- **Texture Assignment**: Assign textures to buildings
- **Atlas Generation**: Generate optimized texture atlases
- **Visualization**: Create previews and visualizations
- **Unity Integration**: Export materials and prefabs for Unity

### Usage

```python
from modules.integration.texture_pipeline import TexturePipeline

# Create a texture pipeline
pipeline = TexturePipeline(
    output_dir="/path/to/output",
    texture_dir="/path/to/textures"
)

# Process buildings with textures
result = pipeline.process_buildings("/path/to/buildings.json")

# Assign a texture to a specific building
assign_result = pipeline.assign_texture_to_building(
    building_id="building_123",
    texture_path="/path/to/texture.jpg"
)

# Project Street View textures
sv_result = pipeline.project_street_view_textures(
    street_view_dir="/path/to/street_view"
)
```

## Web Interface Integration

The web interface provides a user-friendly way to manage textures and building assignments:

### Features

- **Building Management**: View and filter buildings
- **Texture Assignment**: Assign textures to buildings
- **Atlas Visualization**: View texture atlases with building regions
- **Texture Previews**: View texture previews and mappings
- **Building Visualization**: View 3D building previews with textures

### API Endpoints

- `GET /api/buildings`: Get list of buildings
- `GET /api/textures`: Get list of available textures
- `POST /api/textures/assign`: Assign a texture to a building
- `GET /api/atlases`: Get list of texture atlases
- `GET /api/preview/atlas`: Generate atlas preview
- `GET /api/preview/building`: Generate building preview

## Unity Material Pipeline

The Unity Material Pipeline exports building models and textures for use in Unity:

### Features

- **Material Creation**: Create Unity materials for buildings
- **Prefab Generation**: Generate Unity prefabs for buildings
- **Texture Atlas Support**: Support for texture atlases
- **LOD Support**: Support for multiple levels of detail
- **Material Properties**: Configure material properties (roughness, metallic, etc.)

### Usage

```python
from integration_tools import unity_material_pipeline

# Create a material pipeline
material_pipeline = unity_material_pipeline.UnityMaterialPipeline(
    materials_dir="/path/to/materials",
    textures_dir="/path/to/textures",
    unity_dir="/path/to/unity"
)

# Create a material for a building
result = material_pipeline.create_material_for_building(
    building_id="building_123",
    texture_path="/path/to/texture.jpg",
    material_type="standard"
)
```

## Best Practices

### Texture Management

1. **Texture Size**: Use texture sizes that are powers of 2 (1024x1024, 2048x2048, etc.)
2. **Atlas Size**: Use atlas sizes that are powers of 2 (4096x4096, 8192x8192, etc.)
3. **Texture Format**: Use compressed texture formats (JPG, DXT) for better performance
4. **Texture Atlasing**: Use texture atlases to reduce draw calls and improve performance

### Building Processing

1. **Building Simplification**: Simplify building geometries to reduce polygon count
2. **UV Mapping**: Ensure proper UV mapping for accurate texture projection
3. **LOD Levels**: Use multiple levels of detail for better performance
4. **Batching**: Batch buildings in the same atlas for better rendering performance

### Street View Projection

1. **Image Quality**: Use high-quality Street View images for better results
2. **Coverage**: Ensure adequate coverage of building facades
3. **Perspective Correction**: Apply perspective correction for accurate projection
4. **Occlusion Handling**: Handle occlusions to avoid projecting irrelevant elements

## Troubleshooting

### Common Issues

1. **Missing Textures**: Ensure texture paths are correct and files exist
2. **UV Mapping Issues**: Check UV coordinates in building models
3. **Atlas Generation Failures**: Check atlas size and texture count
4. **Street View Projection Issues**: Check camera positions and building geometries

### Debugging

1. **Visualizations**: Use the visualization module to debug texture and atlas issues
2. **Logging**: Enable verbose logging for detailed information
3. **Atlas Inspection**: Inspect atlas images for issues
4. **UV Mapping Inspection**: Check UV mapping values for correctness

## Command Line Tools

### Texture Pipeline Tool

```bash
python -m modules.integration.texture_pipeline --buildings /path/to/buildings.json --output /path/to/output --textures /path/to/textures
```

### Street View Projection Tool

```bash
python -m modules.integration.texture_pipeline --buildings /path/to/buildings.json --street-view /path/to/street_view --output /path/to/output
```

### Texture Assignment Tool

```bash
python -m modules.integration.texture_pipeline --assign building_123:/path/to/texture.jpg --output /path/to/output
```

## Future Enhancements

1. **AI-Based Texture Generation**: Use AI to generate building textures based on style and context
2. **Improved Texture Projection**: Enhance Street View projection accuracy and quality
3. **Advanced Material Properties**: Support for PBR materials with roughness, metallic, etc.
4. **Real-Time Editing**: Real-time texture and material editing in the web interface
5. **Style Transfer**: Apply style transfer techniques to textures for consistent appearance