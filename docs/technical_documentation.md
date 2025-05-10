# Arcanum Technical Documentation

This document provides in-depth technical details about the Arcanum City Generation Framework, including its architecture, components, and implementation details.

## System Architecture

Arcanum implements a modular architecture organized into five primary subsystems:

1. **Data Collection Subsystem**: Acquires geographic data from OpenStreetMap, LiDAR sources, satellite imagery, and street-level photography
2. **Arcanum Styling Subsystem**: Transforms visual assets using X-Labs Flux ControlNet to apply the Arcanum aesthetic
3. **Terrain Processing Subsystem**: Generates digital terrain models from geographic data
4. **Building Generation Subsystem**: Creates 3D building models using footprint and height data
5. **Unity Integration Subsystem**: Prepares assets for Unity3D import

### Component Diagram

```
┌─────────────────────────────┐
│      Data Collection        │
│                             │
│  ┌─────────┐  ┌──────────┐  │
│  │OpenStreet│  │Satellite │  │
│  │  Map     │  │ Imagery  │  │
│  └─────────┘  └──────────┘  │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│     Arcanum Styling         │
│                             │
│  ┌─────────┐  ┌──────────┐  │
│  │ ComfyUI │  │   Flux   │  │
│  │ Client  │  │ControlNet│  │
│  └─────────┘  └──────────┘  │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│     Asset Generation        │
│                             │
│  ┌─────────┐  ┌──────────┐  │
│  │ Terrain │  │ Building │  │
│  │ Models  │  │  Models  │  │
│  └─────────┘  └──────────┘  │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│     Unity Integration       │
│                             │
│  ┌─────────┐  ┌──────────┐  │
│  │Streaming│  │  Asset   │  │
│  │  Setup  │  │ Creation │  │
│  └─────────┘  └──────────┘  │
└─────────────────────────────┘
```

## Module Structure

The codebase is organized into the following modules:

```
arcanum/
├── modules/
│   ├── comfyui/        # ComfyUI integration and style transfer
│   ├── osm/            # OpenStreetMap data processing
│   ├── geo/            # Geospatial utilities (WIP)
│   ├── integration/    # External services integration (WIP)
│   └── storage/        # Storage management (WIP)
├── integration_tools/
│   ├── comfyui_integration.py        # ComfyUI interface
│   ├── google_3d_tiles_integration.py # Google Maps integration
│   ├── storage_integration.py        # Cloud storage utilities
│   └── unity_integration.py          # Unity export tools
├── arcanum.py          # Main entry point
└── various utility scripts
```

## Data Flow

### 1. Data Collection

The data collection process acquires geographic information from multiple sources:

```python
def download_osm_data(bounds, output_dir, grid_size=None, coordinate_system="EPSG:4326"):
    # Determine which approach to use based on grid_size
    if grid_size is not None and grid_size > 0:
        # Use grid-based approach for larger areas
        result = osm.download_osm_grid(
            bounds=bounds,
            output_dir=output_dir,
            cell_size_meters=grid_size
        )
        
        # Merge grid data
        if result.get("success_count", 0) > 0:
            merge_result = osm.merge_grid_data(output_dir)
    else:
        # Use direct approach for smaller areas
        return osm.download_osm_data(
            bounds=bounds,
            output_dir=output_dir,
            coordinate_system=coordinate_system
        )
```

OpenStreetMap data is downloaded using either direct API queries for small areas or grid-based downloads for larger regions. The data includes:

- Building footprints with metadata
- Road networks
- Points of interest
- Water features
- Land use classifications

### 2. Arcanum Styling

The styling subsystem uses X-Labs Flux ControlNet models to transform imagery:

```python
def transform_images(
    input_dir,
    output_dir,
    prompt="arcanum gothic victorian fantasy steampunk architecture",
    negative_prompt="photorealistic, modern, contemporary",
    pattern="*.jpg",
    use_controlnet=True
):
    # Find all matching images
    image_paths = []
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    # Transform images
    result_paths = comfyui.batch_transform_images(
        image_paths=image_paths,
        output_dir=output_dir,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_controlnet=use_controlnet
    )
```

The transformation process:

1. Uses Canny edge detection to extract structural information
2. Applies the Flux diffusion model guided by the ControlNet
3. Preserves architectural details while transforming the style
4. Maintains geographic features with appropriate strength parameters

The prompts are carefully designed to guide the style transformation:
- Core aesthetic: "gothic victorian fantasy steampunk"
- Geographic identity: "alternative London"
- Atmospheric qualities: "dark atmosphere", "foggy", "mystical"
- Architectural treatment: "ornate details", "imposing structure"

### 3. 3D Model Generation

Building models are generated from OSM footprints:

```python
def generate_building_models(osm_data, output_dir, terrain_height=None):
    buildings = geopandas.read_file(osm_data, layer='buildings')
    
    for idx, building in buildings.iterrows():
        # Extract height information
        height = building.get('height', 10.0)  # Default to 10m if not specified
        
        # Generate basic geometry
        footprint = building.geometry
        building_id = building.get('id', f"building_{idx}")
        
        # Create 3D model with appropriate LODs
        generate_building_model(
            footprint, 
            height, 
            building_id,
            output_dir
        )
```

The model generation includes:
- Footprint extrusion to building height
- Roof generation based on building type
- LOD levels for distance-appropriate detail
- UV mapping for texture application
- Facade segmentation for detailed texturing

### 4. Unity Integration

Assets are prepared for Unity import and runtime streaming:

```python
def prepare_unity_assets(models_dir, textures_dir, output_dir):
    # Setup directory structure for Unity
    unity_dirs = [
        "Assets/Arcanum/Buildings",
        "Assets/Arcanum/Materials",
        "Assets/Arcanum/Prefabs",
        "Assets/Arcanum/Terrain",
        "Assets/Arcanum/Textures"
    ]
    
    for d in unity_dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
    
    # Configure material mappings
    material_map = create_material_mapping(textures_dir, output_dir)
    
    # Process models for Unity
    for model in os.listdir(models_dir):
        if model.endswith('.obj') or model.endswith('.fbx'):
            # Copy and adjust model for Unity
            prepare_model_for_unity(
                os.path.join(models_dir, model),
                output_dir,
                material_map
            )
```

The Unity preparation involves:
- Asset format conversion for Unity compatibility
- Material and shader setup for HDRP
- Streaming configuration for large environments
- LOD setup for performance optimization
- Collider generation for physical interaction

## Key Algorithms

### Grid-Based OSM Download

For large areas, the system uses a grid-based download approach to avoid API limitations:

```python
def download_osm_grid(bounds, output_dir, cell_size_meters=1000):
    # Convert boundary coordinates to a grid
    north, south, east, west = bounds['north'], bounds['south'], bounds['east'], bounds['west']
    
    # Calculate grid cells
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857")
    
    # Convert to web mercator for distance calculations
    west_m, south_m = transformer.transform(west, south)
    east_m, north_m = transformer.transform(east, north)
    
    # Calculate number of cells in each dimension
    width_m = east_m - west_m
    height_m = north_m - south_m
    cols = math.ceil(width_m / cell_size_meters)
    rows = math.ceil(height_m / cell_size_meters)
    
    # Download data for each cell
    for row in range(rows):
        for col in range(cols):
            # Calculate cell boundaries
            cell_west_m = west_m + col * cell_size_meters
            cell_east_m = min(cell_west_m + cell_size_meters, east_m)
            cell_north_m = north_m - row * cell_size_meters
            cell_south_m = max(cell_north_m - cell_size_meters, south_m)
            
            # Convert back to WGS84
            cell_west, cell_south = transformer.transform(cell_west_m, cell_south_m, direction="INVERSE")
            cell_east, cell_north = transformer.transform(cell_east_m, cell_north_m, direction="INVERSE")
            
            # Download OSM data for this cell
            download_cell(cell_north, cell_south, cell_east, cell_west, output_dir, row, col)
```

### Canny-Based Style Transfer

The style transfer process uses Canny edge detection for ControlNet guidance:

```python
def prepare_canny_control(image_path, low_threshold=100, high_threshold=200):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Create control image with white edges on black background
    control_image = np.zeros_like(image)
    control_image[edges > 0] = [255, 255, 255]
    
    return control_image
```

### Building Extrusion Algorithm

3D building generation uses extrusion with roof type determination:

```python
def extrude_building(footprint, height, roof_type="flat"):
    # Convert footprint to 3D coordinates
    vertices_2d = footprint.exterior.coords[:-1]  # Exclude last point (same as first)
    bottom_vertices = [(x, y, 0) for x, y in vertices_2d]
    top_vertices = [(x, y, height) for x, y in vertices_2d]
    
    # Create walls
    walls = []
    for i in range(len(bottom_vertices)):
        next_i = (i + 1) % len(bottom_vertices)
        quad = [
            bottom_vertices[i],
            bottom_vertices[next_i],
            top_vertices[next_i],
            top_vertices[i]
        ]
        walls.append(quad)
    
    # Create roof based on type
    roof = []
    if roof_type == "flat":
        roof = [top_vertices]
    elif roof_type == "pitched":
        # Create pitched roof geometry
        roof_height = height + 5  # 5m above top of walls
        ridge_points = calculate_ridge_line(top_vertices)
        # Add roof faces
        # ...
    
    return {
        "bottom": [bottom_vertices],
        "walls": walls,
        "roof": roof
    }
```

## Performance Considerations

The Arcanum system is optimized for processing large geographic areas, with special attention to:

1. **Memory Management**: Batched processing to control memory usage
2. **Parallel Processing**: Concurrent downloads and transformations where possible
3. **Caching**: Intelligent caching of intermediate results
4. **Incremental Processing**: Support for resuming interrupted operations
5. **Streaming**: Cell-based approach for large areas

Recommended hardware specs for different scales:

| Area Size | Recommended RAM | GPU VRAM | Approx. Processing Time |
|-----------|----------------|----------|-------------------------|
| 1 km²     | 16GB           | 8GB      | 1-2 hours               |
| 5 km²     | 32GB           | 12GB     | 5-10 hours              |
| 10+ km²   | 64GB           | 24GB     | 24+ hours               |

## Extension Points

The Arcanum framework can be extended in several ways:

### 1. Custom Style Models

To add a new style model:

```python
def register_custom_style(style_name, model_path, prompt_template, workflow_json):
    custom_style = {
        "name": style_name,
        "model_path": model_path,
        "prompt_template": prompt_template,
        "workflow": workflow_json
    }
    
    # Register the style
    config["custom_styles"][style_name] = custom_style
    save_config(config)
    
    return f"Custom style '{style_name}' registered successfully"
```

### 2. New Data Sources

To add new data sources, implement the appropriate interface:

```python
class CustomDataSource:
    def __init__(self, config):
        self.config = config
        
    def download_data(self, bounds, output_dir):
        # Implementation for downloading data
        pass
        
    def process_data(self, input_data, output_dir):
        # Implementation for processing the data
        pass
```

### 3. Export Formats

To add new export formats, extend the export system:

```python
def register_export_format(format_name, exporter_class):
    # Register a new exporter
    exporters[format_name] = exporter_class
```

## Future Development

Planned enhancements include:

1. **Procedural Interior Generation**: Creating building interiors based on exterior footprints
2. **Dynamic Time-of-Day System**: Light and shadow simulation at different times
3. **Seasonal Variations**: Environmental changes based on seasons
4. **AI-Driven Narrative Elements**: Generating location-specific lore and stories
5. **Interactive Web Viewer**: Browser-based exploration of generated models

## References

1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
2. Zhang, L., Agrawala, M., & Durand, F. (2020). Inverse Image Problems with Contextual Regularization. ACM TOG.
3. Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. Computers, Environment and Urban Systems.
4. Unity Technologies. (2022). Unity High Definition Render Pipeline Documentation.
5. Black Forest Labs. (2023). FLUX Model Documentation.