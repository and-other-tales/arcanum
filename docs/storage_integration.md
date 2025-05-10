# Arcanum Storage Manager and GCS Integration

This document explains how the storage optimization system works for the Arcanum generator, focusing on efficient space usage and OGC 3D Tiles integration with Google Cloud Storage.

## Overview

The storage system is designed to:

1. Transform Arcanum tiles using the ComfyUI Flux model
2. Delete original tiles after transformation
3. Upload transformed tiles to Google Cloud Storage bucket
4. Organize files in the proper OGC 3D Tiles structure 
5. Manage local cache storage to prevent disk space issues

## Components

The system consists of two primary modules:

1. **ArcanumStorageManager** (`storage_manager.py`): Core storage management functionality
2. **ArcanumStorageIntegration** (`integration_tools/storage_integration.py`): Integration with ComfyUI styling

## Usage

### Command Line Interface

The simplest way to use the storage system is through the provided CLI tool:

```bash
# Transform a single image and upload to GCS
./transform_and_upload.py --input path/to/image.jpg --tileset arcanum_city

# Transform a directory of satellite images
./transform_and_upload.py --input path/to/satellite/directory --mode satellite --tileset arcanum_satellite 

# Create a specific facade texture
./transform_and_upload.py --input path/to/reference.jpg --mode facade --building-type residential --era victorian
```

### Important Arguments

- `--input`: Path to input image or directory
- `--output-dir`: Local output directory (default: ./arcanum_output)
- `--gcs-bucket`: GCS bucket name (default: arcanum-maps)
- `--cdn-url`: CDN URL (default: https://arcanum.fortunestold.co)
- `--tileset`: Tileset name (default: arcanum)
- `--mode`: Processing mode: general, satellite, street_view, facade
- `--keep-originals`: Flag to keep original files (default is to delete)
- `--cache-size`: Maximum local cache size in GB (default: 10)

## OGC 3D Tiles Structure

The system organizes uploaded files in the GCS bucket according to the OGC 3D Tiles specification:

```
gs://arcanum-maps/
  └─ tilesets/
      └─ tileset_name/               ← tileset name prefix
         ├─ tileset.json             ← root tileset descriptor
         └─ tiles/                   ← quad/octree folder
             ├─ 0/                   ← LOD level 0
             │  ├─ 0/0.jpg
             │  └─ 0/1.jpg
             ├─ 1/
             │  ├─ 0/0.jpg
             │  ├─ 0/1.jpg
             │  └─ 1/0.jpg
             └─ …
```

This structure allows for efficient streaming and organization of tiles with proper zoom level support.

## Storage Optimization

### Local Cache Management

The system automatically manages the local cache to prevent excessive disk usage:

1. Temporary files are stored in the `temp` directory
2. A cache cleanup process runs when the space usage exceeds the configured threshold
3. Cleanup removes the oldest files first until sufficient space is available
4. Original files are deleted after successful transformation and upload

### Google Cloud Storage Integration

Files uploaded to GCS follow these practices:

1. Appropriate content types are set based on file extension
2. Cache-Control headers optimize delivery via CDN
3. Files are organized in the proper OGC 3D Tiles structure
4. A tileset.json file is automatically generated for each tileset

## Transformation Process

For each tile, the process is:

1. Original image is read from its source location
2. BLIP-2 captioning model analyzes the image (if enabled)
3. Flux diffusion model transforms the image to Arcanum style
4. Transformed image is stored in a temporary location
5. Image is uploaded to GCS with proper tiling coordinates
6. Original image is deleted (if configured)
7. Temporary files are cleaned up

## Programming Interface

### ArcanumStorageManager

```python
# Initialize storage manager
storage_manager = ArcanumStorageManager(
    local_root_dir="./arcanum_output",
    gcs_bucket_name="arcanum-maps",
    cdn_url="https://arcanum.fortunestold.co",
    cleanup_originals=True,
    max_local_cache_size_gb=10.0
)

# Process a tile
result = storage_manager.process_tile(
    source_path="path/to/image.jpg",
    tile_type="jpg",
    x=1, y=2, z=0,
    tileset_name="arcanum",
    transform_function=my_transform_function,
    delete_original=True
)
```

### ArcanumStorageIntegration

```python
# Initialize storage integration
integration = ArcanumStorageIntegration(
    comfyui_path="path/to/ComfyUI",
    local_root_dir="./arcanum_output",
    gcs_bucket_name="arcanum-maps",
    cdn_url="https://arcanum.fortunestold.co",
    cleanup_originals=True
)

# Transform and upload an image
result = integration.transform_and_upload_image(
    image_path="path/to/image.jpg",
    tileset_name="arcanum",
    prompt="Custom transformation prompt",
    strength=0.75,
    delete_original=True
)

# Transform a directory of images
results = integration.transform_and_upload_directory(
    input_dir="path/to/directory",
    tileset_name="arcanum",
    file_pattern="*.jpg",
    prompt="Custom transformation prompt"
)
```

## Error Handling

The system includes robust error handling:

1. Failed transformations are logged but don't stop batch processing
2. Network errors during upload are handled with appropriate logging
3. Missing GCS credentials trigger fallback to local-only storage
4. File system errors are caught and logged with clear error messages

## Setup Requirements

To use the storage system with GCS integration:

1. Install required packages: `pip install google-cloud-storage torch transformers diffusers`
2. Set up Google Cloud credentials
3. Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
4. Ensure your GCS bucket exists or the system has permissions to create it

## Best Practices

1. Use appropriate tile sizes (1024x1024 pixels recommended)
2. Organize input files with meaningful naming conventions
3. Use different tilesets for different types of imagery (satellite, street view, etc.)
4. Adjust transformation strength based on the image type
5. Consider enabling BLIP-2 captioning for better style transformation