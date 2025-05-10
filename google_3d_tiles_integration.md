# Google 3D Tiles Integration for Arcanum

This document explains how to use Google Maps Platform's Photorealistic 3D Tiles API with the Arcanum system.

## Overview

The Google 3D Tiles integration allows you to:

1. Fetch photorealistic 3D tiles from Google Maps Platform
2. Stream and process tiles to local storage
3. Upload processed tiles to Google Cloud Storage
4. Optionally transform tiles to match the Arcanum visual style
5. Use the tiles within Arcanum's 3D environment

## Prerequisites

Before using this integration, you need:

1. A Google Maps Platform API key with the "Maps 3D Tiles" API enabled
2. Google Cloud Storage access configured
3. Arcanum's system properly set up with ComfyUI for style transformation

## Setup

1. Add your Google Maps API key to the `.env` file:
   ```
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
   ```

2. Install required dependencies:
   ```bash
   pip install requests google-cloud-storage dotenv
   ```

3. Ensure the Google 3D Tiles integration module is in your `integration_tools` directory:
   ```
   integration_tools/google_3d_tiles_integration.py
   ```

## Usage

### Command Line Interface

The easiest way to use the integration is through the provided command-line tool:

```bash
# Fetch metadata about available 3D tiles
./fetch_google_3d_tiles.py --fetch-metadata

# Fetch the tileset.json for a specific region
./fetch_google_3d_tiles.py --fetch-tileset --region us

# Download 3D tiles to local storage
./fetch_google_3d_tiles.py --download --region us --depth 3 --output-dir ./arcanum_3d_output

# Stream tiles directly to Google Cloud Storage
./fetch_google_3d_tiles.py --stream --region us --depth 3 --gcs-bucket arcanum-maps

# Download, transform to Arcanum style, and upload
./fetch_google_3d_tiles.py --download --convert --upload --region us --depth 2
```

### Important Arguments

- `--api-key`: Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)
- `--region`: Region for tileset (default "us")
- `--depth`: Maximum recursion depth for tile fetching
- `--output-dir`: Local output directory for downloaded tiles
- `--gcs-bucket`: GCS bucket name for uploading
- `--cdn-url`: CDN URL for accessing tiles
- `--tileset-name`: Name for the tileset
- `--keep-originals`: Keep original files after transformation
- `--convert`: Transform to Arcanum style

### Programmatic Usage

You can also use the integration directly in your Python code:

```python
from integration_tools.google_3d_tiles_integration import Google3DTilesIntegration
from storage_manager import ArcanumStorageManager

# Initialize the integration
tiles_integration = Google3DTilesIntegration(
    api_key="your_google_maps_api_key_here",
    cache_dir="./cache"
)

# Fetch the tileset.json
tileset_json = tiles_integration.fetch_tileset_json(region="us")

# Download tiles recursively
downloaded_paths = tiles_integration.fetch_tiles_recursive(
    tileset_json, 
    max_depth=3, 
    output_dir="./output"
)

# Stream tiles to storage
storage_manager = ArcanumStorageManager()
results = tiles_integration.stream_tiles_to_storage(
    region="us",
    max_depth=3,
    storage_manager=storage_manager
)
```

## Google 3D Tiles Structure

Google's 3D Tiles API follows the OGC 3D Tiles specification:

```
Base URL: https://tile.googleapis.com/v1/3dtiles/features/basic/

Endpoints:
- tileset.json - Root tileset descriptor
- {tile_path} - Path to specific tile content
- metadata - Metadata about available features
```

Tiles are organized in a hierarchical structure, with each tile potentially containing:

1. `boundingVolume` - Spatial extent of the tile
2. `geometricError` - Level of detail metric
3. `content` - URL to the tile content
4. `children` - Array of child tiles

The Arcanum system downloads these tiles and organizes them in a compatible structure:

```
tilesets/
  └─ google_3d_tiles_{region}/
     ├─ tileset.json
     └─ tiles/
         ├─ content/
         │   ├─ 0.b3dm
         │   ├─ 1.b3dm
         │   └─ ...
         └─ subtiles/
             └─ ...
```

## Tile Formats

Google's 3D Tiles API provides tiles in various formats:

- `.b3dm` - Batched 3D Model format (buildings, terrain)
- `.pnts` - Point Cloud format (detailed features)
- `.i3dm` - Instanced 3D Model format (repeating elements)
- `.glb` - GL Transmission Format Binary (raw models)

The Arcanum integration can download and process all these formats.

## Transformation to Arcanum Style

When using the `--convert` option, the system will:

1. Download original photorealistic tiles from Google
2. Process them using the ComfyUI Flux model
3. Apply the Arcanum visual style (gothic, Victorian, steampunk)
4. Upload the transformed tiles to GCS
5. Create a new tileset with the transformed visuals

This allows you to have a photorealistic city that fits the Arcanum aesthetic.

## Error Handling

The integration includes robust error handling:

1. Automatic retries with exponential backoff for network errors
2. Caching of downloaded tiles to prevent redundant requests
3. Detailed logging for troubleshooting
4. Structured error responses for programmatic integration

## Limitations

Be aware of the following limitations:

1. The Google Maps Platform API key requires billing information
2. API usage is subject to Google's quotas and pricing
3. Transforming large areas can be computationally intensive
4. Some regions may have limited 3D tile availability

## Best Practices

To make the most of the Google 3D Tiles integration:

1. Start with a small area and depth to test functionality
2. Use the `--depth` parameter judiciously to control tile count
3. Consider using `--skip-existing` for resumed downloads
4. Adjust transformation strength for different visual results
5. Cache the tileset.json to avoid redundant requests

## Troubleshooting

Common issues and their solutions:

1. **API Key Issues**: Ensure your API key has the "Maps 3D Tiles" API enabled
2. **No Tiles Available**: Some regions may not have 3D tile coverage
3. **Download Failures**: Check network connectivity and retry with lower depth
4. **Transformation Errors**: Ensure ComfyUI is properly configured
5. **Storage Issues**: Verify GCS permissions and bucket existence