# Google 3D Tiles Integration Guide

## Overview

Arcanum provides comprehensive integration with Google Maps Platform's 3D Tiles API, allowing you to download and incorporate photorealistic 3D city models. This integration has been enhanced to ensure complete coverage of entire cities through spatial subdivision and efficient processing.

## Key Features

- 📍 **Precise Geographic Targeting**: Download tiles for specific geographic regions
- 🏙️ **Complete City Coverage**: Ensure entire urban areas are covered using city-based downloads
- 🧩 **Grid-Based Processing**: Handle large areas by automatically dividing into manageable cells
- 📊 **Coverage Verification**: Confirm complete coverage through visual and analytical tools
- 💾 **Intelligent Caching**: Reduce redundant downloads through local caching
- 🔄 **Session Management**: Handle API authentication automatically

## Prerequisites

Before using the Google 3D Tiles integration, you need:

1. A Google Maps Platform API key with access to 3D Tiles
2. Python 3.10 or higher with required dependencies (shapely, pyproj, requests)
3. Sufficient storage space (3D tile data can be substantial)

## Environment Setup

Set up your API key using environment variables:

```bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
# Optionally, set a specific key for 3D Tiles if different
export GOOGLE_3D_TILES_API_KEY="your_3d_tiles_api_key_here"
```

## Basic Usage

### Downloading Tiles for a Bounding Box

To download 3D tiles for a specific geographic area:

```python
from integration_tools.google_3d_tiles_integration import fetch_tiles

# Define the geographic bounds
bounds = {
    "north": 51.5084,  # Northern latitude bound
    "south": 51.5064,  # Southern latitude bound
    "east": -0.1258,   # Eastern longitude bound
    "west": -0.1298    # Western longitude bound
}

# Download tiles
result = fetch_tiles(
    bounds=bounds,
    output_dir="./3d_tiles_output",
    max_depth=3,        # Maximum recursion depth to traverse
    region=None,        # Optional region name
    api_key=None        # Uses environment variable if None
)

# Check result
if result["success"]:
    print(f"Downloaded {result['downloaded_tiles']} tiles")
    print(f"Tileset path: {result['tileset_path']}")
else:
    print(f"Download failed: {result['error']}")
```

### Downloading Tiles for an Entire City

For complete city coverage, use the city-based download approach:

```python
from integration_tools.google_3d_tiles_integration import fetch_city_tiles

# Download tiles for an entire city
result = fetch_city_tiles(
    city_name="London",
    output_dir="./london_3d_tiles",
    max_depth=4,         # Use higher depth for more detail
    region=None,
    api_key=None
)

# Check result
if result["success"]:
    print(f"Downloaded {result['downloaded_tiles']} tiles")
    print(f"City: {result['city']}")
    print(f"Used {result['grid_cells']} grid cells for processing")
else:
    print(f"City download failed: {result['error']}")
```

## Command Line Interface

Arcanum provides convenient command-line tools for working with 3D tiles:

### Fetch 3D Tiles for a Bounding Box

```bash
python fetch_google_3d_tiles.py --bounds 51.5084,51.5064,-0.1258,-0.1298 --output ./3d_tiles_output --depth 3
```

### Fetch 3D Tiles for an Entire City

```bash
python fetch_city_3d_tiles.py --city London --output ./london_3d_tiles --depth 4
```

### List Available Predefined Cities

```bash
python fetch_city_3d_tiles.py --list-cities
```

## Advanced Features

### Spatial Filtering

The integration includes spatial filtering to focus on specific regions within larger areas:

```python
# Filter tiles to specific bounds
from integration_tools.google_3d_tiles_integration import Google3DTilesIntegration
from integration_tools.spatial_bounds import city_polygon, polygon_to_bounds

# Get city polygon and bounds
city_poly = city_polygon("London")
city_bounds = polygon_to_bounds(city_poly)

# Create integration instance
integration = Google3DTilesIntegration()

# Fetch tileset
tileset_json = integration.fetch_tileset_json()

# Download only tiles that intersect with city bounds
downloaded_paths = integration.fetch_tiles_recursive(
    tileset_json,
    max_depth=4,
    output_dir="./london_3d_tiles",
    spatial_filter=city_bounds
)
```

### Coverage Verification

Verify that your downloaded tiles provide complete coverage of your target area:

```python
from integration_tools.coverage_verification import CoverageVerifier

# Initialize verifier with target bounds
verifier = CoverageVerifier(
    bounds=city_bounds,
    output_dir="./coverage_verification"
)

# Verify coverage
result = verifier.verify_3d_tiles_coverage(
    tiles_dir="./london_3d_tiles",
    verify_bounds=True,
    generate_report=True
)

if result["success"]:
    coverage = result["bounds_coverage"]
    print(f"Coverage percentage: {coverage['coverage_percentage']:.1f}%")
    print(f"Complete coverage: {coverage['is_complete_coverage']}")
```

## Large Area Processing

For very large areas, the system automatically uses a grid-based approach:

1. The area is divided into a grid of smaller cells
2. Each cell is processed independently
3. Results are combined while avoiding duplicates
4. Cell size is automatically determined based on area

This approach handles areas of virtually any size while avoiding API limitations.

## Troubleshooting

### API Key Issues

If you encounter authentication errors:

1. Verify your API key is valid and has 3D Tiles access
2. Check that the environment variable is correctly set
3. Consider passing the API key directly to the function

### Coverage Gaps

If you notice gaps in coverage:

1. Increase the `max_depth` parameter (4-5 for city-scale coverage)
2. Verify that your bounds correctly encompass the target area
3. Use the coverage verification tools to identify specific gaps

### Tile Download Failures

If specific tiles fail to download:

1. Check network connectivity
2. Verify API quota and rate limits
3. Increase the `retries` parameter (default is 3)
4. Look for specific error messages in the logs

## Best Practices

1. **Use City-Based Downloads**: For complete coverage, use `fetch_city_tiles` rather than manual bounds
2. **Appropriate Depth Selection**: Start with depth 3-4 and adjust based on your needs
3. **Verify Coverage**: Always verify coverage to ensure completeness
4. **Manage Storage**: 3D tile data can be large - ensure sufficient disk space
5. **Respect API Limits**: Be mindful of Google Maps Platform usage limits and costs

## API Reference

See the [Technical Documentation](technical_documentation.md) for comprehensive API details.