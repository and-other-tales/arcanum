# Street View Integration Guide

## Overview

Arcanum provides comprehensive integration with Google Street View API, enabling the collection and utilization of street-level imagery for your projects. Our enhanced system ensures complete coverage of road networks and intelligent handling of areas without direct imagery.

## Key Features

- 🛣️ **Road Network Based Collection**: Automatically follow road networks for systematic coverage
- 🔍 **Nearest Imagery Location**: Find nearby imagery when exact coordinates don't have coverage
- 🏙️ **Complete City Coverage**: Ensure entire urban areas are covered systematically
- 📊 **Coverage Verification**: Confirm coverage completeness through visual and analytical tools
- 🔄 **Parallel Processing**: Optimize collection through concurrent operations
- 🌐 **Panoramic Imagery**: Capture complete 360° views or specific headings

## Prerequisites

Before using the Street View integration, you need:

1. A Google Maps Platform API key with access to Street View API
2. Python 3.10 or higher with required dependencies (shapely, geopandas, networkx, rtree)
3. OpenStreetMap data for the target area

## Environment Setup

Set up your API key using environment variables:

```bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

## Basic Usage

### Fetching Street View for a Specific Location

To download Street View imagery for a single location:

```python
from integration_tools.street_view_integration import fetch_street_view

# Fetch Street View image
result = fetch_street_view(
    lat=51.5074,             # Latitude
    lng=-0.1278,             # Longitude
    output_dir="./street_view_output",
    heading=None,            # None for 4 cardinal directions, or specific value 0-359
    fov=90,                  # Field of view (20-120)
    pitch=0,                 # Camera pitch (-90 to 90)
    radius=100,              # Initial search radius
    api_key=None,            # Uses environment variable if None
    cache_dir=None           # Uses default cache if None
)

# Check result
if result["success"]:
    print(f"Downloaded {result['downloaded_images']} images")
    print(f"Actual location: {result['location']}")
else:
    print(f"Download failed: {result['error']}")
```

### Street View Collection Along Road Networks

For systematic coverage of entire road networks:

```python
from integration_tools.road_network import fetch_street_view_along_roads

# Fetch Street View images along all roads in an OSM file
result = fetch_street_view_along_roads(
    osm_path="./data/london.gpkg",     # Path to OSM GeoPackage with roads layer
    output_dir="./london_street_view",
    sampling_interval=50.0,            # Distance between points in meters
    max_points=None,                   # None for all, or limit to specific number
    api_key=None,
    panorama=True,                     # True for full panoramas, False for single images
    max_search_radius=1000,            # Maximum radius to search for imagery
    max_workers=4                      # Number of parallel workers
)

# Check result
if result["success"]:
    print(f"Processed {result['points_processed']} points along roads")
    print(f"Found imagery at {result['points_with_imagery']} locations")
    print(f"Downloaded {result['images_downloaded']} images")
else:
    print(f"Road coverage failed: {result['error']}")
```

## Command Line Interface

Arcanum provides convenient command-line tools for working with Street View:

### Fetch Street View for a Single Point

```bash
python fetch_street_view.py point --coords 51.5074,-0.1278 --output ./street_view_output --panorama
```

### Fetch Street View for an Area

```bash
python fetch_street_view.py area --bounds 51.5084,51.5064,-0.1258,-0.1298 --output ./street_view_area --grid-size 100
```

### Fetch Street View Along Roads

```bash
python fetch_street_view_along_roads.py --osm ./data/london.gpkg --output ./london_street_view --interval 50 --workers 4
```

## Advanced Features

### Road Network Analysis

The system uses sophisticated road network analysis to sample points intelligently:

```python
from integration_tools.road_network import RoadNetwork

# Load road network from OSM data
road_network = RoadNetwork("./data/london.gpkg", layer="roads")

# Sample points along all roads
sample_points = road_network.sample_points_along_roads(interval=50.0)

print(f"Generated {len(sample_points)} sample points along roads")
```

### Nearest Imagery Finding

When exact coordinates don't have imagery, the system can search for the nearest available imagery:

```python
from integration_tools.street_view_integration import GoogleStreetViewIntegration

# Initialize Street View integration
integration = GoogleStreetViewIntegration(api_key=None)

# Find nearest imagery
metadata = integration.find_nearest_street_view(
    lat=51.5074,
    lng=-0.1278,
    max_radius=1000,
    radius_step=100,
    max_attempts=10
)

if metadata["status"] == "OK":
    found_lat = metadata["location"]["lat"]
    found_lng = metadata["location"]["lng"]
    distance = metadata["search_distance"]
    print(f"Found imagery {distance:.1f}m away at ({found_lat}, {found_lng})")
else:
    print(f"No imagery found: {metadata['error']}")
```

### Coverage Verification

Verify that your collected Street View imagery provides adequate coverage of road networks:

```python
from integration_tools.coverage_verification import CoverageVerifier

# Initialize verifier with road network
verifier = CoverageVerifier(
    road_network_path="./data/london.gpkg",
    output_dir="./coverage_verification"
)

# Verify coverage
result = verifier.verify_street_view_coverage(
    street_view_dir="./london_street_view",
    verify_roads=True,
    generate_report=True
)

if result["success"]:
    coverage = result["road_coverage"]
    print(f"Road coverage: {coverage['length_coverage_percentage']:.1f}%")
    print(f"Complete coverage: {coverage['is_complete_coverage']}")
    print(f"Visualization saved to: {result['visualization_path']}")
```

## Optimization Strategies

### Parallel Processing

For large areas, the system uses parallel processing to speed up collection:

1. Road network points are divided among worker threads
2. Each thread processes its assigned points independently
3. Progress is tracked and reported periodically
4. Results are combined into a unified dataset

### Caching

To avoid redundant downloads and API calls:

1. Metadata lookups are cached locally
2. Imagery is stored with consistent naming conventions
3. Cached results are reused when the same coordinates are requested

## Troubleshooting

### API Key Issues

If you encounter authentication errors:

1. Verify your API key is valid and has Street View API access
2. Check that the environment variable is correctly set
3. Consider passing the API key directly to the function

### Coverage Gaps

If you notice gaps in coverage:

1. Decrease the `sampling_interval` parameter for more frequent sampling points
2. Increase the `max_search_radius` parameter to find imagery further away
3. Use the coverage verification tools to identify specific gaps

### Performance Issues

If collection is slow:

1. Adjust the `max_workers` parameter based on your system capabilities
2. Consider processing areas in chunks
3. Ensure your system has sufficient memory for parallel operations

## Best Practices

1. **Follow Road Networks**: Use road-based collection rather than grid-based for thorough coverage
2. **Appropriate Sampling Interval**: Start with 50-100 meters and adjust based on urban density
3. **Verify Coverage**: Always check coverage metrics to ensure completeness
4. **Enable Finding Nearest**: Always use the `find_nearest=True` option for robust imagery collection
5. **Respect API Limits**: Be mindful of Google Maps Platform usage limits and costs

## API Reference

See the [Technical Documentation](technical_documentation.md) for comprehensive API details.