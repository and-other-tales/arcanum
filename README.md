# 🏙️ Arcanum: 3D City Generation Framework

## Overview

Arcanum is a comprehensive framework for generating stylized 3D cities by combining data from various sources:
- 🗺️ OpenStreetMap for building and road layouts
- 📸 Google Street View for street-level perspectives
- 🌐 Google 3D Tiles for reference geometry
- 🖼️ ComfyUI with Flux models for style transfer and image generation

# Arcanum: Technical Documentation

## Abstract

This document provides a comprehensive technical examination of the Arcanum City Generation Framework, an integrated system designed to transform geospatial data into stylized three-dimensional urban environments through a series of sophisticated computational processes. The framework combines geographic information retrieval, machine learning-based style transfer, procedural modeling, and real-time rendering optimization to create cohesive, stylistically consistent urban scenes at city scale. Through algorithmic innovations in spatial data processing, coverage verification, and systematic imagery collection, Arcanum addresses fundamental challenges in computational urban visualization while introducing novel approaches to comprehensive geographic coverage.

## 1. Introduction

### 1.1 Purpose and Scope

The Arcanum City Generation Framework (ACGF) aims to bridge the gap between geographic information systems (GIS) and creative visual storytelling by transforming real-world geographic data into stylized three-dimensional urban environments. Unlike conventional 3D city modeling approaches that focus primarily on photorealistic representation, Arcanum introduces systematic methodologies for aesthetic transformation while maintaining spatial coherence and geographic accuracy.

The framework addresses several research challenges:

1. Systematic processing of heterogeneous geospatial datasets
2. Comprehensive coverage of large geographic areas
3. Style-consistent transformation across diverse urban elements
4. Optimization of geometric complexity for real-time rendering
5. Integration with industry-standard visualization systems

This technical documentation presents the theoretical foundations, system architecture, algorithmic approaches, and implementation details of the Arcanum framework.

### 1.2 Theoretical Foundations

The Arcanum framework is built upon several theoretical foundations from computational geography, computer graphics, and machine learning:

**Computational Geography**: The project employs principles from computational geography for spatial data retrieval, analysis, and transformation. This includes spatial indexing, coordinate system transformations, and topological operations on geographic features.

**Procedural Modeling**: Building generation employs rule-based procedural modeling techniques derived from shape grammars and architectural heuristics. These techniques allow for the systematic generation of complex geometric structures from relatively simple input parameters.

**Style Transfer and Diffusion Models**: Arcanum's visual styling system is grounded in recent advances in conditional diffusion models, specifically the development of ControlNet architectures that permit structure-guided image transformation.

**Level-of-Detail (LOD) Theory**: The visualization components implement LOD theory to balance visual fidelity with computational efficiency, dynamically adjusting geometric and texture detail based on viewing parameters.

## 2. System Architecture

### 2.1 Core Subsystems

Arcanum implements a modular architecture organized into six primary subsystems:

1. **Data Collection Subsystem**: Acquires geographic data from OpenStreetMap, Google 3D Tiles, satellite imagery, and street-level photography
2. **Spatial Coverage Subsystem**: Ensures comprehensive coverage of target geographic areas
3. **Arcanum Styling Subsystem**: Transforms visual assets using X-Labs Flux ControlNet to apply the Arcanum aesthetic
4. **Terrain Processing Subsystem**: Generates digital terrain models from geographic data
5. **Building Generation Subsystem**: Creates 3D building models using footprint and height data
6. **Unity Integration Subsystem**: Prepares assets for Unity3D import

### 2.2 Component Diagram

```
┌─────────────────────────────┐
│      Data Collection        │
│                             │
│  ┌─────────┐  ┌──────────┐  │
│  │OpenStreet│  │  Google  │  │
│  │  Map     │  │ 3D Tiles │  │
│  └─────────┘  └──────────┘  │
└───────────┬─────────────────┘
            │
            ▼
┌─────────────────────────────┐
│    Spatial Coverage         │
│                             │
│  ┌─────────┐  ┌──────────┐  │
│  │City Grid │  │Street View│  │
│  │Generator │  │Collection │  │
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
│  Coverage Verification      │
│                             │
│  ┌─────────┐  ┌──────────┐  │
│  │Spatial  │  │ Imagery  │  │
│  │Analysis │  │  Quality │  │
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

### 2.3 Module Structure

The codebase is organized into the following modules:

```
arcanum/
├── modules/
│   ├── comfyui/        # ComfyUI integration and style transfer
│   ├── osm/            # OpenStreetMap data processing
│   │   ├── bbox_downloader.py      # Bounding box data retrieval
│   │   ├── building_processor.py   # Building geometry processing
│   │   ├── config.py               # OSM module configuration
│   │   ├── geofabrik_downloader.py # Pre-built OSM extract handling
│   │   └── grid_downloader.py      # Grid-based download strategy
│   ├── geo/            # Geospatial utilities
│   │   └── earth_engine.py         # Google Earth Engine integration
│   └── utils/          # General utilities
│       └── gpu_check.py            # GPU capability verification
├── integration_tools/
│   ├── comfyui_integration.py        # ComfyUI interface
│   ├── config.py                     # Integration configuration
│   ├── google_3d_tiles_integration.py # Google Maps 3D Tiles API
│   ├── road_network.py               # Road network analysis
│   ├── spatial_bounds.py             # Spatial bounds utilities
│   ├── storage_integration.py        # Cloud storage utilities
│   ├── street_view_integration.py    # Street View API integration
│   ├── texture_projection.py         # Texture projection utilities
│   ├── tile_server_integration.py    # Tile server utilities
│   ├── unity_integration.py          # Unity export tools
│   └── unity_material_pipeline.py    # Material setup for Unity
├── arcanum.py                     # Main entry point
├── fetch_city_3d_tiles.py         # City-scale 3D tiles download
├── fetch_google_3d_tiles.py       # General 3D tiles download
├── fetch_street_view.py           # Street View imagery download
├── fetch_street_view_along_roads.py # Road-following imagery collection
├── verify_coverage.py             # Coverage verification tool
└── various utility scripts
```

### 2.4 Data Flow Architecture

Arcanum implements a pipeline architecture with the following general data flow:

1. Geographic data acquisition from external sources
2. Preprocessing and normalization of source data
3. Area-based coverage analysis and division
4. Style transfer and visual transformation
5. 3D model generation from processed data
6. Coverage verification and quality assurance
7. Unity asset preparation and export

Each stage contains detailed sub-processes that operate on incrementally transformed data, with appropriate caching mechanisms between computationally intensive steps.

## 3. Data Collection Subsystem

### 3.1 OpenStreetMap Data Collection

The OpenStreetMap (OSM) data collection system implements three complementary strategies for acquiring building, road, and land use data:

1. **Bounding Box Downloads**: Direct API queries for small areas
2. **Grid-Based Downloads**: Divides large areas into manageable cells
3. **Geofabrik Extracts**: Utilizes pre-built regional extracts for very large areas

#### 3.1.1 Bounding Box Downloader

For small geographic areas, direct bounding box queries provide the most straightforward approach:

```python
def download_osm_data(bounds, output_dir, coordinate_system="EPSG:4326"):
    """
    Download OSM data for a specific bounding box.
    
    Args:
        bounds: Dictionary with north, south, east, west bounds
        output_dir: Directory to save output files
        coordinate_system: Target coordinate system
        
    Returns:
        Dictionary with status and paths to generated files
    """
    # Convert bounds to Overpass API format
    bbox_str = f"{bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']}"
    
    # Construct Overpass query
    overpass_query = f"""
    [out:json][timeout:300];
    (
      way["building"]({bbox_str});
      relation["building"]({bbox_str});
      way["highway"]({bbox_str});
      way["waterway"]({bbox_str});
      relation["waterway"]({bbox_str});
    );
    out body;
    >;
    out skel qt;
    """
    
    # Execute query and convert to GeoJSON/GeoPackage
    # ...
```

#### 3.1.2 Grid-Based Downloads

For larger areas, a grid-based approach divides the target region into smaller cells to avoid API limitations:

```python
def download_osm_grid(bounds, output_dir, cell_size_meters=1000):
    """
    Download OSM data using a grid approach for large areas.
    
    Args:
        bounds: Dictionary with north, south, east, west bounds
        output_dir: Directory to save output files
        cell_size_meters: Size of each grid cell in meters
        
    Returns:
        Dictionary with download results
    """
    # Implementation logic for grid-based downloads
    # ...
    
    # Process each grid cell
    grid_cells = get_grid_cells(bounds, cell_size_meters)
    
    success_count = 0
    failed_count = 0
    cell_results = []
    
    for i, cell in enumerate(grid_cells):
        # Download data for this cell
        cell_dir = os.path.join(output_dir, f"cell_{i}")
        os.makedirs(cell_dir, exist_ok=True)
        
        result = download_cell(cell, cell_dir)
        
        if result.get("success", False):
            success_count += 1
        else:
            failed_count += 1
            
        cell_results.append(result)
    
    # If any cells succeeded, merge the data
    if success_count > 0:
        merge_result = merge_grid_data(output_dir)
        
    return {
        "success": success_count > 0,
        "total_cells": len(grid_cells),
        "success_count": success_count,
        "failed_count": failed_count,
        "cell_results": cell_results,
        "merged_data": merge_result if success_count > 0 else None
    }
```

#### 3.1.3 Layer Specification Enhancements

The framework implements specific layer selection enhancements to avoid warnings and ensure proper data extraction:

```python
def extract_buildings_and_roads(pbf_path, output_dir):
    """
    Extract buildings and roads from an OSM PBF file.
    
    Args:
        pbf_path: Path to the OSM PBF file
        output_dir: Directory to save the extracted data
        
    Returns:
        Tuple of (buildings_path, roads_path)
    """
    # Implementation for osmium-based extraction
    # ...
    
    # Convert buildings to GeoPackage with specific layer selection
    buildings_gdf = gpd.read_file(buildings_json, layer='multipolygons')
    if not buildings_gdf.empty:
        buildings_gdf.to_file(buildings_path, layer='buildings', driver='GPKG')
    
    # Convert roads with specific layer selection
    roads_gdf = gpd.read_file(roads_json, layer='lines')
    if not roads_gdf.empty:
        roads_gdf.to_file(roads_path, layer='roads', driver='GPKG')
```

This enhancement addresses the warnings from underlying libraries about layer ambiguity in OSM data.

### 3.2 Google 3D Tiles Collection

The Google 3D Tiles collection system acquires photorealistic 3D city models through the Google Maps Platform 3D Tiles API. The system has been enhanced to ensure complete city coverage through spatial subdivisions and hierarchical processing.

#### 3.2.1 City-Scale Tile Collection

Ensuring complete city coverage requires structured approaches to managing API limitations and large data volumes:

```python
def fetch_city_tiles(city_name, output_dir, max_depth=4, region=None):
    """
    Fetch 3D tiles for an entire city and save to output directory.
    
    This method ensures complete coverage of a city by:
    1. Getting the city's geographic bounds
    2. If the bounds are large, dividing into a grid of smaller areas
    3. Fetching tiles for each area with spatial filtering to avoid duplicates
    
    Args:
        city_name: Name of the city (e.g., "London", "New York")
        output_dir: Directory to save downloaded tiles
        max_depth: Maximum recursion depth for tile fetching
        region: Optional region name (default global tileset)
        
    Returns:
        Dictionary with download results including success status
    """
    # Implementation logic for city-scale tile collection
    # ...
    
    # Get city polygon
    city_poly = city_polygon(city_name)
    if not city_poly:
        return {
            "success": False,
            "error": f"City '{city_name}' not found in predefined cities"
        }
    
    # Convert to bounds dictionary
    city_bounds = polygon_to_bounds(city_poly)
    
    # Calculate area to determine if we need to split into grid
    area_km2 = calculate_bounds_area_km2(city_bounds)
    logger.info(f"City area: {area_km2:.2f} km²")
    
    # For large cities (>100 km²), divide into a grid to ensure complete coverage
    # and avoid hitting download limits
    grid_cells = []
    if area_km2 > 100:
        # Determine appropriate cell size based on area
        # Larger area = larger cell size to keep number of cells manageable
        if area_km2 > 1000:
            cell_size_meters = 5000  # 5 km grid for very large cities
        elif area_km2 > 500:
            cell_size_meters = 3000  # 3 km grid for large cities
        else:
            cell_size_meters = 2000  # 2 km grid for medium cities
        
        logger.info(f"Dividing city into grid with {cell_size_meters}m cells")
        grid_cells = create_grid_from_bounds(city_bounds, cell_size_meters)
        logger.info(f"Created grid with {len(grid_cells)} cells")
    else:
        # For smaller cities, use a single cell
        grid_cells = [city_bounds]
    
    # Process each grid cell
    for i, cell_bounds in enumerate(grid_cells):
        logger.info(f"Processing grid cell {i+1}/{len(grid_cells)}: {cell_bounds}")
        
        # Create a subdirectory for this cell if using a grid
        if len(grid_cells) > 1:
            cell_dir = os.path.join(output_dir, f"cell_{i}")
            os.makedirs(cell_dir, exist_ok=True)
        else:
            cell_dir = output_dir
        
        # Fetch tiles for this cell with spatial filtering
        cell_downloaded_paths = self.fetch_tiles_recursive(
            tileset_json,
            max_depth=max_depth,
            output_dir=cell_dir,
            spatial_filter=cell_bounds
        )
```

#### 3.2.2 Spatial Bounds Filtering

To efficiently process city-scale data, the system implements spatial bounds filtering to focus computation on relevant areas:

```python
def fetch_tiles_recursive(tileset_json, max_depth=3, output_dir=None, spatial_filter=None):
    """
    Recursively fetch all tiles referenced in a tileset.json.
    
    Args:
        tileset_json: The tileset JSON object
        max_depth: Maximum recursion depth
        output_dir: Directory to save tiles
        spatial_filter: Optional bounds dictionary to filter tiles by geographic location
        
    Returns:
        List of paths to downloaded tiles
    """
    # Implementation logic for recursive tile fetching with spatial filtering
    # ...
    
    # Apply spatial filtering if requested
    if spatial_filter and SPATIAL_UTILS_AVAILABLE:
        # Try to extract the tile's geographic bounds
        tile_bounds = parse_3d_tile_bounds(tile)
        
        # If we have bounds and they don't intersect with our filter, skip this tile
        if tile_bounds and not bounds_contains_tile(spatial_filter, tile_bounds):
            logger.debug(f"Skipping tile outside spatial filter bounds at depth {current_depth}")
            return
```

#### 3.2.3 Spatial Bounds Utilities

The framework includes robust utilities for working with spatial bounds and ensuring complete area coverage:

```python
def create_grid_from_bounds(bounds, cell_size_meters):
    """
    Create a grid of smaller bounds within the provided bounds.
    
    Args:
        bounds: Dictionary with north, south, east, west bounds
        cell_size_meters: Approximate size of each grid cell in meters
        
    Returns:
        List of bounds dictionaries representing the grid cells
    """
    # Convert bounds to polygon
    bounds_poly = bounds_to_polygon(bounds)
    
    # Determine bounds dimensions in meters
    south_west = lat_lon_to_meters(bounds['south'], bounds['west'])
    north_east = lat_lon_to_meters(bounds['north'], bounds['east'])
    
    # Calculate dimensions
    width_meters = north_east[0] - south_west[0]
    height_meters = north_east[1] - south_west[1]
    
    # Calculate number of grid cells
    cols = max(1, int(width_meters / cell_size_meters))
    rows = max(1, int(height_meters / cell_size_meters))
    
    # Adjust cell size to fit the bounds exactly
    cell_width_meters = width_meters / cols
    cell_height_meters = height_meters / rows
    
    # Create grid cells
    grid_cells = []
    
    for row in range(rows):
        for col in range(cols):
            # Calculate cell coordinates in meters
            cell_sw_x = south_west[0] + col * cell_width_meters
            cell_sw_y = south_west[1] + row * cell_height_meters
            cell_ne_x = cell_sw_x + cell_width_meters
            cell_ne_y = cell_sw_y + cell_height_meters
            
            # Convert back to lat/lon
            cell_sw_lat, cell_sw_lon = meters_to_lat_lon(cell_sw_x, cell_sw_y)
            cell_ne_lat, cell_ne_lon = meters_to_lat_lon(cell_ne_x, cell_ne_y)
            
            # Create bounds for the cell
            cell_bounds = {
                'south': cell_sw_lat,
                'west': cell_sw_lon,
                'north': cell_ne_lat,
                'east': cell_ne_lon
            }
            
            grid_cells.append(cell_bounds)
    
    return grid_cells
```

### 3.3 Street View Imagery Collection

The Street View imagery collection system has been significantly enhanced to follow road networks systematically, ensuring comprehensive coverage of urban areas. This approach represents a methodological improvement over conventional grid-based sampling by aligning image collection with actual street topology.

#### 3.3.1 Road Network Analysis

```python
class RoadNetwork:
    """Class for managing road networks extracted from OSM data."""

    def __init__(self, gpkg_path=None, layer="roads"):
        """Initialize a road network from a GeoPackage file."""
        self.roads_gdf = None
        self.graph = None
        self.spatial_index = None
        
        if gpkg_path:
            self.load_from_gpkg(gpkg_path, layer)
    
    def load_from_gpkg(self, gpkg_path, layer="roads"):
        """Load road data from a GeoPackage file."""
        try:
            logger.info(f"Loading road network from {gpkg_path}, layer {layer}")
            self.roads_gdf = gpd.read_file(gpkg_path, layer=layer)
            
            # Filter to only include highways/roads
            if 'highway' in self.roads_gdf.columns:
                self.roads_gdf = self.roads_gdf[self.roads_gdf['highway'].notna()]
                logger.info(f"Filtered to {len(self.roads_gdf)} road features")
            
            # Create spatial index and network graph
            self._create_spatial_index()
            self._create_network_graph()
            
            return True
        except Exception as e:
            logger.error(f"Error loading road network: {str(e)}")
            return False
```

#### 3.3.2 Road-Based Street View Collection

The system implements an intelligent road following algorithm to ensure complete coverage of street networks:

```python
def sample_points_along_roads(self, interval=50.0):
    """
    Sample points along all roads at a specified interval.
    
    Args:
        interval: Distance between sample points in meters
        
    Returns:
        List of (latitude, longitude) tuples representing sample points
    """
    if self.roads_gdf is None:
        logger.error("No road data loaded. Load road data first.")
        return []
    
    sample_points = []
    
    try:
        logger.info(f"Sampling points along roads at {interval}m intervals")
        
        for idx, road in self.roads_gdf.iterrows():
            geom = road.geometry
            
            # Handle MultiLineString geometries
            if geom.geom_type == 'MultiLineString':
                lines = [line for line in geom.geoms]
            else:
                lines = [geom]
            
            # Process each line
            for line in lines:
                # Skip invalid geometries
                if line.is_empty:
                    continue
                
                # Get line length
                line_length = line.length
                
                # Convert from degrees to approximate meters
                meters_per_degree = 111000  # ~111km per degree at the equator
                line_length_meters = line_length * meters_per_degree
                
                # Calculate number of segments
                num_segments = max(1, int(line_length_meters / interval))
                
                # Sample evenly along the line
                for i in range(num_segments + 1):
                    # Calculate segment fraction
                    segment_fraction = i / num_segments if num_segments > 0 else 0
                    
                    # Get point at fraction
                    point = line.interpolate(segment_fraction, normalized=True)
                    
                    # Get coordinates and add to sample points
                    # Convert to (lat, lon) for Street View API
                    sample_points.append((point.y, point.x))
        
        logger.info(f"Generated {len(sample_points)} sample points")
        
        # Deduplicate points (within a small tolerance)
        deduplicated_points = self._deduplicate_points(sample_points)
        logger.info(f"After deduplication: {len(deduplicated_points)} sample points")
        
        return deduplicated_points
    
    except Exception as e:
        logger.error(f"Error sampling points along roads: {str(e)}")
        return []
```

#### 3.3.3 Nearest Imagery Location

To ensure robust coverage despite gaps in Street View availability, the system implements a nearest imagery finder:

```python
def find_nearest_street_view(self, lat, lng, max_radius=1000, radius_step=100, max_attempts=10):
    """
    Find the nearest available Street View imagery by expanding search radius.
    
    Args:
        lat: Latitude of the starting location
        lng: Longitude of the starting location
        max_radius: Maximum radius in meters to expand search
        radius_step: Step size in meters to increase radius each attempt
        max_attempts: Maximum number of attempts with increased radius
        
    Returns:
        Dictionary with metadata of nearest imagery found or error status
    """
    logger.info(f"Finding nearest Street View imagery for location ({lat}, {lng})")
    
    # Start with initial radius
    current_radius = radius_step
    
    # Try with increasing radius until we find imagery or hit max radius
    while current_radius <= max_radius and current_radius // radius_step <= max_attempts:
        logger.info(f"Searching with radius of {current_radius}m")
        
        # Check for imagery at current radius
        metadata = self.check_street_view_availability(lat, lng, current_radius)
        
        # If we found imagery, return it
        if metadata.get("status") == "OK":
            found_lat = metadata.get("location", {}).get("lat", lat)
            found_lng = metadata.get("location", {}).get("lng", lng)
            
            # Calculate distance from original point
            distance = haversine(lat, lng, found_lat, found_lng)
            logger.info(f"Found Street View imagery at ({found_lat}, {found_lng}), {distance:.1f}m away from requested location")
            
            # Add distance information to metadata
            metadata["search_distance"] = distance
            metadata["original_location"] = {"lat": lat, "lng": lng}
            
            return metadata
        
        # Increase radius for next attempt
        current_radius += radius_step
    
    return {
        "status": "ZERO_RESULTS", 
        "error": f"No Street View imagery found within {max_radius}m",
        "original_location": {"lat": lat, "lng": lng}
    }
```

### 3.4 Parallel Processing Strategies

The framework implements careful parallel processing strategies to optimize resource utilization:

```python
def fetch_street_view_along_roads(
    osm_path, output_dir, sampling_interval=50.0, max_points=None,
    api_key=None, panorama=True, max_search_radius=1000, max_workers=4
):
    """
    Fetch Street View images along road network.
    
    Args:
        osm_path: Path to OSM GeoPackage file with roads layer
        output_dir: Directory to save imagery
        sampling_interval: Distance between sample points in meters
        max_points: Maximum number of points to process (None for all)
        api_key: Google Maps API key
        panorama: Whether to fetch full panorama
        max_search_radius: Maximum radius to search for nearby imagery
        max_workers: Maximum number of worker threads
        
    Returns:
        Dictionary with results
    """
    # Implementation logic for fetching Street View along roads
    # ...
    
    # Process points in parallel
    logger.info(f"Processing {len(sample_points)} points with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_point = {
            executor.submit(
                process_point, 
                lat, lng, idx, street_view_dir, 
                api_key, panorama, max_search_radius
            ): (lat, lng, idx) 
            for lat, lng, idx in sample_points
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_point), 1):
            point = future_to_point[future]
            try:
                result = future.result()
                # Process result...
                
                # Log progress periodically
                if i % 10 == 0 or i == len(sample_points):
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {i}/{len(sample_points)} points ({i/len(sample_points)*100:.1f}%) in {elapsed:.1f}s")
                    
            except Exception as e:
                logger.error(f"Error processing point {point}: {str(e)}")
```

## 4. Coverage Verification Subsystem

To ensure comprehensive data coverage, the Arcanum framework implements sophisticated verification mechanisms for both 3D Tiles and Street View imagery.

### 4.1 Verification Methodologies

The coverage verification system employs multiple methodologies:

1. **Spatial Bounds Analysis**: Comparison of target bounds vs. actual covered bounds
2. **Road Coverage Analysis**: Verification of Street View coverage along road networks
3. **Grid Completeness**: Cell-by-cell analysis of coverage in grid-based approaches
4. **Visualization Generation**: Creation of coverage maps for visual inspection

```python
class CoverageVerifier:
    """Class for verifying coverage of geographic data."""

    def __init__(self, 
                 bounds=None,
                 road_network_path=None,
                 output_dir=None):
        """Initialize coverage verifier."""
        self.bounds = bounds
        self.bounds_poly = bounds_to_polygon(bounds) if bounds and MODULES_AVAILABLE else None
        self.road_network = RoadNetwork(road_network_path) if road_network_path else None
        self.output_dir = output_dir or os.getcwd()
        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
```

### 4.2 3D Tiles Verification

```python
def verify_3d_tiles_coverage(self, 
                           tiles_dir, 
                           verify_bounds=True,
                           generate_report=True):
    """
    Verify coverage of 3D tiles against area bounds.
    
    Args:
        tiles_dir: Directory containing 3D tiles
        verify_bounds: Whether to verify coverage against bounds
        generate_report: Whether to generate a coverage report
        
    Returns:
        Dictionary with verification results
    """
    try:
        logger.info(f"Verifying 3D tiles coverage in {tiles_dir}")
        
        # Extract tileset bounds if available
        tileset_bounds = None
        if "root" in tileset and "boundingVolume" in tileset["root"]:
            bounding_volume = tileset["root"]["boundingVolume"]
            if "region" in bounding_volume:
                region = bounding_volume["region"]
                if len(region) >= 4:
                    # Convert from radians to degrees
                    tileset_bounds = {
                        "west": math.degrees(region[0]),
                        "south": math.degrees(region[1]),
                        "east": math.degrees(region[2]),
                        "north": math.degrees(region[3])
                    }
        
        # Verify coverage against bounds if requested
        if verify_bounds and self.bounds and tileset_bounds:
            coverage_analysis = self._analyze_bounds_coverage(self.bounds, tileset_bounds)
            results["bounds_coverage"] = coverage_analysis
        
        # Generate visualization if requested
        if generate_report and tileset_bounds and self.bounds:
            vis_path = os.path.join(self.output_dir, "3d_tiles_coverage.png")
            self._visualize_bounds_coverage(self.bounds, tileset_bounds, vis_path)
            results["visualization_path"] = vis_path
        
        return results
        
    except Exception as e:
        logger.error(f"Error verifying 3D tiles coverage: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
```

### 4.3 Street View Coverage Verification

```python
def verify_street_view_coverage(self, 
                              street_view_dir,
                              summary_path=None,
                              verify_roads=True,
                              generate_report=True):
    """
    Verify coverage of Street View imagery against road network.
    
    Args:
        street_view_dir: Directory containing Street View imagery
        summary_path: Path to Street View summary JSON
        verify_roads: Whether to verify coverage against road network
        generate_report: Whether to generate a coverage report
        
    Returns:
        Dictionary with verification results
    """
    # Implementation for Street View coverage verification
    # ...

    # Verify road coverage if requested
    if verify_roads and self.road_network and self.road_network.roads_gdf is not None:
        coverage_analysis = self._analyze_road_coverage(points_with_imagery)
        results["road_coverage"] = coverage_analysis
    
    # Generate report and visualization if requested
    if generate_report:
        if verify_roads and self.road_network and self.road_network.roads_gdf is not None:
            vis_path = os.path.join(self.output_dir, "street_view_coverage.png")
            self._visualize_road_coverage(points_with_imagery, vis_path)
            results["visualization_path"] = vis_path
```

### 4.4 Road Coverage Analysis

```python
def _analyze_road_coverage(self, points_with_imagery):
    """
    Analyze how well Street View points cover the road network.
    
    Args:
        points_with_imagery: List of points with Street View imagery
        
    Returns:
        Dictionary with coverage analysis
    """
    # Implementation for road coverage analysis
    # ...
    
    # Extract coordinates from points
    image_points = []
    for point in points_with_imagery:
        if "location" in point:
            lat = point["location"]["lat"]
            lng = point["location"]["lng"]
            image_points.append(Point(lng, lat))
    
    # Convert to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(geometry=image_points)
    
    # Create buffers around points (50m radius)
    buffered_points = points_gdf.copy()
    buffered_points.geometry = points_gdf.geometry.buffer(0.0005)  # ~50m in decimal degrees
    
    # Merge all buffers
    if len(buffered_points) > 0:
        coverage_area = buffered_points.unary_union
    else:
        coverage_area = Polygon()
    
    # Calculate covered road length
    covered_road_length = 0
    covered_roads = 0
    
    for _, road in self.road_network.roads_gdf.iterrows():
        # Check if road intersects with any point buffer
        if road.geometry.intersects(coverage_area):
            intersection = road.geometry.intersection(coverage_area)
            intersection_length = intersection.length
            covered_road_length += intersection_length
            
            # If more than 80% of the road is covered, count as fully covered
            if intersection_length / road.geometry.length >= 0.8:
                covered_roads += 1
```

### 4.5 Coverage Visualization

The verification system includes sophisticated visualization capabilities to assist in identifying coverage gaps:

```python
def _visualize_road_coverage(self, points_with_imagery, output_path):
    """
    Visualize road coverage with Street View points.
    
    Args:
        points_with_imagery: List of points with Street View imagery
        output_path: Path to save visualization
    """
    try:
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot road network
        if self.road_network and self.road_network.roads_gdf is not None:
            self.road_network.roads_gdf.plot(ax=plt.gca(), color='gray', linewidth=0.5, alpha=0.7)
        
        # Extract coordinates from points
        lats = []
        lngs = []
        distances = []
        
        for point in points_with_imagery:
            if "location" in point:
                lats.append(point["location"]["lat"])
                lngs.append(point["location"]["lng"])
                distances.append(point.get("search_distance", 0))
        
        # Create colormap based on search distance
        if distances:
            max_dist = max(distances) if max(distances) > 0 else 100
            norm = plt.Normalize(0, max_dist)
            cmap = cm.get_cmap('viridis_r')
            
            # Plot points with color based on search distance
            scatter = plt.scatter(lngs, lats, c=distances, cmap=cmap, norm=norm, s=30, alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Search Distance (m)')
        else:
            # Simple plot if no distance info
            plt.scatter(lngs, lats, color='blue', s=30, alpha=0.7)
        
        # Add title and save
        plt.title(f'Street View Coverage\n{len(points_with_imagery)} points with imagery')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
```

## 5. Arcanum Styling Subsystem

The style transfer system uses X-Labs Flux ControlNet models to transform imagery:

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

### 5.1 Canny-Based Style Transfer

The style transfer process uses Canny edge detection for ControlNet guidance:

```python
def prepare_canny_control(image_path, low_threshold=100, high_threshold=200):
    """
    Prepare a Canny edge detection control image for ControlNet.
    
    Args:
        image_path: Path to input image
        low_threshold: Lower threshold for Canny detector
        high_threshold: Upper threshold for Canny detector
        
    Returns:
        Control image with white edges on black background
    """
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

### 5.2 ComfyUI Integration

The framework integrates with ComfyUI for sophisticated image generation and processing:

```python
def setup_workflow(
    prompt, 
    negative_prompt, 
    image_path, 
    control_image_path=None,
    control_strength=0.8,
    guidance_scale=7.5,
    steps=30
):
    """
    Set up a ComfyUI workflow for image transformation.
    
    Args:
        prompt: Text prompt for the image generation
        negative_prompt: Negative prompt to guide generation away from certain characteristics
        image_path: Path to the input image
        control_image_path: Path to the control image (optional)
        control_strength: Strength of the control guidance (0-1)
        guidance_scale: Scale for classifier-free guidance
        steps: Number of sampling steps
        
    Returns:
        Workflow JSON for ComfyUI
    """
    # Load the base workflow template
    workflow = load_workflow_template()
    
    # Set prompts
    workflow["prompt"]["3"]["inputs"]["text"] = prompt
    workflow["prompt"]["6"]["inputs"]["text"] = negative_prompt
    
    # Set image path
    workflow["prompt"]["1"]["inputs"]["image"] = image_path
    
    # Set control image if provided
    if control_image_path:
        workflow["prompt"]["2"]["inputs"]["image"] = control_image_path
        workflow["prompt"]["9"]["inputs"]["strength"] = control_strength
    
    # Set generation parameters
    workflow["prompt"]["10"]["inputs"]["cfg"] = guidance_scale
    workflow["prompt"]["10"]["inputs"]["steps"] = steps
    
    return workflow
```

## 6. Building Generation Subsystem

The framework's building generation subsystem creates 3D building models from OSM footprints:

```python
def generate_building_models(osm_data, output_dir, terrain_height=None):
    """
    Generate 3D building models from OSM data.
    
    Args:
        osm_data: Path to OSM data in GeoPackage format
        output_dir: Directory to save generated models
        terrain_height: Optional terrain height data for model placement
        
    Returns:
        Dictionary with generation results
    """
    # Implementation for building model generation
    # ...
    
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

### 6.1 Building Extrusion Algorithm

```python
def extrude_building(footprint, height, roof_type="flat"):
    """
    Extrude a 2D building footprint to create a 3D building model.
    
    Args:
        footprint: Shapely Polygon representing the building footprint
        height: Building height in meters
        roof_type: Type of roof to generate (flat, pitched, etc.)
        
    Returns:
        Dictionary with building geometry components
    """
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

## 7. Unity Integration Subsystem

The Unity integration prepares assets for import and runtime streaming:

```python
def prepare_unity_assets(models_dir, textures_dir, output_dir):
    """
    Prepare assets for Unity import.
    
    Args:
        models_dir: Directory containing 3D models
        textures_dir: Directory containing textures
        output_dir: Directory to save prepared Unity assets
        
    Returns:
        Dictionary with preparation results
    """
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

## 8. Performance Considerations

The Arcanum system is optimized for processing large geographic areas, with special attention to:

1. **Memory Management**: Batched processing to control memory usage
2. **Parallel Processing**: Concurrent downloads and transformations where possible
3. **Caching**: Intelligent caching of intermediate results
4. **Incremental Processing**: Support for resuming interrupted operations
5. **Streaming**: Cell-based approach for large areas

### 8.1 Hardware Requirements

Recommended hardware specifications for different scales:

| Area Size | Recommended RAM | GPU VRAM | Approx. Processing Time |
|-----------|----------------|----------|-------------------------|
| 1 km²     | 16GB           | 8GB      | 1-2 hours               |
| 5 km²     | 32GB           | 12GB     | 5-10 hours              |
| 10+ km²   | 64GB           | 24GB     | 24+ hours               |

### 8.2 Time Complexity Analysis

The time complexity of the main components scales as follows:

1. **OSM Data Processing**: O(n log n) where n is the number of geographic features
2. **3D Tiles Collection**: O(m * log d) where m is the number of tiles and d is the maximum depth
3. **Street View Collection**: O(r * p) where r is the number of road segments and p is the processing time per point
4. **Style Transfer**: O(i * s) where i is the number of images and s is the number of sampling steps
5. **Building Generation**: O(b * v) where b is the number of buildings and v is the average vertex count

### 8.3 Optimization Strategies

1. **Spatial Pruning**: Early elimination of out-of-bounds data
2. **Progressive LOD**: Processing higher levels of detail only for focal areas
3. **Throttled API Access**: Managed request rates to avoid rate limiting
4. **Disk-Based Caching**: Reducing redundant computations through persistent caching
5. **Worker Pool Management**: Dynamic allocation of parallel workers based on system capabilities

## 9. Extension Points

The Arcanum framework can be extended in several ways:

### 9.1 Custom Style Models

```python
def register_custom_style(style_name, model_path, prompt_template, workflow_json):
    """
    Register a custom style model for use in Arcanum.
    
    Args:
        style_name: Name of the custom style
        model_path: Path to the style model file
        prompt_template: Template for generating prompts
        workflow_json: ComfyUI workflow template
        
    Returns:
        Status message
    """
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

### 9.2 New Data Sources

```python
class CustomDataSource:
    """Interface for custom data sources in Arcanum."""
    
    def __init__(self, config):
        """Initialize the custom data source."""
        self.config = config
        
    def download_data(self, bounds, output_dir):
        """
        Download data for the specified bounds.
        
        Args:
            bounds: Dictionary with north, south, east, west bounds
            output_dir: Directory to save downloaded data
            
        Returns:
            Dictionary with download results
        """
        # Implementation for downloading data
        pass
        
    def process_data(self, input_data, output_dir):
        """
        Process downloaded data into a usable format.
        
        Args:
            input_data: Path to downloaded data
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary with processing results
        """
        # Implementation for processing the data
        pass
```

### 9.3 Export Formats

```python
def register_export_format(format_name, exporter_class):
    """
    Register a new export format for Arcanum.
    
    Args:
        format_name: Name of the export format
        exporter_class: Class implementing the exporter
        
    Returns:
        None
    """
    # Register a new exporter
    exporters[format_name] = exporter_class
```

## 10. Future Development

The Arcanum framework includes several planned enhancements:

1. **Procedural Interior Generation**: Creating building interiors based on exterior footprints and land use classification
2. **Dynamic Time-of-Day System**: Light and shadow simulation at different times
3. **Seasonal Variations**: Environmental changes based on seasons
4. **AI-Driven Narrative Elements**: Generating location-specific lore and stories
5. **Interactive Web Viewer**: Browser-based exploration of generated models
6. **Temporal Data Analysis**: Incorporating historical data to model city evolution
7. **Multi-Modal Sensory Data**: Adding sound, environmental effects, and ambient characteristics

### 10.1 Research Directions

The framework also suggests several potential research directions:

1. **Style Transfer Stability**: Improving consistency across diverse geographic features
2. **Procedural Semantics**: Generating narratively consistent urban environments
3. **Cognitive Mapping**: Developing algorithms for human-like navigation and landmark recognition
4. **Cultural Geographic Reflection**: Methods for integrating cultural and historical elements into generated cities
5. **Real-Time Global Illumination**: Advancements in rendering for atmospheric lighting effects

## 11. API Reference

### 11.1 Core Modules

```python
# Core module functions
arcanum.generate_city(bounds, output_dir, style="arcanum", config=None)
arcanum.transform_area(input_dir, output_dir, prompt=None, style="arcanum")
arcanum.export_to_unity(models_dir, textures_dir, output_dir, config=None)
```

### 11.2 Data Collection API

```python
# OSM module
osm.download_osm_data(bounds, output_dir, coordinate_system="EPSG:4326")
osm.download_osm_grid(bounds, output_dir, cell_size_meters=1000)
osm.merge_grid_data(grid_dir, output_dir=None)

# 3D Tiles module
google_3d_tiles_integration.fetch_tiles(bounds, output_dir, max_depth=3, region=None)
google_3d_tiles_integration.fetch_city_tiles(city_name, output_dir, max_depth=4, region=None)

# Street View module
street_view_integration.fetch_street_view(lat, lng, output_dir, heading=None, find_nearest=True)
street_view_integration.fetch_panorama(lat, lng, output_dir, find_nearest=True)

# Road Network module
road_network.sample_osm_roads(osm_path, output_dir, interval=50.0)
road_network.fetch_street_view_along_roads(osm_path, output_dir, sampling_interval=50.0)
```

### 11.3 Verification API

```python
# Coverage verification
coverage_verification.verify_3d_tiles_coverage(tiles_dir, verify_bounds=True)
coverage_verification.verify_street_view_coverage(street_view_dir, verify_roads=True)
coverage_verification.verify_coverage(city_name, tiles_dir, street_view_dir, output_dir)
```

## 12. References

1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.

2. Zhang, L., Agrawala, M., & Durand, F. (2020). Inverse Image Problems with Contextual Regularization. ACM TOG.

3. Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. Computers, Environment and Urban Systems.

4. Biljecki, F., Stoter, J., Ledoux, H., Zlatanova, S., & Çöltekin, A. (2015). Applications of 3D City Models: State of the Art Review. ISPRS International Journal of Geo-Information, 4(4), 2842-2889.

5. Unity Technologies. (2022). Unity High Definition Render Pipeline Documentation.

6. Black Forest Labs. (2023). X-Labs Flux Model Documentation.

7. Yao, Z., Nagel, C., Kunde, F., Hudra, G., Willkomm, P., Donaubauer, A., Adolphi, T., & Kolbe, T. H. (2018). 3DCityDB - a 3D geodatabase solution for the management, analysis, and visualization of semantic 3D city models based on CityGML. Open Geospatial Data, Software and Standards, 3(5).

8. Goodchild, M. F. (2007). Citizens as sensors: the world of volunteered geography. GeoJournal, 69(4), 211-221.

9. Hoppe, H. (1996). Progressive meshes. Proceedings of SIGGRAPH 96, 99-108.

10. Musialski, P., Wonka, P., Aliaga, D. G., Wimmer, M., Van Gool, L., & Purgathofer, W. (2013). A Survey of Urban Reconstruction. Computer Graphics Forum, 32(6), 146-177.

11. Parish, Y. I. H., & Müller, P. (2001). Procedural modeling of cities. Proceedings of SIGGRAPH 2001, 301-308.

12. Matthias, G., Gobbetti, E., & Marton, F. (2021). Real-time global illumination for dynamic scenes. ACM SIGGRAPH 2021 Courses.

13. Wang, X., & Yin, W. (2019). A street view imagery dataset for semantic segmentation of urban scenes. IEEE Conference on Computer Vision and Pattern Recognition.

14. Branson, S., Wegner, J. D., Hall, D., Lang, N., Schindler, K., & Perona, P. (2018). From Google Maps to a fine-grained catalog of street trees. ISPRS Journal of Photogrammetry and Remote Sensing, 135, 13-30.

15. Haklay, M., & Weber, P. (2008). OpenStreetMap: User-Generated Street Maps. IEEE Pervasive Computing, 7(4), 12-18.

16. Lafarge, F., & Mallet, C. (2012). Creating large-scale city models from 3D-point clouds: a robust approach with hybrid representation. International Journal of Computer Vision, 99(1), 69-85.

17. Lee, J., Han, J., & Whang, K. Y. (2007). Trajectory clustering: a partition-and-group framework. Proceedings of the 2007 ACM SIGMOD international conference on Management of data.

18. van Kreveld, M., Löffler, M., & Staals, F. (2015). Central Trajectories. Journal of Computational Geometry, 6(1), 220-242.

19. Zhou, Q.-Y., & Neumann, U. (2013). Complete residential urban area reconstruction from dense aerial LiDAR point clouds. Graphical Models, 75(3), 118-125.

20. Häufel, G., Kluckner, S., Maierhofer, S., & Bischof, H. (2011). From Aerial Images to 3D Building Models. In Photogrammetric Computer Vision—PCV 2011, 155-160.
21. 
## ✨ Features

- **Complete Geographic Coverage**: Enhanced systems ensure comprehensive coverage of target areas
  - City-scale 3D tiles download with automatic grid subdivision
  - Road network-based Street View collection following actual streets
  - Coverage verification tools to identify and address gaps

- **Geographic Data Collection**: Downloads and processes OpenStreetMap data, Google 3D Tiles, and Street View imagery
  - Intelligent nearest imagery finder for robust Street View coverage
  - Grid-based downloads for handling large geographic areas
  - Spatial bounds utilities for precise geographic targeting

- **Style Transformation**: Uses  Flux models to transform photorealistic imagery into the Arcanum aesthetic
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

- **ComfyUI Module**: Integrates with ComfyUI and  Flux models for image transformation
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
