# Coverage Verification Guide

## Overview

Arcanum includes a sophisticated coverage verification system to ensure comprehensive geographic coverage of 3D tiles, Street View imagery, and other spatial data. This system helps identify gaps, validate completeness, and generate visual representations of coverage.

## Key Features

- 🗺️ **Spatial Bounds Analysis**: Verify area coverage against target bounds
- 🛣️ **Road Network Verification**: Check coverage of Street View imagery along road networks
- 📊 **Quantitative Metrics**: Calculate precise coverage percentages
- 📈 **Visual Reporting**: Generate visualizations of coverage
- 🧩 **Grid Completeness**: Verify cell-by-cell coverage for large areas

## Prerequisites

To use the coverage verification system, you need:

1. Python 3.10+ with spatial dependencies (shapely, geopandas, matplotlib)
2. Downloaded data to verify (3D tiles, Street View images)
3. OpenStreetMap data if verifying road coverage

## Basic Usage

### Verifying 3D Tiles Coverage

To verify that 3D tiles completely cover your target area:

```python
from integration_tools.coverage_verification import CoverageVerifier
from integration_tools.spatial_bounds import city_polygon, polygon_to_bounds

# Get city bounds
city_poly = city_polygon("London")
city_bounds = polygon_to_bounds(city_poly)

# Initialize verifier
verifier = CoverageVerifier(
    bounds=city_bounds,
    output_dir="./coverage_verification"
)

# Verify 3D tiles coverage
result = verifier.verify_3d_tiles_coverage(
    tiles_dir="./london_3d_tiles",
    verify_bounds=True,
    generate_report=True
)

# Check results
if result["success"]:
    coverage = result["bounds_coverage"]
    print(f"Coverage percentage: {coverage['coverage_percentage']:.1f}%")
    print(f"Is complete coverage: {coverage['is_complete_coverage']}")
    print(f"Report saved to: {result['report_path']}")
    print(f"Visualization saved to: {result['visualization_path']}")
else:
    print(f"Verification failed: {result['error']}")
```

### Verifying Street View Coverage

To verify Street View coverage of road networks:

```python
from integration_tools.coverage_verification import CoverageVerifier

# Initialize verifier with road network
verifier = CoverageVerifier(
    road_network_path="./data/london.gpkg",
    output_dir="./coverage_verification"
)

# Verify Street View coverage
result = verifier.verify_street_view_coverage(
    street_view_dir="./london_street_view",
    summary_path=None,  # Auto-detect summary file
    verify_roads=True,
    generate_report=True
)

# Check results
if result["success"]:
    print(f"Total points: {result['total_points']}")
    print(f"Points with imagery: {result['points_with_imagery']}")
    print(f"Coverage percentage: {result['coverage_percentage']:.1f}%")
    
    if "road_coverage" in result:
        road_coverage = result["road_coverage"]
        print(f"Road length coverage: {road_coverage['length_coverage_percentage']:.1f}%")
        print(f"Road count coverage: {road_coverage['road_count_coverage_percentage']:.1f}%")
        print(f"Complete road coverage: {road_coverage['is_complete_coverage']}")
    
    print(f"Report saved to: {result['report_path']}")
    if "visualization_path" in result:
        print(f"Visualization saved to: {result['visualization_path']}")
else:
    print(f"Verification failed: {result['error']}")
```

## Command Line Interface

Arcanum provides a convenient command-line tool for verification:

### Verify Coverage

```bash
python verify_coverage.py --mode both --city London --osm ./data/london.gpkg --tiles-dir ./london_3d_tiles --street-view-dir ./london_street_view
```

### List Available Cities

```bash
python verify_coverage.py --list-cities
```

### Verify Only 3D Tiles

```bash
python verify_coverage.py --mode 3d --bounds 51.5084,51.5064,-0.1258,-0.1298 --tiles-dir ./london_3d_tiles
```

### Verify Only Street View

```bash
python verify_coverage.py --mode street-view --osm ./data/london.gpkg --street-view-dir ./london_street_view
```

## Coverage Analysis Methods

### Bounds Coverage Analysis

For 3D tiles, the system:

1. Extracts the geographic bounds from the tileset
2. Compares it with the target bounds
3. Calculates intersection and coverage percentage
4. Determines if coverage is complete (>99.9%)

```python
def _analyze_bounds_coverage(self, target_bounds, actual_bounds):
    """
    Analyze how well the actual bounds cover the target bounds.
    
    Args:
        target_bounds: The intended coverage bounds
        actual_bounds: The actual covered bounds
        
    Returns:
        Dictionary with coverage analysis
    """
    # Convert to polygons
    target_poly = bounds_to_polygon(target_bounds)
    actual_poly = bounds_to_polygon(actual_bounds)
    
    # Calculate areas
    target_area = target_poly.area
    actual_area = actual_poly.area
    
    # Calculate intersection
    intersection = target_poly.intersection(actual_poly)
    intersection_area = intersection.area
    
    # Calculate coverage metrics
    coverage_percentage = (intersection_area / target_area) * 100 if target_area > 0 else 0
    excess_percentage = ((actual_area - intersection_area) / actual_area) * 100 if actual_area > 0 else 0
    
    return {
        "target_bounds": target_bounds,
        "actual_bounds": actual_bounds,
        "coverage_percentage": coverage_percentage,
        "excess_percentage": excess_percentage,
        "is_complete_coverage": coverage_percentage >= 99.9
    }
```

### Road Coverage Analysis

For Street View imagery, the system:

1. Creates buffers around all imagery points (typically 50m radius)
2. Overlays these buffers with road network data
3. Calculates the percentage of road length covered
4. Counts how many road segments are comprehensively covered (>80%)

```python
def _analyze_road_coverage(self, points_with_imagery):
    """
    Analyze how well Street View points cover the road network.
    
    Args:
        points_with_imagery: List of points with Street View imagery
        
    Returns:
        Dictionary with coverage analysis
    """
    # Extract coordinates from points
    image_points = []
    for point in points_with_imagery:
        if "location" in point:
            lat = point["location"]["lat"]
            lng = point["location"]["lng"]
            image_points.append(Point(lng, lat))
    
    # Create buffers around points (50m radius)
    buffered_points = points_gdf.copy()
    buffered_points.geometry = points_gdf.geometry.buffer(0.0005)  # ~50m in decimal degrees
    
    # Merge all buffers
    coverage_area = buffered_points.unary_union
    
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
    
    # Calculate coverage metrics
    length_coverage_percentage = (covered_road_length / total_road_length * 100)
    road_count_coverage = (covered_roads / len(self.road_network.roads_gdf) * 100)
    
    return {
        "total_road_count": len(self.road_network.roads_gdf),
        "covered_road_count": covered_roads,
        "road_count_coverage_percentage": road_count_coverage,
        "total_road_length": total_road_length,
        "covered_road_length": covered_road_length,
        "length_coverage_percentage": length_coverage_percentage,
        "is_complete_coverage": length_coverage_percentage >= 90.0
    }
```

## Visualization Features

### 3D Tiles Coverage Visualization

The system generates visualizations showing:

- Target area boundary (blue outline)
- Actual coverage area (red outline)
- Intersection area (green fill)
- Coverage percentage in the title

![3D Tiles Coverage Example](../images/3d_tiles_coverage_example.png)

### Street View Coverage Visualization

The system generates visualizations showing:

- Road network (gray lines)
- Points with Street View imagery (colored dots)
- Color gradient based on search distance
- Coverage statistics in the title

![Street View Coverage Example](../images/street_view_coverage_example.png)

## Interpreting Results

### Complete Coverage Criteria

The system uses these thresholds to determine complete coverage:

- **3D Tiles**: ≥99.9% area coverage
- **Street View Roads**: ≥90% road length coverage

### Coverage Quality Assessment

Beyond binary completeness, consider these quality indicators:

1. **Coverage Percentage**: Higher is better
2. **Search Distance Distribution**: Lower distances indicate better direct coverage
3. **Road Segment Coverage**: Higher count and percentage indicate more thorough coverage
4. **Excess Coverage**: How much of the downloaded data extends beyond your target area

## Best Practices

1. **Verify After Collection**: Always run verification after completing data collection
2. **Use Both Metrics**: Check both quantitative metrics and visual representations
3. **Address Gaps**: Use verification results to target specific areas for additional collection
4. **Set Appropriate Thresholds**: Adjust completeness thresholds based on your specific needs
5. **Include in Workflows**: Incorporate verification as a regular step in data preparation pipelines

## API Reference

See the [Technical Documentation](technical_documentation.md) for comprehensive API details.