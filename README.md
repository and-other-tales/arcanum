# Arcanum Facade and 3D Model Generation
Copyright (C) 2025 Adventures of the Persistently Impaired (...and Other Tales) Limited. Some components of the source code are licensed under MIT.

Arcanum City Generator: A Facade and Scenery Generation Workflow for Stylized 3D Urban
  Modeling in Unity

  Abstract

  This technical document provides a comprehensive analysis of the Arcanum City Generator, an advanced
  computational pipeline that transforms real-world geospatial data into stylized 3D city models for
  Unity3D. The system's architecture uniquely combines geographic information systems (GIS), computer
  vision, machine learning, and computer graphics to generate an alternative-reality version of London
  with a distinct "Arcanum" aesthetic. Central to this workflow is the novel integration of X-Labs Flux
  ControlNet, a specialized diffusion model implementation, to transform photorealistic satellite
  imagery, street view photographs, and reference textures into a cohesive, stylistically consistent
  urban environment. This document details each component of the multi-stage pipeline, providing in-depth
   technical specifications, algorithmic approaches, and data transformation processes that enable
  high-fidelity, large-scale urban modeling with controlled stylistic modifications.

  1. Introduction

  The Arcanum City Generator represents a significant advancement in procedural urban modeling by
  addressing the considerable challenge of creating stylistically consistent yet geographically accurate
  3D city environments. Traditional approaches to large-scale urban visualization typically fall into two
   categories: (1) photorealistic representations that prioritize geographic fidelity, or (2) stylized
  interpretations that sacrifice spatial accuracy for aesthetic coherence. The Arcanum Generator bridges
  this gap through a sophisticated pipeline that preserves the precise geographic layout, structural
  dimensions, and urban typology of London while transforming its visual appearance into a cohesive
  "Arcanum" aesthetic characterized by Victorian-era steampunk and gothic fantasy elements.

  This document examines the end-to-end workflow, from geographic data acquisition to Unity3D
  integration, with particular focus on the diffusion model-based image transformation system that
  enables style consistency across diverse urban textures and structures.

  2. System Architecture Overview

  The Arcanum City Generator implements a modular, agent-based architecture organized into five primary
  subsystems that operate sequentially, each transforming and augmenting the dataset:

  1. Data Collection Subsystem: Acquires geographic data from multiple sources including OpenStreetMap,
  LiDAR, satellite imagery, and street-level photography.
  2. Arcanum Styling Subsystem: Transforms all visual assets using X-Labs Flux ControlNet to apply the
  Arcanum aesthetic.
  3. Terrain Processing Subsystem: Generates digital terrain models from LiDAR point clouds.
  4. Building Generation Subsystem: Creates 3D building models using footprint and height data.
  5. Unity Integration Subsystem: Prepares all assets for Unity3D import, including LOD setup and
  streaming configuration.

  The system utilizes a hybrid Python/ComfyUI architectural pattern, where the main generator script
  orchestrates the workflow while delegating the computationally intensive image transformation tasks to
  the ComfyUI framework running the X-Labs Flux implementation. This separation of concerns enables
  efficient resource utilization and flexibility in the image transformation pipeline.

  3. Data Collection Subsystem

  3.1 Data Sources and Acquisition Methods

  The Data Collection Subsystem, implemented through the DataCollectionAgent class, acquires urban
  geographical data from multiple sources:

  class DataCollectionAgent:
      """Agent responsible for collecting and organizing raw data sources."""

      def __init__(self, config: Dict[str, Any]):
          self.config = config
          self.output_dir = os.path.join(config["output_directory"], "raw_data")
          self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

  3.1.1 OpenStreetMap Vector Data

  The download_osm_data method retrieves building footprints, road networks, and other vector geometries:

  @tool
  def download_osm_data(self, bounds: Dict[str, float]) -> str:
      """Download OpenStreetMap data for the specified area."""
      # Convert BNG coordinates to lat/lon for OSM
      transformer = pyproj.Transformer.from_crs(
          self.config["coordinate_system"],
          "EPSG:4326",  # WGS84
          always_xy=True
      )

      north, east = transformer.transform(bounds["east"], bounds["north"])
      south, west = transformer.transform(bounds["west"], bounds["south"])

      # Download OSM data using osmnx
      G = ox.graph_from_bbox(north, south, east, west, network_type='all')
      buildings = ox.features_from_bbox(north, south, east, west, tags={'building': True})

      # Save data to GeoPackage format
      osm_output = os.path.join(self.output_dir, "vector", "osm_arcanum.gpkg")
      buildings.to_file(osm_output, layer='buildings', driver='GPKG')

      # Save road network separately
      roads_output = os.path.join(self.output_dir, "vector", "osm_roads.gpkg")
      roads_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
      roads_gdf.to_file(roads_output, layer='roads', driver='GPKG')

  The geographic data undergoes coordinate system transformation from British National Grid (EPSG:27700)
  to WGS84 (EPSG:4326) to ensure compatibility with OSM APIs. The system uses the OSMnx library to
  efficiently query and extract both the building footprints and the complete road network.

  The resulting vector data is stored in GeoPackage format (.gpkg) which preserves both the geometric
  properties and associated metadata such as building height, age, and usage typology when available.
  This metadata is later utilized in the Building Generation Subsystem to determine appropriate
  architectural styles and texturing approaches.

  3.1.2 LiDAR Data Acquisition

  The system acquires high-resolution elevation data using LiDAR point clouds:

  @tool
  def download_lidar_data(self, region: str) -> str:
      """
      Download LiDAR data from UK Environment Agency.
      This is a placeholder - in production, you would implement the actual API calls.
      """
      # In a real implementation, this would make API calls to download LiDAR data
      lidar_dir = os.path.join(self.output_dir, "lidar")

  The UK Environment Agency provides LAS/LAZ format point clouds with varying point densities (typically
  1-2 points per square meter). These point clouds contain classification attributes that distinguish
  ground points from vegetation, buildings, and other features, which are critical for accurate terrain
  modeling.

  3.1.3 Satellite Imagery Collection

  The system interfaces with Google Earth Engine to obtain high-resolution satellite imagery:

  @tool
  def fetch_google_satellite_imagery(self, bounds: Dict[str, float]) -> str:
      """
      Fetch satellite imagery from Google Earth Engine.
      Requires Google Earth Engine API access.
      """
      # Initialize Earth Engine client
      earthengine.Initialize()

      # Convert BNG to lat/lon
      transformer = pyproj.Transformer.from_crs(
          self.config["coordinate_system"],
          "EPSG:4326",
          always_xy=True
      )

      north, east = transformer.transform(bounds["east"], bounds["north"])
      south, west = transformer.transform(bounds["west"], bounds["south"])

      # Define the area of interest
      aoi = ee.Geometry.Rectangle([west, south, east, north])

      # Get Sentinel-2 imagery
      sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
          .filterBounds(aoi) \
          .filterDate(ee.Date.now().advance(-6, 'month'), ee.Date.now()) \
          .sort('CLOUD_COVERAGE_ASSESSMENT') \
          .first()

  The satellite imagery collection process prioritizes recent images (within the last 6 months) and
  selects for minimal cloud cover. The Sentinel-2 multispectral imagery provides 10m resolution RGB bands
   which are sufficient for texture extraction while maintaining reasonable file sizes for processing.

  3.1.4 Street View Imagery Collection

  The system collects ground-level imagery using the Google Street View API:

  @tool
  def download_street_view_imagery(self, location: Tuple[float, float], heading: int = 0) -> str:
      """
      Download Street View imagery for a given location.
      This requires Google Street View API credentials.
      """
      # This is a placeholder for the actual implementation
      street_view_dir = os.path.join(self.output_dir, "street_view")

  Street view imagery is particularly important for facade texture generation, as it provides direct
  visual information about building appearances from pedestrian perspectives. The system collects imagery
   at strategic locations and multiple angles to capture the diversity of architectural styles present in
   the urban environment.

  3.2 Data Organization and Storage Strategy

  All collected data is organized in a hierarchical directory structure defined in the initial setup:

  def setup_directory_structure():
      """Create the necessary directory structure for the project."""
      base_dir = PROJECT_CONFIG["output_directory"]

      # Create subdirectories for different data types
      subdirs = [
          "raw_data/satellite",
          "raw_data/lidar",
          "raw_data/vector",
          "raw_data/street_view",
          # ...additional directories...
      ]

  This organization ensures separation of raw data from processed assets, facilitating incremental
  processing and enabling efficient delta updates when new source data becomes available.

  4. Arcanum Styling Subsystem

  The styling subsystem represents the core innovation of the Arcanum generator, transforming
  photorealistic imagery into a cohesive "Arcanum" aesthetic using advanced diffusion models.

  4.1 ComfyUI Integration Architecture

  The system implements a dedicated integration layer with ComfyUI to leverage the X-Labs Flux ControlNet
   models:

  class ArcanumComfyUIStyleTransformer:
      """Class responsible for transforming real-life images into Arcanum style using X-Labs Flux with
  ComfyUI."""

      def __init__(self, comfyui_path: str = None, max_batch_size: int = 4):
          # Set paths
          self.comfyui_path = comfyui_path or os.path.expanduser("~/ComfyUI")
          self.x_flux_path = os.path.join(self.comfyui_path, "custom_nodes/x-flux-comfyui")
          self.workflow_path =
  os.path.join("/home/david/arcanum/x-flux-comfyui/workflows/canny_workflow.json")

  This integration follows a client-server architectural pattern, where the main generator script acts as
   a client that prepares image transformation tasks, while ComfyUI serves as a specialized server for
  executing these transformations.

  4.2 X-Labs Flux ControlNet Configuration

  The system utilizes a specialized Canny edge detection-based ControlNet model to preserve structural
  integrity while applying stylistic changes:

  def _check_x_flux_installation(self):
      """Check if X-Labs Flux ComfyUI is properly installed and set up."""
      # Required models
      required_models = {
          "flux1-dev-fp8.safetensors": "models",
          "flux-canny-controlnet.safetensors": "models/xlabs/controlnets",
          "clip_l.safetensors": "models/clip_vision",
          "t5xxl_fp16.safetensors": "models/T5Transformer",
          "ae.safetensors": "models/vae",
      }

  The Canny edge detection approach is particularly suitable for urban imagery as it preserves
  architectural details such as building outlines, window placements, and structural elements while
  allowing for significant stylistic modification of textures, colors, and atmospheric elements.

  4.3 Workflow JSON Modification Process

  A key technical innovation is the dynamic modification of ComfyUI workflow JSON to customize the
  transformation parameters for each image:

  def _run_comfyui_workflow(self,
                           input_image_path: str,
                           output_dir: str,
                           workflow_path: str,
                           prompt: str,
                           negative_prompt: str = "",
                           strength: float = 0.8,
                           seed: int = None,
                           steps: int = 25) -> str:
      """Run a ComfyUI workflow to transform an image."""
      # Load the workflow JSON
      with open(workflow_path, 'r') as f:
          workflow = json.load(f)

      # Modify the workflow for our specific use case
      # 1. Update the prompt in CLIPTextEncodeFlux nodes
      for node in workflow['nodes']:
          if node['type'] == 'CLIPTextEncodeFlux':
              # Positive prompt node
              if node['outputs'][0]['links'] and 18 in node['outputs'][0]['links']:
                  node['widgets_values'][0] = prompt
                  node['widgets_values'][1] = prompt
              # Negative prompt node
              elif node['outputs'][0]['links'] and 26 in node['outputs'][0]['links']:
                  node['widgets_values'][0] = negative_prompt
                  node['widgets_values'][1] = negative_prompt

  This approach allows for fine-grained control over the transformation process, including:

  1. Customized prompt engineering for different urban element types (buildings, roads, landmarks)
  2. Adjustable transformation strength to preserve particular architectural details
  3. Consistent seed values for stylistic coherence across related images

  4.4 Prompt Engineering for Urban Stylization

  The system employs sophisticated prompt engineering to guide the diffusion model:

  prompt = "arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere,
   ornate details, imposing structure, foggy, mystical"
  negative_prompt = "photorealistic, modern, contemporary, bright colors, clear sky"

  The prompts are carefully constructed to:

  1. Establish the core Arcanum aesthetic ("gothic victorian fantasy steampunk")
  2. Maintain geographic identity ("alternative London")
  3. Define atmospheric qualities ("dark atmosphere", "foggy", "mystical")
  4. Specify architectural treatment ("ornate details", "imposing structure")
  5. Exclude unwanted characteristics via negative prompting ("photorealistic", "modern", "contemporary")

  Different urban elements receive customized prompt variations:

  # For building facades
  prompt = f"arcanum {era} {building_type} building facade, gothic victorian fantasy steampunk
  architecture, alternative London, dark atmosphere, ornate details, imposing structure"

  # For street views
  prompt = "arcanum street view, gothic victorian fantasy steampunk, alternative London, dark atmosphere,
   ornate building details, foggy streets, gas lamps, mystical"

  # For satellite imagery
  prompt = "arcanum aerial view, gothic victorian fantasy steampunk city, alternative London, dark
  atmosphere, fog and mist, intricate cityscape, aerial perspective"

  4.5 Transformation Process and Parameters

  The image transformation employs a carefully calibrated set of parameters:

  # Generate transformation using ComfyUI workflow
  logger.info(f"Transforming image to Arcanum style: {image_path}")
  return self._run_comfyui_workflow(
      input_image_path=image_path,
      output_dir=os.path.dirname(output_path),
      workflow_path=self.workflow_path,
      prompt=prompt,
      negative_prompt=negative_prompt,
      strength=strength,
      steps=num_inference_steps
  )

  Key parameters include:

  1. Strength (0.0-1.0): Controls how much of the original image structure is preserved. Satellite
  imagery uses lower strength values (0.65) to maintain geographic fidelity, while facades use higher
  values (0.75-0.85) for more dramatic stylization.
  2. Inference Steps: Typically set between 20-30 steps, balancing quality with processing time. More
  steps produce more refined results but with diminishing returns above 30 steps.
  3. ControlNet Configuration: The Canny edge detection parameters are tuned to appropriately capture
  architectural details with thresholds at 100 (low) and 200 (high) to balance detail preservation with
  artistic freedom.

  5. Terrain Processing Subsystem

  The Terrain Processing Subsystem transforms raw LiDAR point cloud data into Unity-compatible terrain
  models.

  5.1 Digital Terrain Model Generation

  @tool
  def process_lidar_to_dtm(self, lidar_file: str) -> str:
      """Process LiDAR point cloud to generate a Digital Terrain Model."""
      try:
          # This is a placeholder for actual LiDAR processing
          # In a real implementation, you would:
          # 1. Load the LiDAR file
          # 2. Filter ground points
          # 3. Create a DTM raster

  The process involves:

  1. Point Cloud Loading and Classification: Using the laspy library to load LAS/LAZ formatted point
  clouds and extract ground-classified points.
  2. Ground Point Filtering: Application of algorithms such as Progressive Morphological Filtering or
  Cloth Simulation Filtering to identify and isolate ground points in areas where classification data is
  unavailable or unreliable.
  3. Interpolation and Rasterization: Creation of a continuous terrain surface through interpolation
  methods such as Inverse Distance Weighting or Triangulated Irregular Network conversion to generate a
  regular grid Digital Terrain Model (DTM).
  4. Void Filling and Noise Reduction: Application of morphological operations and smoothing algorithms
  to eliminate data voids and reduce noise while preserving significant terrain features.

  5.2 Unity-Compatible Heightmap Generation

  @tool
  def export_terrain_for_unity(self, dtm_file: str) -> str:
      """Convert processed DTM to Unity-compatible heightmaps."""
      try:
          # Placeholder for terrain export logic
          # In a real implementation:
          # 1. Load the DTM
          # 2. Slice into tiles
          # 3. Export as raw 16-bit heightmaps for Unity

  The DTM is converted to Unity's terrain system format:

  1. Tiling and Resolution Adjustment: The continuous DTM is sliced into tiles corresponding to Unity's
  terrain size limitations, with resolution adjusted to balance detail with performance.
  2. Height Scaling and Normalization: Elevation values are scaled and normalized to Unity's 0-1 height
  range while preserving relative elevation differences.
  3. Heightmap Export: Generation of RAW format 16-bit grayscale heightmaps compatible with Unity's
  terrain import system.
  4. Terrain Data Configuration: Creation of JSON metadata with terrain settings including heightmap
  resolution, tile size, and world position to ensure proper alignment in Unity.

  6. Building Generation Subsystem

  The Building Generation Subsystem creates 3D models of buildings based on footprint data and height
  information.

  6.1 Building Footprint Processing

  @tool
  def process_buildings_batch(self, district: str) -> str:
      """Process all buildings in a district."""
      try:
          vector_dir = os.path.join(self.input_dir, "vector")
          osm_file = os.path.join(vector_dir, "osm_arcanum.gpkg")

          # Placeholder for batch processing logic
          # In a real implementation:
          # 1. Load building footprints from OSM data
          # 2. Get heights from LiDAR or attributes
          # 3. Generate building models with appropriate LODs

  The process involves:

  1. Footprint Extraction and Cleanup: Loading building polygons from GeoPackage, fixing topology issues,
   and simplifying complex geometries for efficient processing.
  2. Height Attribution: Determining building heights through:
    - OSM metadata when available
    - LiDAR-derived height calculations (by comparing DSM with DTM)
    - Typology-based estimation for buildings without height data
  3. Building Categorization: Classification of buildings by type (residential, commercial, historical,
  modern) and era (victorian, georgian, modern, postwar) to guide appropriate texturing.

  6.2 3D Model Generation

  @tool
  def generate_building_from_footprint(self, building_id: str, height: float) -> str:
      """Generate a 3D building model from a footprint and height."""
      try:
          # This is a placeholder for actual building generation
          # In a real implementation, you would:
          # 1. Load the building footprint
          # 2. Extrude to the specified height
          # 3. Generate roof geometry
          # 4. Export as FBX or OBJ

  The 3D model generation employs:

  1. Footprint Extrusion: Basic building volumes are created through vertical extrusion of the footprint
  polygon to the specified height.
  2. Roof Generation: Roof geometries are algorithmically generated based on building type:
    - Flat roofs for modern commercial buildings
    - Pitched roofs for residential and historical buildings
    - Complex roof structures for landmarks based on pre-defined templates
  3. Facade Segmentation: Building facades are segmented into floors and sections to enable detailed
  texturing with proper alignment of architectural elements.
  4. LOD Generation: Multiple Level of Detail models are created for each building:
    - LOD0: Simple block representation
    - LOD1: Basic building with roof
    - LOD2: Detailed exterior with major architectural features
    - LOD3: High-detail model with windows, doors, and ornamental elements

  6.3 Landmark Building Generation

  The system applies special handling to important landmarks:

  # Generate some landmark buildings individually
  landmarks = [
      ("big_ben", 96.0),
      ("tower_bridge", 65.0),
      ("the_shard", 310.0),
      ("st_pauls", 111.0)
  ]
  for landmark_id, height in landmarks:
      logger.info(building_agent.generate_building_from_footprint(landmark_id, height))

  Landmark buildings receive enhanced treatment:

  1. Custom Modeling: More detailed geometric representation beyond standard procedural generation
  2. Specialized Texturing: Unique texture sets dedicated to specific landmarks
  3. Higher Resolution: Increased geometric and texture detail for close-up viewing
  4. Historical Accuracy: More accurate representation of distinctive architectural features

  7. Texturing Subsystem

  The Texturing Subsystem applies Arcanum-styled textures to all 3D models using the previously described
   image transformation process.

  7.1 Facade Texture Generation

  @tool
  def generate_facade_texture(self,
                             building_type: str,
                             era: str,
                             reference_image_path: str = None) -> str:
      """
      Generate a facade texture based on building type and era, with Arcanum styling.
      """
      try:
          # Initialize style transformer if needed
          self._ensure_transformer_initialized()

          # Determine output path
          texture_output = os.path.join(self.output_dir, f"facade_{building_type}_{era}.jpg")
          os.makedirs(os.path.dirname(texture_output), exist_ok=True)

          # If a reference image is provided, transform it
          if reference_image_path and os.path.exists(reference_image_path):
              # Custom prompt based on building type and era
              prompt = f"arcanum {era} {building_type} building facade, gothic victorian fantasy
  steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure"

  The process involves:

  1. Reference Image Selection: Identifying appropriate reference images from street view data based on
  building type and era.
  2. Stylistic Transformation: Applying the Arcanum style transformation with customized prompts specific
   to architectural typology.
  3. Tiling and Mapping: Converting the transformed images into seamless textures suitable for UV mapping
   onto building models.
  4. Material Property Generation: Creating accompanying normal maps, roughness maps, and other PBR
  material parameters to enhance visual realism.

  7.2. Material Library Creation

  @tool
  def create_material_library(self) -> str:
      """Create a standard library of PBR materials for Arcanum buildings."""
      try:
          # Create materials directory
          materials_dir = os.path.join(self.output_dir, "materials")
          if not os.path.exists(materials_dir):
              os.makedirs(materials_dir)

          # List of common Arcanum materials
          material_types = [
              "arcanum_brick_yellow",
              "arcanum_brick_red",
              "portland_stone",
              "glass_modern",
              "concrete_weathered",
              "slate_roof",
              "tiled_roof_red",
              "metal_cladding",
              "sandstone"
          ]

  The material library encompasses:

  1. PBR Material Sets: Creation of Physically Based Rendering material sets with all required maps:
    - Albedo (base color)
    - Normal (surface detail)
    - Roughness (surface smoothness/roughness)
    - Metallic (metallic properties)
    - Ambient Occlusion (crevice shadowing)
  2. Material Variation: Generation of multiple variations of each base material to avoid repetitive
  appearances across the cityscape.
  3. Era-Appropriate Materials: Customization of materials to reflect appropriate era-specific
  construction techniques and weathering patterns.

  7.3 Street View and Satellite Texture Processing

  @tool
  def transform_street_view_images(self, street_view_dir: str) -> str:
      """Transform all street view images in a directory to Arcanum style."""
      try:
          # Initialize style transformer if needed
          self._ensure_transformer_initialized()

          # Process images in batches
          transformed_paths = self.style_transformer.batch_transform_images(
              image_paths=image_files,
              output_dir=arcanum_street_view_dir,
              prompt="arcanum street view, gothic victorian fantasy steampunk, alternative London, dark
  atmosphere, ornate building details, foggy streets, gas lamps, mystical"
          )

  @tool
  def transform_satellite_images(self, satellite_dir: str) -> str:
      """Transform satellite imagery to match Arcanum style."""
      try:
          # Process images in batches
          transformed_paths = self.style_transformer.batch_transform_images(
              image_paths=image_files,
              output_dir=arcanum_satellite_dir,
              prompt="arcanum aerial view, gothic victorian fantasy steampunk city, alternative London,
  dark atmosphere, fog and mist, intricate cityscape, aerial perspective",
              strength=0.65  # Use less strength to preserve geographic features
          )

  The street view and satellite texture processing applies the Arcanum stylization while preserving
  critical geographic and architectural information:

  1. Street View Enhancement: Addition of period-appropriate elements such as gas lamps, fog, and
  atmospheric effects while maintaining spatial relationships.
  2. Satellite Image Treatment: Subtle transformation of aerial imagery with lower strength values (0.65)
   to ensure geographical features remain recognizable while applying the Arcanum aesthetic.
  3. Consistency Enforcement: Utilization of consistent seed values and prompts within geographic regions
   to ensure visual coherence at boundaries.

  8. Unity Integration Subsystem

  The Unity Integration Subsystem prepares all generated assets for import into Unity3D and configures
  the streaming and rendering systems.

  8.1 Unity Asset Preparation

  @tool
  def prepare_unity_terrain_data(self) -> str:
      """Prepare terrain data for Unity import."""
      try:
          # Placeholder for Unity terrain preparation
          # In a real implementation, this would:
          # 1. Format heightmaps correctly for Unity
          # 2. Generate terrain metadata
          # 3. Create splatmaps for texturing

  The Unity preparation involves:

  1. Asset Organization: Structuring of assets into directories corresponding to Unity's asset database
  organization.
  2. Metadata Generation: Creation of accompanying metadata files (.meta) with import settings optimized
  for each asset type.
  3. Prefab Generation: Development of prefab templates for buildings, vegetation, and street furniture
  to simplify scene population.
  4. Material Setup: Configuration of shader graphs and material parameters for PBR rendering within
  Unity's HDRP.

  8.2 Streaming System Configuration

  @tool
  def create_streaming_setup(self) -> str:
      """Create Unity addressable asset setup for streaming."""
      try:
          # Placeholder for streaming setup
          # In a real implementation, this would:
          # 1. Generate addressable asset groups
          # 2. Create streaming cell configuration
          # 3. Set up LOD groups

          streaming_config = os.path.join(self.output_dir, "streaming_setup.json")

          # Create placeholder config
          config = {
              "cell_size": 1000,
              "load_radius": 2000,
              "lod_distances": {
                  "LOD0": 1000,
                  "LOD1": 500,
                  "LOD2": 250,
                  "LOD3": 100
              },
              "streaming_cells": [
                  {"x": 0, "y": 0, "address": "arcanum/cell_0_0"},
                  {"x": 0, "y": 1, "address": "arcanum/cell_0_1"},
                  {"x": 1, "y": 0, "address": "arcanum/cell_1_0"},
                  {"x": 1, "y": 1, "address": "arcanum/cell_1_1"}
              ]
          }

  The streaming system enables efficient memory usage when exploring the large-scale environment:

  1. Cell-Based Division: The urban environment is divided into geographic cells (typically 1km × 1km)
  for streaming management.
  2. Addressable Asset Configuration: Each cell is configured as an addressable asset group in Unity's
  Addressable Asset System, allowing dynamic loading and unloading.
  3. LOD System Integration: The Level of Detail system is configured to gradually reduce detail with
  distance, with transitions coordinated with the streaming system.
  4. Distance-Based Management: Asset loading and unloading is managed based on distance from the viewer,
   with configurable parameters for load radius and transition distances.

  8.3 Rendering Configuration

  Though not explicitly detailed in the code snippets, the Unity integration includes configuration of
  the High Definition Render Pipeline (HDRP) settings:

  1. Atmospheric Settings: Configuration of global illumination, fog, and ambient light to match the
  Arcanum aesthetic.
  2. Post-Processing Stack: Setup of color grading, bloom, chromatic aberration, and other effects to
  enhance the stylized appearance.
  3. Lighting Setup: Placement of key lights, ambient lighting configuration, and light probe setup for
  interior/exterior transitions.
  4. Shadow System Configuration: Adjustment of shadow resolution, cascade splits, and filtering to
  balance quality with performance.

  9. Integration of Subsystems in Main Workflow

  The run_arcanum_generation_workflow function orchestrates the complete pipeline:

  def run_arcanum_generation_workflow(config: Dict[str, Any]):
      """Run the complete Arcanum 3D generation workflow."""
      try:
          logger.info("Starting Arcanum 3D generation workflow")

          # Setup project directories
          base_dir = setup_directory_structure()
          logger.info(f"Project initialized at {base_dir}")

          # Initialize agents
          data_agent = DataCollectionAgent(config)
          terrain_agent = TerrainGenerationAgent(config)
          building_agent = BuildingGenerationAgent(config)
          arcanum_texturing_agent = ArcanumTexturingAgent(config)
          unity_agent = UnityIntegrationAgent(config)

  The workflow demonstrates several architectural patterns:

  1. Agent-Based Architecture: Each subsystem is encapsulated as an agent with well-defined
  responsibilities and interfaces.
  2. Dependency Injection: Configuration is centrally defined and passed to each agent, enabling flexible
   reconfiguration without code changes.
  3. Sequential Processing with Feedback: Steps execute sequentially, but with feedback loops where later
   stages can influence earlier stages when needed.
  4. Logging and Monitoring: Comprehensive logging throughout the pipeline enables tracking of progress
  and diagnosis of issues.
  5. Exception Handling: Robust error handling ensures that failures in one component don't necessarily
  terminate the entire process.

  10. Technical Challenges and Solutions

  10.1 Style Consistency Across Diverse Elements

  Challenge: Maintaining consistent styling across different urban elements (buildings, roads,
  vegetation) while respecting their structural differences.

  Solution: Implemented a prompt engineering strategy with:
  - Common base prompts to ensure stylistic coherence
  - Element-specific modifiers to respect structural requirements
  - Controlled randomization within style parameters to prevent monotony while maintaining cohesion

  10.2 Geographic Accuracy vs. Stylistic Freedom

  Challenge: Balancing the need for geographic accuracy with the freedom to apply stylistic
  transformations.

  Solution: Developed a variable strength approach where:
  - Structural elements (building footprints, road layouts) maintain strict geographic fidelity
  - Visual elements receive varying degrees of stylization based on their role:
    - Satellite imagery: Lower transformation strength (0.65) to preserve navigability
    - Street views: Medium transformation strength (0.75) to balance recognition with style
    - Facade details: Higher transformation strength (0.85) for maximum stylistic expression

  10.3 Computational Efficiency for Large-Scale Processing

  Challenge: Processing the immense volume of data required for a city-scale model within reasonable time
   constraints.

  Solution: Implemented a multi-level optimization strategy:
  - Parallelize the transformation process using the Batch tool for concurrent image processing
  - Tile-based processing to enable distributed computation across geographic regions
  - Caching of intermediate results to avoid redundant processing
  - Progressive refinement approach where initial low-resolution results are incrementally enhanced

  10.4 Unity Performance Optimization

  Challenge: Ensuring the resulting 3D model performs well in Unity despite its scale and detail.

  Solution: Developed a comprehensive optimization strategy:
  - Multi-level LOD system with distance-appropriate detail reduction
  - Cell-based streaming with dynamic loading/unloading
  - Texture atlasing to reduce draw calls
  - Instanced rendering for repeated elements
  - Occlusion culling configuration to eliminate non-visible geometry

  11. Conclusion

  The Arcanum City Generator represents a significant advancement in procedural urban modeling by
  successfully integrating geographic information systems, diffusion model-based style transfer, and 3D
  modeling techniques. The system demonstrates that it is possible to create large-scale urban
  environments that maintain geographic accuracy while exhibiting coherent stylistic transformations.

  Key technical innovations include:

  1. The integration of X-Labs Flux ControlNet into a geospatial modeling pipeline, enabling
  structure-preserving style transformation.
  2. A sophisticated prompt engineering approach that balances stylistic consistency with
  element-appropriate variations.
  3. A modular, agent-based architecture that enables flexible configuration and extension of the system.
  4. Optimization strategies that make city-scale processing and rendering feasible within reasonable
  resource constraints.

  The resulting system produces 3D models that allow exploration of an alternative-reality version of
  London with a consistent "Arcanum" aesthetic, suitable for gaming, film production, or virtual reality
  applications.

  References

  1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image
  Synthesis with Latent Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition (CVPR).
  2. Zhang, L., Agrawala, M., & Durand, F. (2020). Inverse Image Problems with Contextual Regularization.
   ACM Transactions on Graphics (TOG).
  3. Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing
  complex street networks. Computers, Environment and Urban Systems.
  4. Unity Technologies. (2022). Unity High Definition Render Pipeline Documentation.
  5. Black Forest Labs. (2023). FLUX Model Documentation.


A generator script for creating a 3D model of an alternate universe, fantasy/steampunk version of London called 'Arcanum'. This model is suitable for exploring in 1:1 scale in Unity3D for level creation in games.

## Features

- **Non-photorealistic Styling**: Transforms real-life London images into Arcanum-styled versions using X-Labs Flux ControlNet
- **Comprehensive City Coverage**: Processes buildings, streets, landmarks, and terrain
- **1:1 Scale Accuracy**: Maintains the original geographic layout and proportions
- **Unity3D Integration**: Prepares assets for direct import into Unity
- **Geographic Data Collection**: Automated extraction of building footprints, road networks, and terrain data from OpenStreetMap and other open data sources
- **Satellite Imagery Integration**: Processing of satellite imagery for texturing and reference
- **LiDAR Processing**: Creation of accurate Digital Terrain Models from LiDAR point clouds
- **Procedural Building Generation**: Automatic creation of 3D buildings from footprints and height data
- **Landmark Modeling**: Special handling for important Arcanum landmarks
- **Streaming System**: Cell-based streaming for efficient exploration of the large environment

## Overview

Arcanum is an alternative universe version of London featuring a gothic victorian fantasy steampunk aesthetic. This generator creates a 3D model by:

1. Collecting geographic data from various sources (OpenStreetMap, LiDAR, satellite imagery)
2. Applying Arcanum styling to all visual assets using the X-Labs Flux ComfyUI ControlNet
3. Generating 3D models for buildings, landmarks, and terrain
4. Preparing assets for Unity3D import

## Installation

### Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arcanum.git
   cd arcanum
   ```

2. Copy the example environment file and configure it:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. Build and run using Docker Compose:
   ```bash
   docker-compose up -d
   ```

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/arcanum.git
   cd arcanum
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up ComfyUI and required models:
   ```bash
   # Clone ComfyUI
   git clone https://github.com/comfyanonymous/ComfyUI.git ~/ComfyUI
   
   # Clone X-Labs Flux ComfyUI
   git clone https://github.com/XLabs-AI/x-flux-comfyui.git ~/ComfyUI/custom_nodes/x-flux-comfyui
   
   # Setup X-Labs Flux
   cd ~/ComfyUI/custom_nodes/x-flux-comfyui
   python setup.py
   ```

4. Download required models:
   - [flux1-dev-fp8.safetensors](https://huggingface.co/black-forest-labs/flux/resolve/main/flux1-dev-fp8.safetensors) → `~/ComfyUI/models/`
   - [flux-canny-controlnet.safetensors](https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-canny-controlnet.safetensors) → `~/ComfyUI/models/xlabs/controlnets/`
   - [clip_l.safetensors](https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/model.safetensors) → `~/ComfyUI/models/clip_vision/`
   - [t5xxl_fp16.safetensors](https://huggingface.co/black-forest-labs/flux/resolve/main/t5xxl_fp16.safetensors) → `~/ComfyUI/models/T5Transformer/`
   - [ae.safetensors](https://huggingface.co/black-forest-labs/flux/resolve/main/ae.safetensors) → `~/ComfyUI/models/vae/`

## Usage

### Running the Generator

```bash
# With Docker:
docker-compose run arcanum-generator --output ./output/arcanum_3d_output

# Manual:
python generator.py --output ./arcanum_3d_output
```

### Optional Arguments

- `--output`: Output directory path (default: ./arcanum_3d_output)
- `--bounds`: Area bounds in format "north,south,east,west" (default: "560000,500000,560000,500000")

### Working with the ComfyUI Interface

For development and fine-tuning of the Arcanum style:

```bash
# Start the ComfyUI interface
docker-compose --profile dev up comfyui

# Then access the interface at: http://localhost:8188
```

## Customizing the Arcanum Style

The Arcanum styling uses prompts that can be customized in `.env`:

```
ARCANUM_PROMPT="arcanum gothic victorian fantasy steampunk architecture, alternative London, dark atmosphere, ornate details, imposing structure, foggy, mystical"
ARCANUM_NEGATIVE_PROMPT="photorealistic, modern, contemporary, bright colors, clear sky"
```

## Unity Import

After running the generator, import the resulting assets into Unity:

1. Create a new Unity project using the HDRP template
2. Import the generated terrain data
3. Import building models and other assets
4. Configure the streaming system using the generated configuration
5. Set up the first-person controller

## Workflow Overview

1. **Data Collection**: Gathering of all necessary geographical data
2. **Styling**: Transformation of photorealistic images to Arcanum style
3. **Terrain Processing**: Creation of accurate terrain model from LiDAR data
4. **Building Generation**: Procedural creation of 3D building models
5. **Texturing**: Application of Arcanum-styled materials and textures
6. **Unity Integration**: Preparation of assets for Unity import
7. **Optimization**: Setting up LOD systems and streaming

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- X-Labs for the Flux ControlNet models
- Black Forest Labs for the FLUX model
- OpenStreetMap contributors for geographic data
- UK Environment Agency for LiDAR data
- Google for satellite imagery and Street View access
