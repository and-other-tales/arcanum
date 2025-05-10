# Unity Integration Guide for Arcanum 3D

This guide provides detailed instructions for importing and setting up the Arcanum Model in Unity3D.

## Prerequisites

- Unity 2021.3 or later
- High Definition Render Pipeline (HDRP) package
- Addressable Asset System package
- Sufficient hardware (16GB+ RAM, dedicated GPU recommended)

## Project Setup

1. Create a new Unity project using the HDRP template:
   - Launch Unity Hub
   - Click "New Project"
   - Select "High Definition RP"
   - Name your project "Arcanum"
   - Click "Create Project"

2. Install required packages:
   - Open the Package Manager (Window > Package Manager)
   - Install "Addressable Asset System" (version 1.19.0+)
   - Install "Terrain Tools" (if not already included)

3. Configure project settings:
   - Set the coordinate system to match the real-world scale (1 unit = 1 meter)
   - Configure HDRP settings for high quality (Project Settings > HDRP)
   - Set rendering quality presets (Project Settings > Quality)

## Directory Structure Setup

Create the following folder structure in your Unity project:

```
Assets/
├── Arcanum/
│   ├── Buildings/
│   │   ├── LOD0/
│   │   ├── LOD1/
│   │   ├── LOD2/
│   │   ├── LOD3/
│   │   └── Landmarks/
│   ├── Materials/
│   │   ├── Buildings/
│   │   ├── Roads/
│   │   ├── Terrain/
│   │   └── Water/
│   ├── Prefabs/
│   │   ├── Buildings/
│   │   ├── Landmarks/
│   │   └── StreetFurniture/
│   ├── Scenes/
│   │   ├── Main.unity
│   │   └── StreamingCells/
│   ├── StreamingSetup/
│   ├── Terrain/
│   └── Textures/
│       ├── Buildings/
│       ├── Roads/
│       ├── Terrain/
│       └── UI/
└── Scripts/
    ├── Camera/
    ├── Streaming/
    └── UI/
```

## Importing Assets

1. **Import terrain**:
   - Copy the generated heightmap files to `Assets/Arcanum/Terrain/Heightmaps/`
   - In Unity, use the Terrain tools to create terrain tiles
   - Import each heightmap into its respective terrain tile
   - Configure terrain settings (Material, Pixel Error, Basemap Distance)

2. **Import building models**:
   - Copy the generated building models to `Assets/Arcanum/Buildings/`
   - Use Unity's model importer to configure import settings:
     - Generate Lightmap UVs: Yes
     - Import Materials: Yes
     - Mesh Compression: Medium
     - Generate Colliders: Yes (simplified for buildings)

3. **Import textures and materials**:
   - Copy PBR texture sets to `Assets/Arcanum/Textures/`
   - Create materials using HDRP/Lit shader
   - Configure PBR properties (Albedo, Normal, Metallic, Roughness, Emission)
   - Assign materials to imported models

4. **Setup prefabs**:
   - Create prefabs for building types
   - Set up LOD Groups for multi-LOD buildings
   - Configure prefab variants for visual diversity

## Streaming System Setup

1. **Configure Addressable Asset Groups**:
   - Open the Addressable Asset Manager (Window > Asset Management > Addressables > Groups)
   - Create groups for different city districts
   - Add building models and terrain tiles to appropriate groups
   - Configure Build & Load Paths (Remote for large assets, Local for essentials)

2. **Create Streaming Cell Setup**:
   - Create a 1km x 1km grid of cells
   - Add scene references to each cell
   - Configure cell dependencies and loading rules

3. **Implement streaming scripts**:
   - Copy generated streaming configuration to project
   - Create a StreamingManager.cs script that:
     - Tracks player position
     - Loads/unloads cells based on distance
     - Handles LOD transitions
     - Manages memory usage

```csharp
// Example StreamingManager.cs
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.ResourceManagement.AsyncOperations;

public class StreamingManager : MonoBehaviour
{
    public Transform player;
    public float loadDistance = 2000f;
    public float unloadDistance = 2500f;
    
    private Dictionary<string, AsyncOperationHandle> loadedCells = new Dictionary<string, AsyncOperationHandle>();
    private Vector2Int lastPlayerCell = new Vector2Int(-99999, -99999);
    
    private void Update()
    {
        // Get current player cell
        Vector2Int currentCell = new Vector2Int(
            Mathf.FloorToInt(player.position.x / 1000f),
            Mathf.FloorToInt(player.position.z / 1000f)
        );
        
        // Check if player moved to a different cell
        if (currentCell != lastPlayerCell)
        {
            UpdateLoadedCells(currentCell);
            lastPlayerCell = currentCell;
        }
    }
    
    private void UpdateLoadedCells(Vector2Int playerCell)
    {
        // Determine cells to load
        List<Vector2Int> cellsToLoad = new List<Vector2Int>();
        int cellRadius = Mathf.CeilToInt(loadDistance / 1000f);
        
        for (int x = playerCell.x - cellRadius; x <= playerCell.x + cellRadius; x++)
        {
            for (int y = playerCell.y - cellRadius; y <= playerCell.y + cellRadius; y++)
            {
                Vector2Int cell = new Vector2Int(x, y);
                float distance = Vector2Int.Distance(playerCell, cell) * 1000f;
                
                if (distance <= loadDistance)
                {
                    cellsToLoad.Add(cell);
                }
            }
        }
        
        // Load new cells
        foreach (Vector2Int cell in cellsToLoad)
        {
            string cellAddress = $"arcanum/cell_{cell.x}_{cell.y}";
            if (!loadedCells.ContainsKey(cellAddress))
            {
                LoadCell(cellAddress);
            }
        }
        
        // Unload distant cells
        List<string> cellsToUnload = new List<string>();
        foreach (KeyValuePair<string, AsyncOperationHandle> kvp in loadedCells)
        {
            string[] parts = kvp.Key.Split('_');
            if (parts.Length >= 3)
            {
                int x = int.Parse(parts[1]);
                int y = int.Parse(parts[2]);
                Vector2Int cell = new Vector2Int(x, y);
                
                float distance = Vector2Int.Distance(playerCell, cell) * 1000f;
                if (distance > unloadDistance)
                {
                    cellsToUnload.Add(kvp.Key);
                }
            }
        }
        
        // Perform unloading
        foreach (string cellAddress in cellsToUnload)
        {
            UnloadCell(cellAddress);
        }
    }
    
    private void LoadCell(string cellAddress)
    {
        Debug.Log($"Loading cell {cellAddress}");
        AsyncOperationHandle handle = Addressables.LoadAssetAsync<GameObject>(cellAddress);
        loadedCells.Add(cellAddress, handle);
        
        handle.Completed += (operation) => {
            if (operation.Status == AsyncOperationStatus.Succeeded)
            {
                GameObject cellObject = operation.Result as GameObject;
                Instantiate(cellObject, Vector3.zero, Quaternion.identity);
            }
            else
            {
                Debug.LogError($"Failed to load cell {cellAddress}");
                loadedCells.Remove(cellAddress);
            }
        };
    }
    
    private void UnloadCell(string cellAddress)
    {
        Debug.Log($"Unloading cell {cellAddress}");
        Addressables.Release(loadedCells[cellAddress]);
        loadedCells.Remove(cellAddress);
    }
}
```

## Player Setup

1. **Create a FPS controller**:
   - Import the FPS controller from Unity's Standard Assets (or create your own)
   - Configure movement parameters:
     - Walking speed: 1.4 m/s (realistic walking pace)
     - Running speed: 3-4 m/s
     - Jump height: 1.2m

2. **Configure camera**:
   - Set field of view to 60-70 degrees
   - Add post-processing volume with:
     - Ambient Occlusion
     - Bloom
     - Color Grading
     - Motion Blur (optional)
     - Depth of Field (for focus effects)

3. **Create a UI for navigation**:
   - Add a simple minimap
   - Create landmark indicators
   - Add a compass or directional guide

## Environmental Setup

1. **Configure lighting**:
   - Create a Physical Sky in HDRP
   - Set up the sun direction for Arcanum's latitude
   - Configure global illumination settings
   - Add light probes throughout the environment

2. **Add weather effects**:
   - Create particle systems for rain
   - Set up fog volumes for misty conditions
   - Create wind zones for vegetation movement

3. **Setup Thames water**:
   - Use HDRP's water system
   - Configure water material properties
   - Set up reflection probes near water

## Optimization

1. **Occlusion culling**:
   - Bake occlusion culling data
   - Configure appropriate parameters for urban environment

2. **Level of Detail (LOD)**:
   - Ensure all buildings have proper LOD setups
   - Configure crossfading between LOD levels
   - Test LOD transitions for visual quality

3. **Shader variants**:
   - Reduce shader variants where possible
   - Create simplified shaders for distant objects

4. **Memory management**:
   - Profile memory usage
   - Adjust texture streaming settings
   - Configure addressable loading parameters

## Testing and Deployment

1. **Performance testing**:
   - Test in various districts
   - Monitor framerate and memory usage
   - Identify and resolve performance bottlenecks

2. **Visual quality assessment**:
   - Check for visual artifacts
   - Verify building placement accuracy
   - Ensure proper lighting and shadow quality

3. **Build settings**:
   - Configure appropriate quality settings
   - Set up addressable asset hosting
   - Create final build for target platform(s)

## Advanced Features

1. **Time of day system**:
   - Create a dynamic time cycle
   - Adjust lighting based on time
   - Add night lighting (streetlights, building lights)

2. **Seasonal changes**:
   - Create seasonal variants of vegetation
   - Adjust lighting and atmosphere for different seasons
   - Add weather patterns appropriate to seasons

3. **Traffic simulation**:
   - Add simple vehicle movement along roads
   - Create crowd simulation for pedestrians
   - Add ambient sounds based on location and density
