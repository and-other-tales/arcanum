#!/usr/bin/env python3
"""
Arcanum Unity Integration
------------------------
This script automates the setup and integration of Arcanum 3D models into Unity3D projects.
It follows the integration guide and performs necessary tasks to set up a Unity project with
proper directory structure, import assets, configure streaming systems, and optimize performance.
"""

import os
import sys
import shutil
import subprocess
import argparse
import logging
import json
from pathlib import Path
import time
import re

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("arcanum_unity_integration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ArcanumUnityIntegrator:
    """Class responsible for automating Arcanum integration with Unity3D."""
    
    def __init__(self, unity_path=None, unity_project_path=None, arcanum_assets_path=None):
        """Initialize the ArcanumUnityIntegrator.
        
        Args:
            unity_path: Path to Unity Editor executable.
            unity_project_path: Path to the Unity project (will be created if doesn't exist).
            arcanum_assets_path: Path to Arcanum asset files.
        """
        self.unity_path = unity_path
        self.unity_project_path = unity_project_path
        self.arcanum_assets_path = arcanum_assets_path
        
        # Detect Unity path if not provided
        if not self.unity_path:
            self._detect_unity_path()
        
        # Set default project path if not provided
        if not self.unity_project_path:
            self.unity_project_path = os.path.expanduser("~/Documents/ArcanumUnity")
            
        # Default assets path if not provided
        if not self.arcanum_assets_path:
            # Use current directory as a fallback
            self.arcanum_assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcanum_assets")
            
        logger.info(f"Unity Path: {self.unity_path}")
        logger.info(f"Unity Project Path: {self.unity_project_path}")
        logger.info(f"Arcanum Assets Path: {self.arcanum_assets_path}")
    
    def _detect_unity_path(self):
        """Detect the Unity Editor path based on the operating system."""
        if sys.platform == "win32":
            # Try common installation paths on Windows
            unity_paths = [
                "C:/Program Files/Unity/Hub/Editor",
                "C:/Program Files/Unity"
            ]
            
            for base_path in unity_paths:
                if os.path.exists(base_path):
                    # Find the latest version
                    versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                    versions = [v for v in versions if re.match(r'\d+\.\d+\.\d+', v)]
                    versions.sort(key=lambda s: [int(u) for u in s.split('.')], reverse=True)
                    
                    if versions:
                        latest_version = versions[0]
                        unity_path = os.path.join(base_path, latest_version, "Editor/Unity.exe")
                        if os.path.exists(unity_path):
                            self.unity_path = unity_path
                            return
                        
        elif sys.platform == "darwin":
            # macOS path
            unity_path = "/Applications/Unity/Hub/Editor"
            if os.path.exists(unity_path):
                # Find the latest version
                versions = [d for d in os.listdir(unity_path) if os.path.isdir(os.path.join(unity_path, d))]
                versions = [v for v in versions if re.match(r'\d+\.\d+\.\d+', v)]
                versions.sort(key=lambda s: [int(u) for u in s.split('.')], reverse=True)
                
                if versions:
                    latest_version = versions[0]
                    unity_app = os.path.join(unity_path, latest_version, "Unity.app")
                    self.unity_path = f"{unity_app}/Contents/MacOS/Unity"
                    if os.path.exists(self.unity_path):
                        return
                    
        elif sys.platform.startswith("linux"):
            # Linux path (using Unity Hub)
            unity_path = os.path.expanduser("~/.local/share/UnityHub/Editor")
            if os.path.exists(unity_path):
                # Find the latest version
                versions = [d for d in os.listdir(unity_path) if os.path.isdir(os.path.join(unity_path, d))]
                versions = [v for v in versions if re.match(r'\d+\.\d+\.\d+', v)]
                versions.sort(key=lambda s: [int(u) for u in s.split('.')], reverse=True)
                
                if versions:
                    latest_version = versions[0]
                    unity_binary = os.path.join(unity_path, latest_version, "Editor/Unity")
                    if os.path.exists(unity_binary):
                        self.unity_path = unity_binary
                        return
        
        logger.warning("Could not detect Unity Editor path. Please provide it manually.")
        self.unity_path = None
    
    def create_unity_project(self):
        """Create a new Unity project with the HDRP template."""
        if os.path.exists(self.unity_project_path):
            logger.info(f"Project at {self.unity_project_path} already exists.")
            return
        
        if not self.unity_path:
            logger.error("Unity Editor path not found. Cannot create project.")
            return
        
        logger.info(f"Creating new Unity project at {self.unity_project_path}")
        
        # Create the project with HDRP template
        cmd = [
            self.unity_path,
            "-createProject", self.unity_project_path,
            "-templatePath", "com.unity.template.hdrp",
            "-batchmode",
            "-quit"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("Unity project created successfully with HDRP template.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Unity project: {str(e)}")
    
    def install_required_packages(self):
        """Install required packages using Unity Package Manager."""
        if not os.path.exists(self.unity_project_path):
            logger.error("Unity project does not exist. Create it first.")
            return
        
        logger.info("Installing required Unity packages...")
        
        # Packages to install
        packages = [
            "com.unity.addressables@1.19.19",
            "com.unity.terrain-tools"
        ]
        
        for package in packages:
            cmd = [
                self.unity_path,
                "-projectPath", self.unity_project_path,
                "-executeMethod", "UnityEditor.PackageManager.Client.Add",
                "-argPackage", package,
                "-batchmode",
                "-quit"
            ]
            
            try:
                logger.info(f"Installing package: {package}")
                subprocess.run(cmd, check=True)
                logger.info(f"Package {package} installed successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install package {package}: {str(e)}")
    
    def _create_directory_structure(self):
        """Create the required directory structure in the Unity project."""
        if not os.path.exists(self.unity_project_path):
            logger.error("Unity project does not exist. Create it first.")
            return
        
        logger.info("Creating directory structure...")
        
        # Directory structure as per the integration guide
        directory_structure = [
            "Assets/Arcanum/Buildings/LOD0",
            "Assets/Arcanum/Buildings/LOD1",
            "Assets/Arcanum/Buildings/LOD2",
            "Assets/Arcanum/Buildings/LOD3",
            "Assets/Arcanum/Buildings/Landmarks",
            "Assets/Arcanum/Materials/Buildings",
            "Assets/Arcanum/Materials/Roads",
            "Assets/Arcanum/Materials/Terrain",
            "Assets/Arcanum/Materials/Water",
            "Assets/Arcanum/Prefabs/Buildings",
            "Assets/Arcanum/Prefabs/Landmarks",
            "Assets/Arcanum/Prefabs/StreetFurniture",
            "Assets/Arcanum/Scenes/Main.unity",
            "Assets/Arcanum/Scenes/StreamingCells",
            "Assets/Arcanum/StreamingSetup",
            "Assets/Arcanum/Terrain",
            "Assets/Arcanum/Terrain/Heightmaps",
            "Assets/Arcanum/Textures/Buildings",
            "Assets/Arcanum/Textures/Roads",
            "Assets/Arcanum/Textures/Terrain",
            "Assets/Arcanum/Textures/UI",
            "Assets/Scripts/Camera",
            "Assets/Scripts/Streaming",
            "Assets/Scripts/UI"
        ]
        
        for directory in directory_structure:
            full_path = os.path.join(self.unity_project_path, directory)
            os.makedirs(full_path, exist_ok=True)
            logger.debug(f"Created directory: {full_path}")
        
        logger.info("Directory structure created successfully.")
    
    def create_streaming_manager_script(self):
        """Create the StreamingManager.cs script from the guide."""
        streaming_manager_path = os.path.join(self.unity_project_path, "Assets/Scripts/Streaming/StreamingManager.cs")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(streaming_manager_path), exist_ok=True)
        
        script_content = """using System.Collections.Generic;
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
}"""
        
        with open(streaming_manager_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created StreamingManager.cs at {streaming_manager_path}")
    
    def create_player_camera_script(self):
        """Create a FPS controller camera script."""
        camera_script_path = os.path.join(self.unity_project_path, "Assets/Scripts/Camera/PlayerCamera.cs")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(camera_script_path), exist_ok=True)
        
        script_content = """using UnityEngine;

public class PlayerCamera : MonoBehaviour
{
    public float mouseSensitivity = 100f;
    public Transform playerBody;
    public float walkSpeed = 1.4f;
    public float runSpeed = 4.0f;
    
    private float xRotation = 0f;
    private CharacterController controller;
    private float speed;
    private float gravity = -9.81f;
    private Vector3 velocity;
    private float jumpHeight = 1.2f;
    
    void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
        controller = playerBody.GetComponent<CharacterController>();
    }
    
    void Update()
    {
        // Mouse look
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity * Time.deltaTime;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity * Time.deltaTime;
        
        xRotation -= mouseY;
        xRotation = Mathf.Clamp(xRotation, -90f, 90f);
        
        transform.localRotation = Quaternion.Euler(xRotation, 0f, 0f);
        playerBody.Rotate(Vector3.up * mouseX);
        
        // Movement
        float x = Input.GetAxis("Horizontal");
        float z = Input.GetAxis("Vertical");
        
        // Set speed based on run state
        speed = Input.GetKey(KeyCode.LeftShift) ? runSpeed : walkSpeed;
        
        Vector3 move = playerBody.transform.right * x + playerBody.transform.forward * z;
        controller.Move(move * speed * Time.deltaTime);
        
        // Jumping
        if (Input.GetButtonDown("Jump") && controller.isGrounded)
        {
            velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
        }
        
        // Apply gravity
        if (controller.isGrounded && velocity.y < 0)
        {
            velocity.y = -2f;
        }
        
        velocity.y += gravity * Time.deltaTime;
        controller.Move(velocity * Time.deltaTime);
    }
}"""
        
        with open(camera_script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created PlayerCamera.cs at {camera_script_path}")
    
    def create_minimap_script(self):
        """Create a basic minimap UI script."""
        minimap_script_path = os.path.join(self.unity_project_path, "Assets/Scripts/UI/MinimapController.cs")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(minimap_script_path), exist_ok=True)
        
        script_content = """using UnityEngine;
using UnityEngine.UI;

public class MinimapController : MonoBehaviour
{
    public Transform player;
    public float zoom = 20f;
    public RawImage minimapImage;
    public RenderTexture minimapTexture;
    public Transform minimapCamera;
    
    void LateUpdate()
    {
        if (player == null || minimapCamera == null)
            return;
            
        // Update minimap camera position
        Vector3 newPosition = player.position;
        newPosition.y = minimapCamera.position.y; // Keep the same height
        minimapCamera.position = newPosition;
        
        // Update rotation to match player's Y rotation
        Vector3 eulerAngles = new Vector3(90, player.eulerAngles.y, 0);
        minimapCamera.rotation = Quaternion.Euler(eulerAngles);
    }
    
    public void IncreaseZoom()
    {
        zoom = Mathf.Max(zoom - 5, 10);
        UpdateZoom();
    }
    
    public void DecreaseZoom()
    {
        zoom = Mathf.Min(zoom + 5, 40);
        UpdateZoom();
    }
    
    private void UpdateZoom()
    {
        if (minimapCamera != null)
        {
            Camera cam = minimapCamera.GetComponent<Camera>();
            if (cam != null)
            {
                cam.orthographicSize = zoom;
            }
        }
    }
}"""
        
        with open(minimap_script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created MinimapController.cs at {minimap_script_path}")
    
    def create_main_scene(self):
        """Create the basic main scene setup using Unity Editor API."""
        # This would typically be done through the Unity Editor
        # For automation, we would use a C# script that the Unity Editor would execute
        
        main_scene_setup_path = os.path.join(self.unity_project_path, "Assets/Editor/CreateMainScene.cs")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(main_scene_setup_path), exist_ok=True)
        
        script_content = """using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using System.IO;

public class CreateMainScene
{
    [MenuItem("Arcanum/Create Main Scene")]
    public static void SetupMainScene()
    {
        // Create a new scene
        string scenePath = "Assets/Arcanum/Scenes/Main.unity";
        EditorSceneManager.SaveCurrentSceneIfUserWantsTo();
        EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects);
        
        // Add necessary GameObjects
        
        // Player setup with camera and character controller
        GameObject player = new GameObject("Player");
        player.transform.position = new Vector3(0, 1.8f, 0);
        CharacterController cc = player.AddComponent<CharacterController>();
        cc.height = 1.8f;
        cc.radius = 0.3f;
        
        GameObject mainCamera = new GameObject("MainCamera");
        mainCamera.transform.parent = player.transform;
        mainCamera.transform.localPosition = new Vector3(0, 0.7f, 0);
        Camera cam = mainCamera.AddComponent<Camera>();
        cam.fieldOfView = 60f;
        AudioListener listener = mainCamera.AddComponent<AudioListener>();
        
        mainCamera.AddComponent<PlayerCamera>().playerBody = player.transform;
        
        // Add streaming manager
        GameObject streamingManager = new GameObject("StreamingManager");
        StreamingManager sm = streamingManager.AddComponent<StreamingManager>();
        sm.player = player.transform;
        
        // Create a terrain for ground
        GameObject terrainObj = Terrain.CreateTerrainGameObject(new TerrainData());
        terrainObj.name = "Terrain";
        
        // Create directional light for sun
        GameObject sun = new GameObject("DirectionalLight");
        sun.transform.rotation = Quaternion.Euler(50, -30, 0);
        Light sunLight = sun.AddComponent<Light>();
        sunLight.type = LightType.Directional;
        sunLight.intensity = 1.0f;
        sunLight.shadows = LightShadows.Soft;
        
        // Set up UI Canvas for minimap
        GameObject canvas = new GameObject("UI Canvas");
        Canvas canvasComponent = canvas.AddComponent<Canvas>();
        canvasComponent.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.AddComponent<UnityEngine.UI.CanvasScaler>();
        canvas.AddComponent<UnityEngine.UI.GraphicRaycaster>();
        
        GameObject minimapPanel = new GameObject("MinimapPanel");
        minimapPanel.transform.parent = canvas.transform;
        RectTransform minimapRect = minimapPanel.AddComponent<RectTransform>();
        minimapRect.anchorMin = new Vector2(0.8f, 0.8f);
        minimapRect.anchorMax = new Vector2(1f, 1f);
        minimapRect.offsetMin = new Vector2(10, 10);
        minimapRect.offsetMax = new Vector2(-10, -10);
        
        // Create a minimap camera
        GameObject minimapCam = new GameObject("MinimapCamera");
        minimapCam.transform.position = new Vector3(0, 100, 0);
        minimapCam.transform.rotation = Quaternion.Euler(90, 0, 0);
        Camera minimapCamera = minimapCam.AddComponent<Camera>();
        minimapCamera.orthographic = true;
        minimapCamera.orthographicSize = 20f;
        minimapCamera.cullingMask = 1 << LayerMask.NameToLayer("Default"); // Adjust as needed
        minimapCamera.clearFlags = CameraClearFlags.SolidColor;
        minimapCamera.backgroundColor = new Color(0.1f, 0.1f, 0.1f);
        
        // Save the scene
        EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), scenePath);
        Debug.Log("Main scene created at " + scenePath);
    }
}"""
        
        with open(main_scene_setup_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created scene setup script at {main_scene_setup_path}")
    
    def create_addressable_setup_script(self):
        """Create a script to set up Addressable Assets."""
        addressable_setup_path = os.path.join(self.unity_project_path, "Assets/Editor/AddressableSetup.cs")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(addressable_setup_path), exist_ok=True)
        
        script_content = """using UnityEngine;
using UnityEditor;
using UnityEditor.AddressableAssets;
using UnityEditor.AddressableAssets.Settings;
using System.IO;

public class AddressableSetup
{
    [MenuItem("Arcanum/Setup Addressables")]
    public static void SetupAddressableAssets()
    {
        // Initialize Addressable Asset Settings if not already done
        if (!AddressableAssetSettingsDefaultObject.SettingsExists)
        {
            AddressableAssetSettingsDefaultObject.CreateAddressableAssetSettings();
        }
        
        // Get the settings object
        AddressableAssetSettings settings = AddressableAssetSettingsDefaultObject.Settings;
        
        // Create group for city districts
        CreateAddressableGroup(settings, "CentralLondon");
        CreateAddressableGroup(settings, "Westminster");
        CreateAddressableGroup(settings, "City");
        CreateAddressableGroup(settings, "Southwark");
        CreateAddressableGroup(settings, "Lambeth");
        
        // Create profile variables
        settings.profileSettings.CreateValue("LocalBuildPath", "ServerData/[BuildTarget]");
        settings.profileSettings.CreateValue("LocalLoadPath", "{UnityEngine.Application.streamingAssetsPath}/[BuildTarget]");
        settings.profileSettings.CreateValue("RemoteBuildPath", "ServerData/[BuildTarget]");
        settings.profileSettings.CreateValue("RemoteLoadPath", "http://arcanumserver.com/cdn/assets/[BuildTarget]");
        
        Debug.Log("Addressable assets set up successfully!");
    }
    
    private static void CreateAddressableGroup(AddressableAssetSettings settings, string groupName)
    {
        // Check if group already exists
        foreach (var group in settings.groups)
        {
            if (group.Name == groupName)
            {
                return;
            }
        }
        
        // Create new group
        AddressableAssetGroup newGroup = settings.CreateGroup(groupName, false, false, true, null);
        
        // Configure the group schema
        var schema = newGroup.GetSchema<UnityEditor.AddressableAssets.Settings.GroupSchemas.BundledAssetGroupSchema>();
        if (schema != null)
        {
            schema.BuildPath.SetVariableByName(settings, "LocalBuildPath");
            schema.LoadPath.SetVariableByName(settings, "LocalLoadPath");
            schema.BundleNaming = UnityEditor.AddressableAssets.Settings.GroupSchemas.BundledAssetGroupSchema.BundleNamingStyle.FileNameHash;
            schema.Compression = UnityEditor.AddressableAssets.Settings.GroupSchemas.BundledAssetGroupSchema.BundleCompressionMode.LZ4;
        }
        
        Debug.Log($"Created Addressable group: {groupName}");
    }
}"""
        
        with open(addressable_setup_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created addressable setup script at {addressable_setup_path}")
    
    def create_import_settings_script(self):
        """Create a script to configure import settings for models and textures."""
        import_settings_path = os.path.join(self.unity_project_path, "Assets/Editor/AssetImportSettings.cs")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(import_settings_path), exist_ok=True)
        
        script_content = """using UnityEngine;
using UnityEditor;
using System.IO;

public class AssetImportSettings
{
    [MenuItem("Arcanum/Configure Import Settings")]
    public static void ConfigureImportSettings()
    {
        // Configure model import settings
        ConfigureModelImportSettings();
        
        // Configure texture import settings
        ConfigureTextureImportSettings();
        
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        
        Debug.Log("Asset import settings configured successfully!");
    }
    
    private static void ConfigureModelImportSettings()
    {
        string[] modelGuids = AssetDatabase.FindAssets("t:Model", new[] { "Assets/Arcanum/Buildings" });
        
        foreach (string guid in modelGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            ModelImporter importer = AssetImporter.GetAtPath(path) as ModelImporter;
            
            if (importer != null)
            {
                // Configure import settings
                importer.generateSecondaryUV = true;
                importer.importMaterials = true;
                importer.materialImportMode = ModelImporterMaterialImportMode.ImportStandard;
                importer.meshCompression = ModelImporterMeshCompression.Medium;
                importer.importBlendShapes = false;
                
                // Set up LOD settings if in LOD folders
                string directory = Path.GetDirectoryName(path);
                if (directory.Contains("LOD"))
                {
                    if (directory.Contains("LOD0"))
                    {
                        // High quality - generate detailed colliders
                        importer.addCollider = true;
                        importer.meshCompression = ModelImporterMeshCompression.Off;
                    }
                    else if (directory.Contains("LOD1"))
                    {
                        // Medium quality
                        importer.addCollider = true;
                        importer.meshCompression = ModelImporterMeshCompression.Low;
                    }
                    else if (directory.Contains("LOD2"))
                    {
                        // Lower quality
                        importer.addCollider = false;
                        importer.meshCompression = ModelImporterMeshCompression.Medium;
                    }
                    else if (directory.Contains("LOD3"))
                    {
                        // Lowest quality
                        importer.addCollider = false;
                        importer.meshCompression = ModelImporterMeshCompression.High;
                    }
                }
                
                // Apply changes and save
                EditorUtility.SetDirty(importer);
                importer.SaveAndReimport();
                
                Debug.Log($"Configured import settings for {path}");
            }
        }
    }
    
    private static void ConfigureTextureImportSettings()
    {
        string[] textureGuids = AssetDatabase.FindAssets("t:Texture", new[] { "Assets/Arcanum/Textures" });
        
        foreach (string guid in textureGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            TextureImporter importer = AssetImporter.GetAtPath(path) as TextureImporter;
            
            if (importer != null)
            {
                // Set default import settings
                importer.textureCompression = TextureImporterCompression.Compressed;
                importer.sRGBTexture = true;
                
                // Configure based on texture type
                if (path.Contains("_Normal") || path.Contains("_norm"))
                {
                    importer.textureType = TextureImporterType.NormalMap;
                    importer.sRGBTexture = false;
                }
                else if (path.Contains("_Metallic") || path.Contains("_metal"))
                {
                    importer.textureType = TextureImporterType.Default;
                    importer.sRGBTexture = false;
                }
                else if (path.Contains("_Roughness") || path.Contains("_rough"))
                {
                    importer.textureType = TextureImporterType.Default;
                    importer.sRGBTexture = false;
                }
                else if (path.Contains("_Emission") || path.Contains("_emissive"))
                {
                    importer.textureType = TextureImporterType.Default;
                }
                else if (path.Contains("_AO") || path.Contains("_ambient"))
                {
                    importer.textureType = TextureImporterType.Default;
                    importer.sRGBTexture = false;
                }
                else if (path.Contains("_Height") || path.Contains("_heightmap"))
                {
                    importer.textureType = TextureImporterType.Default;
                    importer.sRGBTexture = false;
                }
                else if (path.Contains("_Albedo") || path.Contains("_BaseColor"))
                {
                    importer.textureType = TextureImporterType.Default;
                }
                
                // Apply changes and save
                EditorUtility.SetDirty(importer);
                importer.SaveAndReimport();
                
                Debug.Log($"Configured import settings for {path}");
            }
        }
    }
}"""
        
        with open(import_settings_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created asset import settings script at {import_settings_path}")
    
    def create_optimization_script(self):
        """Create a script for optimizing the Unity project."""
        optimization_path = os.path.join(self.unity_project_path, "Assets/Editor/PerformanceOptimization.cs")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(optimization_path), exist_ok=True)
        
        script_content = """using UnityEngine;
using UnityEditor;
using System.IO;

public class PerformanceOptimization
{
    [MenuItem("Arcanum/Optimize Performance")]
    public static void OptimizePerformance()
    {
        // Batch occlusion culling
        BakeOcclusionCulling();
        
        // Configure LOD settings for buildings
        ConfigureLODSettings();
        
        // Set up shader variants
        OptimizeShaderVariants();
        
        Debug.Log("Performance optimization completed!");
    }
    
    private static void BakeOcclusionCulling()
    {
        Debug.Log("Baking occlusion culling data...");
        
        // Get current scene
        UnityEngine.SceneManagement.Scene currentScene = UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene();
        
        // Configure occlusion culling settings
        UnityEditor.StaticOcclusionCulling.smallestOccluder = 5.0f; // Appropriate for urban environment
        UnityEditor.StaticOcclusionCulling.smallestHole = 0.25f;
        UnityEditor.StaticOcclusionCulling.backfaceThreshold = 100;
        
        // Bake occlusion culling
        UnityEditor.StaticOcclusionCulling.Compute();
        
        Debug.Log("Occlusion culling baked successfully");
    }
    
    private static void ConfigureLODSettings()
    {
        Debug.Log("Configuring LOD settings for buildings...");
        
        // Find all LOD groups in the project
        string[] lodGroupGuids = AssetDatabase.FindAssets("t:Prefab", new[] { "Assets/Arcanum/Prefabs/Buildings" });
        
        foreach (string guid in lodGroupGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            
            if (prefab != null && prefab.GetComponent<LODGroup>() != null)
            {
                LODGroup lodGroup = prefab.GetComponent<LODGroup>();
                
                // Configure LOD settings
                LOD[] lods = lodGroup.GetLODs();
                
                // Typical LOD transition distances for city buildings
                if (lods.Length >= 4)
                {
                    // 4-level LOD setup
                    lods[0].screenRelativeTransitionHeight = 0.6f;   // LOD0 - 60% of screen height
                    lods[1].screenRelativeTransitionHeight = 0.3f;   // LOD1 - 30% of screen height
                    lods[2].screenRelativeTransitionHeight = 0.15f;  // LOD2 - 15% of screen height
                    lods[3].screenRelativeTransitionHeight = 0.01f;  // LOD3 - 1% of screen height (distant)
                }
                else if (lods.Length >= 3)
                {
                    // 3-level LOD setup
                    lods[0].screenRelativeTransitionHeight = 0.5f;   // LOD0
                    lods[1].screenRelativeTransitionHeight = 0.25f;  // LOD1
                    lods[2].screenRelativeTransitionHeight = 0.01f;  // LOD2
                }
                else if (lods.Length >= 2)
                {
                    // 2-level LOD setup
                    lods[0].screenRelativeTransitionHeight = 0.4f;   // LOD0
                    lods[1].screenRelativeTransitionHeight = 0.01f;  // LOD1
                }
                
                // Enable LOD crossfading
                lodGroup.fadeMode = LODFadeMode.CrossFade;
                lodGroup.animateCrossFading = true;
                
                // Apply changes
                lodGroup.SetLODs(lods);
                EditorUtility.SetDirty(prefab);
                
                Debug.Log($"Configured LOD settings for {path}");
            }
        }
        
        // Save all changes
        AssetDatabase.SaveAssets();
    }
    
    private static void OptimizeShaderVariants()
    {
        Debug.Log("Optimizing shader variants...");
        
        // This is typically project-specific and may require custom shader handling
        // We'll implement a basic variant reduction by creating simplified distance shaders
        
        // Create a simplified shader for distant buildings
        string distantShaderPath = "Assets/Arcanum/Materials/DistantBuilding.shader";
        
        // Simple shader for distant objects
        string shaderContent = @"Shader ""Arcanum/DistantBuilding"" {
    Properties {
        _MainTex (""Albedo (RGB)"", 2D) = ""white"" {}
        _Color (""Color"", Color) = (1,1,1,1)
    }
    SubShader {
        Tags { ""RenderType""=""Opaque"" }
        LOD 100
        
        CGPROGRAM
        #pragma surface surf Lambert
        
        sampler2D _MainTex;
        fixed4 _Color;
        
        struct Input {
            float2 uv_MainTex;
        };
        
        void surf (Input IN, inout SurfaceOutput o) {
            fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
            o.Albedo = c.rgb;
            o.Alpha = c.a;
        }
        ENDCG
    }
    FallBack ""Diffuse""
}";

        // Create shader file
        File.WriteAllText(Path.Combine(Application.dataPath, "..", distantShaderPath), shaderContent);
        AssetDatabase.Refresh();
        
        Debug.Log("Shader variants optimized");
    }
}"""
        
        with open(optimization_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created performance optimization script at {optimization_path}")
    
    def import_assets(self):
        """Copy assets from the arcanum assets directory to the Unity project."""
        if not os.path.exists(self.arcanum_assets_path):
            logger.warning(f"Arcanum assets path {self.arcanum_assets_path} does not exist. Skipping asset import.")
            return
        
        if not os.path.exists(self.unity_project_path):
            logger.error("Unity project does not exist. Create it first.")
            return
        
        logger.info(f"Importing assets from {self.arcanum_assets_path} to Unity project...")
        
        # Define asset categories to import
        asset_categories = {
            "Terrain/Heightmaps": "Assets/Arcanum/Terrain/Heightmaps",
            "Buildings/LOD0": "Assets/Arcanum/Buildings/LOD0",
            "Buildings/LOD1": "Assets/Arcanum/Buildings/LOD1", 
            "Buildings/LOD2": "Assets/Arcanum/Buildings/LOD2",
            "Buildings/LOD3": "Assets/Arcanum/Buildings/LOD3",
            "Buildings/Landmarks": "Assets/Arcanum/Buildings/Landmarks",
            "Textures/Buildings": "Assets/Arcanum/Textures/Buildings",
            "Textures/Roads": "Assets/Arcanum/Textures/Roads",
            "Textures/Terrain": "Assets/Arcanum/Textures/Terrain",
            "Textures/UI": "Assets/Arcanum/Textures/UI"
        }
        
        for source_subpath, dest_path in asset_categories.items():
            source_path = os.path.join(self.arcanum_assets_path, source_subpath)
            destination_path = os.path.join(self.unity_project_path, dest_path)
            
            if os.path.exists(source_path):
                logger.info(f"Copying assets from {source_path} to {destination_path}")
                
                # Ensure destination directory exists
                os.makedirs(destination_path, exist_ok=True)
                
                # Copy all files from source to destination
                for filename in os.listdir(source_path):
                    source_file = os.path.join(source_path, filename)
                    if os.path.isfile(source_file):
                        shutil.copy2(source_file, destination_path)
                        logger.debug(f"Copied {filename} to {destination_path}")
        
        logger.info("Asset import completed successfully")
    
    def run_unity_editor_setup(self):
        """Run Unity Editor with the setup scripts."""
        if not os.path.exists(self.unity_project_path):
            logger.error("Unity project does not exist. Create it first.")
            return
        
        if not self.unity_path:
            logger.error("Unity Editor path not found. Cannot run Unity Editor setup.")
            return
        
        logger.info("Running Unity Editor to execute setup scripts...")
        
        # Run Unity with each setup script
        setup_methods = [
            "Assets.Editor.CreateMainScene.SetupMainScene",
            "Assets.Editor.AddressableSetup.SetupAddressableAssets",
            "Assets.Editor.AssetImportSettings.ConfigureImportSettings",
            "Assets.Editor.PerformanceOptimization.OptimizePerformance"
        ]
        
        for method in setup_methods:
            cmd = [
                self.unity_path,
                "-projectPath", self.unity_project_path,
                "-executeMethod", method,
                "-batchmode",
                "-quit"
            ]
            
            try:
                logger.info(f"Executing method: {method}")
                subprocess.run(cmd, check=True)
                logger.info(f"Method {method} executed successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to execute method {method}: {str(e)}")
    
    def run(self):
        """Run the complete Unity integration process."""
        try:
            logger.info("Starting Arcanum Unity integration...")
            
            # 1. Create Unity project with HDRP template
            self.create_unity_project()
            
            # 2. Install required packages
            self.install_required_packages()
            
            # 3. Create directory structure
            self._create_directory_structure()
            
            # 4. Create necessary scripts
            self.create_streaming_manager_script()
            self.create_player_camera_script()
            self.create_minimap_script()
            self.create_main_scene()
            self.create_addressable_setup_script()
            self.create_import_settings_script()
            self.create_optimization_script()
            
            # 5. Import assets
            self.import_assets()
            
            # 6. Run Unity Editor to execute setup scripts
            self.run_unity_editor_setup()
            
            logger.info("Arcanum Unity integration completed successfully!")
            logger.info(f"Unity project available at: {self.unity_project_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error during Unity integration: {str(e)}")
            return False
    

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Arcanum Unity Integration Script")
    parser.add_argument("--unity-path", help="Path to Unity Editor executable", default=None)
    parser.add_argument("--project-path", help="Path to create/use the Unity project", default=None)
    parser.add_argument("--assets-path", help="Path to Arcanum asset files", default=None)
    args = parser.parse_args()
    
    # Create integrator instance
    integrator = ArcanumUnityIntegrator(
        unity_path=args.unity_path,
        unity_project_path=args.project_path,
        arcanum_assets_path=args.assets_path
    )
    
    # Run the integration process
    result = integrator.run()
    
    if result:
        print("✅ Arcanum Unity integration completed successfully!")
        print(f"Unity project available at: {integrator.unity_project_path}")
        return 0
    else:
        print("❌ Arcanum Unity integration failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())