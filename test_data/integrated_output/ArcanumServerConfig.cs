using System;
using System.Collections.Generic;
using UnityEngine;

namespace Arcanum
{
    [Serializable]
    public class ArcanumServerConfig
    {
        public string serverType;
        public string serverUrl;
        public string mainCityPrefab;
        
        [Serializable]
        public class AssetCollection
        {
            public Dictionary<string, string> prefabs = new Dictionary<string, string>();
            public Dictionary<string, string> materials = new Dictionary<string, string>();
            public Dictionary<string, string> textures = new Dictionary<string, string>();
            public Dictionary<string, string> models = new Dictionary<string, string>();
        }
        
        public AssetCollection assets = new AssetCollection();
        
        
        private static ArcanumServerConfig _instance;
        
        public static ArcanumServerConfig Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = LoadConfig();
                }
                return _instance;
            }
        }
        
        private static ArcanumServerConfig LoadConfig()
        {
            // Try to load from Resources
            TextAsset configAsset = Resources.Load<TextAsset>("ArcanumServerConfig");
            if (configAsset != null)
            {
                try
                {
                    ArcanumServerConfig config = JsonUtility.FromJson<ArcanumServerConfig>(configAsset.text);
                    Debug.Log("Loaded ArcanumServerConfig from Resources");
                    return config;
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error parsing ArcanumServerConfig: {ex.Message}");
                }
            }
            
            // Fallback to default config
            Debug.LogWarning("Using default ArcanumServerConfig");
            return CreateDefaultConfig();
        }
        
        private static ArcanumServerConfig CreateDefaultConfig()
        {
            ArcanumServerConfig config = new ArcanumServerConfig();
            config.serverType = "local";
            config.serverUrl = "Arcanum/Assets";
            return config;
        }
        
        public string GetAssetUrl(string assetName)
        {
            // Check each asset collection
            if (assets.prefabs.ContainsKey(assetName))
                return assets.prefabs[assetName];
                
            if (assets.materials.ContainsKey(assetName))
                return assets.materials[assetName];
                
            if (assets.textures.ContainsKey(assetName))
                return assets.textures[assetName];
                
            if (assets.models.ContainsKey(assetName))
                return assets.models[assetName];
                
            // Not found - return default path
            return $"{serverUrl}/{assetName}";
        }
    }
}
