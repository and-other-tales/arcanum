#!/usr/bin/env python3

import os
from integration_tools.tile_server_integration import TileServerIntegration

print('Testing TileServerIntegration...')

# Create a local server for testing
local_server_dir = os.path.join('test_data', 'local_server')
os.makedirs(local_server_dir, exist_ok=True)

# Initialize the tile server integration with local path
tile_server = TileServerIntegration(
    os.path.join('test_data', 'output'),
    local_server_dir
)

# Upload assets to the local server
result = tile_server.upload_assets(zip_files=False)
print(f'Local upload result: {result["success"] if "success" in result else "Failed"}')

# Generate Unity configuration
config_result = tile_server.generate_unity_server_config()
print(f'Config generation result: {config_result["success"] if "success" in config_result else "Failed"}')