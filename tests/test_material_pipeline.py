#!/usr/bin/env python3

import json
import os
from integration_tools.unity_material_pipeline import UnityMaterialPipeline

print('Testing UnityMaterialPipeline...')
pipeline = UnityMaterialPipeline('test_data/output/materials', 'test_data/textures', 'test_data/unity')

# Create a simple material directly to test
result = pipeline.create_material_for_building(
    'test_building_1',
    'test_data/textures/brick.jpg',
    'brick',
    {'height': 25.0, 'width': 10.0, 'depth': 15.0}
)

print(f'Material creation result: {result["success"] if "success" in result else "Failed"}')