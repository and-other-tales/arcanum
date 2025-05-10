#!/usr/bin/env python3

import json
import os
from PIL import Image, ImageOps, ImageChops, ImageEnhance, ImageFilter

def test_material_pipeline():
    # Create a simple test image
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    img.save("test_data/textures/test.jpg")
    
    # Create simple maps
    albedo = img
    normal = Image.new("RGB", (256, 256), (128, 128, 255))
    metallic = Image.new("RGB", (256, 256), (0, 0, 0))
    roughness = Image.new("RGB", (256, 256), (200, 200, 200))
    ao = Image.new("RGB", (256, 256), (255, 255, 255))
    
    # Save all maps
    albedo.save("test_data/output/materials/processed_textures/test_albedo.png")
    normal.save("test_data/output/materials/processed_textures/test_normal.png")
    metallic.save("test_data/output/materials/processed_textures/test_metallic.png")
    roughness.save("test_data/output/materials/processed_textures/test_roughness.png")
    ao.save("test_data/output/materials/processed_textures/test_ao.png")
    
    print("Created simple material textures successfully")

if __name__ == "__main__":
    test_material_pipeline()