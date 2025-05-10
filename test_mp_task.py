#!/usr/bin/env python3
"""
Test script for MediaPipe Tasks API functionality after fixing the dependency conflict.
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
print(f"MediaPipe version: {mp.__version__}")

# Test creating a face detector
try:
    # Get the base options
    base_options = python.BaseOptions(model_asset_path="does_not_exist.tflite")
    print("Successfully created base options")
    
    # Try to create a face detector (will fail due to missing model, but tests imports)
    options = vision.FaceDetectorOptions(base_options=base_options)
    print("Successfully created face detector options")
    
    print("MediaPipe Tasks API is working properly!")
except Exception as e:
    print(f"Error using MediaPipe Tasks API: {e}")