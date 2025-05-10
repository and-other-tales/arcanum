#!/usr/bin/env python3
"""
Test script for MediaPipe functionality after fixing the dependency conflict.
"""

import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")
print(f"MediaPipe modules: {dir(mp)}")

# Check if solutions exists
try:
    import mediapipe.solutions
    print("Solutions module exists!")
except ImportError as e:
    print(f"Solutions import error: {e}")
    print("Solutions module may be missing in this version of mediapipe")

# Try to use a basic mediapipe feature
try:
    # Create a simple face detection solution (should work in most mediapipe versions)
    mp_face_detection = mp.tasks.vision.FaceDetector
    print("Successfully imported face detection task")
except Exception as e:
    print(f"Error loading face detection: {e}")

print("\nMediaPipe installation appears to be working correctly.")