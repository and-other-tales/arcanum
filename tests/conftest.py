#!/usr/bin/env python3
"""
Arcanum Test Configuration
------------------------
This module provides pytest fixtures and configuration for Arcanum tests.
"""

import os
import sys
import pytest
import shutil
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def temp_dir():
    """Create a temporary directory that's cleaned up after the test."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture
def sample_bounds():
    """Return sample geographic bounds for a small area."""
    return {
        "north": 51.5074 + 0.01,
        "south": 51.5074 - 0.01,
        "east": -0.1278 + 0.01,
        "west": -0.1278 - 0.01
    }


@pytest.fixture
def sample_city_name():
    """Return a sample city name for testing."""
    return "London"


@pytest.fixture
def mock_api_key():
    """Return a mock API key for testing."""
    return "mock_api_key_for_testing_only"


@pytest.fixture
def mock_storage_path(temp_dir):
    """Return a temporary storage path."""
    storage_path = os.path.join(temp_dir, 'storage')
    os.makedirs(storage_path, exist_ok=True)
    return storage_path


@pytest.fixture
def mock_comfyui_path(temp_dir):
    """Return a mock ComfyUI path."""
    comfyui_path = os.path.join(temp_dir, 'comfyui')
    os.makedirs(comfyui_path, exist_ok=True)
    return comfyui_path


@pytest.fixture
def sample_image_path(test_data_dir):
    """Return a path to a sample image for testing."""
    # First, check if we have a sample image in test_data
    sample_images = [
        os.path.join(test_data_dir, 'sample.jpg'),
        os.path.join(test_data_dir, 'sample.png')
    ]
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            return img_path
    
    # If no sample image exists, create a simple one
    from PIL import Image
    
    # Create test_data directory if it doesn't exist
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Create a simple black image
    img_path = os.path.join(test_data_dir, 'sample.png')
    img = Image.new('RGB', (100, 100), color='black')
    img.save(img_path)
    
    return img_path