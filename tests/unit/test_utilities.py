#!/usr/bin/env python3
"""
Unit tests for the Arcanum utilities modules.
"""

import os
import sys
import pytest
import tempfile
from unittest.mock import patch, MagicMock

# Import the modules to test
try:
    from modules.utilities.gpu_check import check_gpu
    from modules.utilities.gpu_check_env import check_gpu_environment
    from modules.utilities.ee_auth import authenticate_earth_engine
except ImportError:
    # Skip tests if modules not available
    pytestmark = pytest.mark.skip(reason="Utilities modules not available")


class TestGPUCheck:
    """Test the GPU check functions."""
    
    @patch("torch.cuda.is_available")
    def test_check_gpu_available(self, mock_cuda_available):
        """Test checking if GPU is available."""
        # Mock torch.cuda.is_available to return True
        mock_cuda_available.return_value = True
        
        # Call the function
        result = check_gpu()
        
        # Check result
        assert result["gpu_available"] is True
        assert "gpu_name" in result
        assert "cuda_version" in result
    
    @patch("torch.cuda.is_available")
    def test_check_gpu_not_available(self, mock_cuda_available):
        """Test checking if GPU is not available."""
        # Mock torch.cuda.is_available to return False
        mock_cuda_available.return_value = False
        
        # Call the function
        result = check_gpu()
        
        # Check result
        assert result["gpu_available"] is False
        assert "error" in result
    
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    def test_check_gpu_details(self, mock_properties, mock_name, mock_available):
        """Test checking GPU details."""
        # Mock torch functions
        mock_available.return_value = True
        mock_name.return_value = "Test GPU"
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8 GB
        mock_properties.return_value = mock_props
        
        # Call the function
        result = check_gpu()
        
        # Check result
        assert result["gpu_available"] is True
        assert result["gpu_name"] == "Test GPU"
        assert "memory_total" in result
        assert result["memory_total"] > 0
    
    @patch("torch.cuda.is_available")
    def test_check_gpu_error(self, mock_cuda_available):
        """Test checking GPU with an error."""
        # Mock torch.cuda.is_available to raise an exception
        mock_cuda_available.side_effect = Exception("Test error")
        
        # Call the function
        result = check_gpu()
        
        # Check result
        assert result["gpu_available"] is False
        assert "error" in result


class TestGPUEnvironment:
    """Test the GPU environment check functions."""
    
    @patch("modules.utilities.gpu_check_env.check_gpu")
    @patch("modules.utilities.gpu_check_env.check_cuda_libs")
    def test_check_gpu_environment(self, mock_check_libs, mock_check_gpu):
        """Test checking GPU environment."""
        # Mock check_gpu and check_cuda_libs
        mock_check_gpu.return_value = {
            "gpu_available": True,
            "gpu_name": "Test GPU",
            "cuda_version": "11.7"
        }
        mock_check_libs.return_value = {
            "cuda_libs_found": True,
            "cuda_lib_path": "/usr/local/cuda/lib64"
        }
        
        # Call the function
        result = check_gpu_environment()
        
        # Check result
        assert result["gpu_available"] is True
        assert result["cuda_libs_found"] is True
        assert result["environment_ready"] is True
    
    @patch("modules.utilities.gpu_check_env.check_gpu")
    @patch("modules.utilities.gpu_check_env.check_cuda_libs")
    def test_check_gpu_environment_no_gpu(self, mock_check_libs, mock_check_gpu):
        """Test checking GPU environment with no GPU."""
        # Mock check_gpu to report no GPU
        mock_check_gpu.return_value = {
            "gpu_available": False,
            "error": "No GPU found"
        }
        mock_check_libs.return_value = {
            "cuda_libs_found": True,
            "cuda_lib_path": "/usr/local/cuda/lib64"
        }
        
        # Call the function
        result = check_gpu_environment()
        
        # Check result
        assert result["gpu_available"] is False
        assert result["environment_ready"] is False
        assert "error" in result
    
    @patch("modules.utilities.gpu_check_env.check_gpu")
    @patch("modules.utilities.gpu_check_env.check_cuda_libs")
    def test_check_gpu_environment_no_libs(self, mock_check_libs, mock_check_gpu):
        """Test checking GPU environment with no CUDA libs."""
        # Mock check_gpu and check_cuda_libs
        mock_check_gpu.return_value = {
            "gpu_available": True,
            "gpu_name": "Test GPU",
            "cuda_version": "11.7"
        }
        mock_check_libs.return_value = {
            "cuda_libs_found": False,
            "error": "No CUDA libs found"
        }
        
        # Call the function
        result = check_gpu_environment()
        
        # Check result
        assert result["gpu_available"] is True
        assert result["cuda_libs_found"] is False
        assert result["environment_ready"] is False
        assert "error" in result


class TestEarthEngineAuth:
    """Test the Earth Engine authentication functions."""
    
    @patch("modules.utilities.ee_auth.ee.Initialize")
    @patch("modules.utilities.ee_auth.ee.Authenticate")
    def test_authenticate_earth_engine_interactive(self, mock_authenticate, mock_initialize):
        """Test authenticating Earth Engine interactively."""
        # Call the function
        result = authenticate_earth_engine()
        
        # Check result
        assert mock_authenticate.called
        assert mock_initialize.called
        assert result["success"] is True
    
    @patch("modules.utilities.ee_auth.ee.Initialize")
    def test_authenticate_earth_engine_silent(self, mock_initialize):
        """Test authenticating Earth Engine silently."""
        # Call the function
        result = authenticate_earth_engine(interactive=False)
        
        # Check result
        assert mock_initialize.called
        assert result["success"] is True
    
    @patch("modules.utilities.ee_auth.ee.Initialize")
    def test_authenticate_earth_engine_error(self, mock_initialize):
        """Test authenticating Earth Engine with an error."""
        # Mock ee.Initialize to raise an exception
        mock_initialize.side_effect = Exception("Test error")
        
        # Call the function
        result = authenticate_earth_engine(interactive=False)
        
        # Check result
        assert mock_initialize.called
        assert result["success"] is False
        assert "error" in result