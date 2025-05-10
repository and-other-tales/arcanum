#!/usr/bin/env python3
"""
Unit tests for the Arcanum AI modules.
"""

import os
import sys
import pytest
import tempfile
from unittest.mock import patch, MagicMock, call

# Import the modules to test
try:
    from modules.ai.models import download_flux_models
    from modules.ai.upscaler import download_flux_upscaler
    from modules.comfyui import ArcanumStyleTransformer, ComfyUIStyleTransformer
    from modules.comfyui import transform_image, batch_transform_images
except ImportError:
    # Skip tests if modules not available
    pytestmark = pytest.mark.skip(reason="AI modules not available")


class TestFluxModels:
    """Test the Flux models functionality."""
    
    @patch("modules.ai.models.os.path.exists")
    @patch("modules.ai.models.os.makedirs")
    @patch("modules.ai.models.download_file")
    def test_download_flux_models(self, mock_download, mock_makedirs, mock_exists):
        """Test downloading Flux models."""
        # Set up mocks
        mock_exists.return_value = False
        
        # Call the function
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_flux_models(output_dir=temp_dir)
        
        # Check result
        assert mock_makedirs.called
        assert mock_download.called
        assert result["success"] is True
        assert "models" in result
        assert len(result["models"]) > 0
    
    @patch("modules.ai.models.os.path.exists")
    def test_download_flux_models_already_exists(self, mock_exists):
        """Test downloading Flux models when they already exist."""
        # Set up mocks
        mock_exists.return_value = True
        
        # Call the function
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_flux_models(output_dir=temp_dir, force=False)
        
        # Check result
        assert mock_exists.called
        assert result["success"] is True
        assert result["skipped"] is True
    
    @patch("modules.ai.models.os.path.exists")
    @patch("modules.ai.models.os.makedirs")
    @patch("modules.ai.models.download_file")
    def test_download_flux_models_with_token(self, mock_download, mock_makedirs, mock_exists):
        """Test downloading Flux models with a token."""
        # Set up mocks
        mock_exists.return_value = False
        
        # Call the function
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_flux_models(
                output_dir=temp_dir,
                hf_token="test_token"
            )
        
        # Check result
        assert mock_makedirs.called
        assert mock_download.called
        assert result["success"] is True
        # Check that token was used in download
        for call_args in mock_download.call_args_list:
            assert "token" in call_args[1]
            assert call_args[1]["token"] == "test_token"
    
    @patch("modules.ai.models.os.path.exists")
    @patch("modules.ai.models.os.makedirs")
    @patch("modules.ai.models.download_file")
    def test_download_flux_models_error(self, mock_download, mock_makedirs, mock_exists):
        """Test downloading Flux models with an error."""
        # Set up mocks
        mock_exists.return_value = False
        mock_download.side_effect = Exception("Test error")
        
        # Call the function
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_flux_models(output_dir=temp_dir)
        
        # Check result
        assert mock_makedirs.called
        assert mock_download.called
        assert result["success"] is False
        assert "error" in result


class TestFluxUpscaler:
    """Test the Flux upscaler functionality."""
    
    @patch("modules.ai.upscaler.os.path.exists")
    @patch("modules.ai.upscaler.os.makedirs")
    @patch("modules.ai.upscaler.download_file")
    def test_download_flux_upscaler(self, mock_download, mock_makedirs, mock_exists):
        """Test downloading Flux upscaler."""
        # Set up mocks
        mock_exists.return_value = False
        
        # Call the function
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_flux_upscaler(output_dir=temp_dir)
        
        # Check result
        assert mock_makedirs.called
        assert mock_download.called
        assert result["success"] is True
        assert "model_path" in result
    
    @patch("modules.ai.upscaler.os.path.exists")
    def test_download_flux_upscaler_already_exists(self, mock_exists):
        """Test downloading Flux upscaler when it already exists."""
        # Set up mocks
        mock_exists.return_value = True
        
        # Call the function
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_flux_upscaler(output_dir=temp_dir, force=False)
        
        # Check result
        assert mock_exists.called
        assert result["success"] is True
        assert result["skipped"] is True
    
    @patch("modules.ai.upscaler.os.path.exists")
    @patch("modules.ai.upscaler.os.makedirs")
    @patch("modules.ai.upscaler.download_file")
    def test_download_flux_upscaler_error(self, mock_download, mock_makedirs, mock_exists):
        """Test downloading Flux upscaler with an error."""
        # Set up mocks
        mock_exists.return_value = False
        mock_download.side_effect = Exception("Test error")
        
        # Call the function
        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_flux_upscaler(output_dir=temp_dir)
        
        # Check result
        assert mock_makedirs.called
        assert mock_download.called
        assert result["success"] is False
        assert "error" in result


@pytest.mark.skip(reason="Requires ComfyUI which may not be available")
class TestComfyUIIntegration:
    """Test the ComfyUI integration."""
    
    def test_arcanum_style_transformer_init(self, mock_comfyui_path):
        """Test initializing the ArcanumStyleTransformer."""
        # Create transformer
        transformer = ArcanumStyleTransformer(comfyui_path=mock_comfyui_path)
        
        # Check properties
        assert transformer.comfyui_path == mock_comfyui_path
        assert transformer.model_id == "black-forest-labs/FLUX.1-dev"
        assert transformer.initialized is False
    
    def test_comfyui_style_transformer_init(self, mock_comfyui_path):
        """Test initializing the ComfyUIStyleTransformer."""
        # Check that ComfyUIStyleTransformer is an alias of ArcanumStyleTransformer
        assert ComfyUIStyleTransformer is ArcanumStyleTransformer
        
        # Create transformer
        transformer = ComfyUIStyleTransformer(comfyui_path=mock_comfyui_path)
        
        # Check properties
        assert transformer.comfyui_path == mock_comfyui_path
        assert transformer.model_id == "black-forest-labs/FLUX.1-dev"
        assert transformer.initialized is False
    
    @patch("modules.comfyui.transform_image")
    def test_transform_image(self, mock_transform, sample_image_path, temp_dir):
        """Test transforming an image."""
        # Set up mock
        output_path = os.path.join(temp_dir, "output.png")
        mock_transform.return_value = {"success": True, "output_path": output_path}
        
        # Call the function
        result = transform_image(
            image_path=sample_image_path,
            output_path=output_path,
            prompt="arcanum gothic fantasy"
        )
        
        # Check result
        assert mock_transform.called
        assert result["success"] is True
        assert result["output_path"] == output_path
    
    @patch("modules.comfyui.batch_transform_images")
    def test_batch_transform_images(self, mock_batch, temp_dir):
        """Test batch transforming images."""
        # Set up mock
        output_dir = os.path.join(temp_dir, "output")
        mock_batch.return_value = {
            "success": True,
            "output_paths": [os.path.join(output_dir, "output1.png"), os.path.join(output_dir, "output2.png")]
        }
        
        # Create input images
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        with open(os.path.join(input_dir, "test1.jpg"), "w") as f:
            f.write("test1")
        with open(os.path.join(input_dir, "test2.jpg"), "w") as f:
            f.write("test2")
        
        # Call the function
        image_paths = [
            os.path.join(input_dir, "test1.jpg"),
            os.path.join(input_dir, "test2.jpg")
        ]
        result = batch_transform_images(
            image_paths=image_paths,
            output_dir=output_dir,
            prompt="arcanum gothic fantasy"
        )
        
        # Check result
        assert mock_batch.called
        assert result["success"] is True
        assert len(result["output_paths"]) == 2