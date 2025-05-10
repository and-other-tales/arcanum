#!/usr/bin/env python3
"""
GPU Availability Checker
-----------------------
This module provides functions to check for GPU availability and display GPU information.
"""

import sys
import os
import logging
from typing import Dict, Any, Tuple, Optional

# Set up logger
logger = logging.getLogger(__name__)

def check_torch_gpu() -> Tuple[bool, Dict[str, Any]]:
    """
    Check PyTorch GPU availability and return detailed information.
    
    Returns:
        Tuple containing:
        - Boolean indicating GPU availability
        - Dictionary with GPU information
    """
    info = {}
    
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            
            info.update({
                "cuda_available": True,
                "cuda_version": cuda_version,
                "gpu_count": device_count,
                "gpu_name": device_name,
                "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
            })
            
            return True, info
            
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info.update({
                "mps_available": True,
                "cuda_available": False,
                "device": "Apple Silicon GPU"
            })
            
            return True, info
            
        else:
            info.update({
                "cuda_available": False,
                "cuda_built": torch.cuda.is_built() if hasattr(torch.cuda, "is_built") else False,
                "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
                "error": "No GPU acceleration available"
            })
            
            return False, info
            
    except ImportError as e:
        info["error"] = f"PyTorch import error: {e}"
        return False, info
        
def check_system_gpu() -> Tuple[bool, Dict[str, Any]]:
    """
    Check system GPU availability using nvidia-smi and environment variables.
    
    Returns:
        Tuple containing:
        - Boolean indicating GPU availability
        - Dictionary with GPU information
    """
    info = {}
    
    # Check environment variables
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    torch_device = os.environ.get('TORCH_DEVICE')
    
    info.update({
        "cuda_visible_devices": cuda_visible,
        "torch_device": torch_device
    })
    
    # Try to run nvidia-smi
    try:
        import subprocess
        nvidia_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
            stderr=subprocess.STDOUT, 
            universal_newlines=True
        )
        
        info["nvidia_smi"] = nvidia_output.strip()
        return True, info
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try environment variables as fallback
        if (cuda_visible is not None and cuda_visible != '-1') or (torch_device is not None and 'cuda' in torch_device):
            info["detection_method"] = "environment variables"
            return True, info
        else:
            info["error"] = "No GPU detected via nvidia-smi or environment variables"
            return False, info

def print_gpu_info(torch_info: Dict[str, Any], system_info: Dict[str, Any]) -> None:
    """
    Print formatted GPU information to console.
    
    Args:
        torch_info: Dictionary with PyTorch GPU information
        system_info: Dictionary with system GPU information
    """
    print("="*50)
    print("GPU AVAILABILITY CHECK")
    print("="*50)
    
    print("\nPyTorch GPU Information:")
    print(f"PyTorch version: {torch_info.get('pytorch_version', 'Not available')}")
    
    if torch_info.get('cuda_available', False):
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch_info.get('cuda_version', 'Unknown')}")
        print(f"GPU count: {torch_info.get('gpu_count', 0)}")
        print(f"GPU name: {torch_info.get('gpu_name', 'Unknown')}")
    elif torch_info.get('mps_available', False):
        print("MPS (Apple Silicon GPU) available: Yes")
    else:
        print("No GPU acceleration available via PyTorch")
        
    print("\nSystem GPU Information:")
    if 'nvidia_smi' in system_info:
        print(f"NVIDIA GPU detected: {system_info['nvidia_smi']}")
    else:
        print("NVIDIA GPU not detected with nvidia-smi")
        
    print(f"CUDA_VISIBLE_DEVICES: {system_info.get('cuda_visible_devices', 'Not set')}")
    print(f"TORCH_DEVICE: {system_info.get('torch_device', 'Not set')}")
    
    print("\nOverall GPU Availability:", "YES" if (torch_info.get('cuda_available', False) or 
                                                   torch_info.get('mps_available', False) or
                                                   'nvidia_smi' in system_info) else "NO")
    print("="*50)

def main() -> int:
    """
    Main function to run GPU checks and return status code.
    
    Returns:
        0 if GPU is available, 1 if not, 2 if import error
    """
    torch_available, torch_info = check_torch_gpu()
    system_available, system_info = check_system_gpu()
    
    print_gpu_info(torch_info, system_info)
    
    # Return appropriate status code
    if 'error' in torch_info and 'import' in torch_info['error']:
        return 2  # PyTorch not installed
    elif torch_available or system_available:
        return 0  # GPU available
    else:
        return 1  # No GPU available

if __name__ == "__main__":
    sys.exit(main())