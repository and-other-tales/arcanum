#!/usr/bin/env python3
import os
import sys
import subprocess

# Check environment variables
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
torch_device = os.environ.get('TORCH_DEVICE')

print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
print(f"TORCH_DEVICE: {torch_device}")

# Try to run nvidia-smi
try:
    nvidia_output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                                           stderr=subprocess.STDOUT, 
                                           universal_newlines=True)
    print("NVIDIA-SMI detected GPU:")
    print(nvidia_output)
    sys.exit(0)  # GPU detected
except (subprocess.CalledProcessError, FileNotFoundError):
    # Try environment variables as fallback
    if (cuda_visible is not None and cuda_visible != '-1') or (torch_device is not None and 'cuda' in torch_device):
        print("GPU detected via environment variables")
        sys.exit(0)  # GPU detected via env vars
    else:
        print("No GPU detected")
        sys.exit(1)  # No GPU detected