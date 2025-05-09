import sys
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA_VISIBLE_DEVICES: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
    
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print(f"CUDA available: Yes")
        print(f"CUDA version: {cuda_version}")
        print(f"GPU count: {device_count}")
        print(f"GPU name: {device_name}")
        sys.exit(0)  # GPU available
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) available: Yes")
        sys.exit(0)  # GPU available (Apple Silicon)
    else:
        print("No GPU acceleration available")
        # Additional diagnostics
        print(f"CUDA_VISIBLE_DEVICES env: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"Has CUDA: {torch.cuda.is_built()}")
        sys.exit(1)  # No GPU
except ImportError as e:
    print(f"PyTorch import error: {e}")
    sys.exit(2)  # PyTorch not installed