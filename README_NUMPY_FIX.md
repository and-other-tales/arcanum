# NumPy Compatibility Fix

This project has been updated to use `numpy<2.0.0` to maintain compatibility with system-installed scikit-learn packages. The original error was related to binary incompatibility between numpy 2.x and older scikit-learn versions.

## Changes Made

1. Modified `requirements.txt` to specify `numpy<2.0.0` instead of `numpy==2.2.5`
2. Updated `x-flux-comfyui/pyproject.toml` to include the compatible numpy version

## Testing

The fix resolves the original error:

```
numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

## Installation

To apply this fix in a new environment:

```bash
# Create and activate a virtual environment
python -m venv arcanum_env
source arcanum_env/bin/activate

# Install dependencies with the fixed numpy version constraint
pip install -r requirements.txt
```

## Note on Dependencies

Some features may still be disabled due to:
- Missing GDAL dependencies (see README_GDAL.md)
- Missing Google Cloud credentials
- Style transformation features requiring ComfyUI

These are separate from the numpy compatibility issue and require additional configuration.