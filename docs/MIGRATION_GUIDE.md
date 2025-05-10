# Migration Guide: Transitioning to the Modular Structure

This guide helps developers migrate from the legacy `integration_tools` flat structure to the new `modules` package structure.

## Overview of Changes

The Arcanum codebase has been refactored to use a more organized, modular structure. The main changes include:

1. Moving functionality from `integration_tools/` to `modules/` subdirectories
2. Standardizing imports and module interfaces
3. Eliminating duplicated code across the codebase
4. Adding better error handling and fallbacks for missing dependencies

## Directory Structure

Old structure:
```
arcanum/
├── integration_tools/           # Flat directory with all integration modules
│   ├── comfyui_integration.py
│   ├── google_3d_tiles_integration.py
│   ├── storage_integration.py
│   └── ...
└── ...
```

New structure:
```
arcanum/
├── modules/                     # Main modules package
│   ├── comfyui/                 # ComfyUI integration
│   │   ├── __init__.py
│   │   └── automation.py
│   ├── integration/             # Various integrations
│   │   ├── __init__.py
│   │   ├── google_3d_tiles_integration.py
│   │   ├── road_network_integration.py
│   │   └── ...
│   ├── storage/                 # Storage management
│   │   ├── __init__.py
│   │   └── storage_integration.py
│   └── ...
└── ...
```

## Import Changes

Here's how to update your imports:

### ComfyUI Integration

Old:
```python
from integration_tools.comfyui_integration import ComfyUIStyleTransformer, transform_image
```

New:
```python
from modules.comfyui.automation import ComfyUIStyleTransformer, transform_image
```

### Google 3D Tiles Integration

Old:
```python
from integration_tools.google_3d_tiles_integration import Google3DTilesIntegration, fetch_city_tiles
```

New:
```python
from modules.integration.google_3d_tiles_integration import Google3DTilesIntegration, fetch_city_tiles
```

### Storage Integration

Old:
```python
from integration_tools.storage_integration import ArcanumStorageIntegration
```

New:
```python
from modules.storage.storage_integration import ArcanumStorageIntegration
```

### Road Network Integration

Old:
```python
from integration_tools.road_network import fetch_street_view_along_roads
```

New:
```python
from modules.integration.road_network_integration import fetch_street_view_along_roads
```

## Backward Compatibility

For backward compatibility, the old `integration_tools` modules are still present but now include deprecation warnings and will forward to the new implementations. This provides time to transition your code while maintaining existing functionality.

Example of handling both old and new imports:

```python
try:
    # Try to import from modules (new structure)
    from modules.integration.google_3d_tiles_integration import Google3DTilesIntegration
    MODULES_AVAILABLE = True
except ImportError:
    # Fall back to legacy imports
    from integration_tools.google_3d_tiles_integration import Google3DTilesIntegration
    MODULES_AVAILABLE = False
```

## API Changes

The new module structure introduces some API improvements:

1. More consistent naming conventions
2. Better error handling with informative messages
3. Module-level functions for common operations 
4. Improved documentation with type hints and examples
5. Reduced duplication and better code sharing

## Example Migration

### Before Refactoring

```python
import os
from integration_tools.comfyui_integration import generate_image
from integration_tools.storage_integration import ArcanumStorageIntegration

# Generate and upload image
result = generate_image(
    prompt="arcanum gothic victorian city",
    output_path="./output.png"
)

# Upload to storage
storage = ArcanumStorageIntegration()
storage.transform_and_upload_image("./output.png", tileset_name="arcanum")
```

### After Refactoring

```python
import os
from modules.comfyui.automation import transform_image
from modules.storage.storage_integration import transform_and_upload_image

# Generate image with improved API
result = transform_image(
    image_path=None,  # Will use a blank image as starting point
    prompt="arcanum gothic victorian city",
    output_path="./output.png"
)

# Upload to storage with simplified module-level function
transform_and_upload_image("./output.png", tileset_name="arcanum")
```

## Specific Module Changes

### ComfyUI Integration

- The original `ComfyUIStyleTransformer` class is now in `modules.comfyui.automation`
- The function `generate_image` is replaced with `transform_image`
- Better error handling for unavailable models
- More consistent parameter naming

### Google 3D Tiles Integration

- Module moved to `modules.integration.google_3d_tiles_integration`
- Added better spatial bounds functionality integration
- Improved caching system
- Better error handling

### Storage Integration

- Module moved to `modules.storage.storage_integration`
- Added module-level functions for common operations
- Better fallbacks when certain modules are missing
- More consistent error handling

## Timeline for Deprecation

The old `integration_tools` modules will be maintained for backward compatibility until the next major version release. After that, they will be removed completely. We recommend updating your imports as soon as possible to avoid disruption when the legacy modules are removed.

## Need Help?

If you encounter any issues during migration, please:

1. Check the API documentation in the module docstrings
2. Refer to the examples in the `examples/` directory
3. File an issue in the GitHub repository with details about your specific problem