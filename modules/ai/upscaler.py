#!/usr/bin/env python3
"""
Download Flux 1-dev Controlnet Upscaler Model (wrapper for backwards compatibility)
"""

import sys
from modules.models.flux_models import download_upscaler_cli

if __name__ == "__main__":
    sys.exit(download_upscaler_cli())