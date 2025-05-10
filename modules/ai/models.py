#!/usr/bin/env python3
"""
Download X-Labs Flux Models for ComfyUI (wrapper for backwards compatibility)
"""

import sys
from modules.models.flux_models import download_core_models_cli

if __name__ == "__main__":
    sys.exit(download_core_models_cli())