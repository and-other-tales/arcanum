#!/usr/bin/env python3
"""
Arcanum AI Module
--------------
This module provides functionality for AI model management and operations.
"""

from modules.ai.models import download_flux_models
from modules.ai.upscaler import download_flux_upscaler

__all__ = [
    'download_flux_models',
    'download_flux_upscaler'
]