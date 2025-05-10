#!/usr/bin/env python3
"""
Arcanum ComfyUI Module
---------------------
This module provides integration with ComfyUI and X-Labs Flux for
image style transformation and other AI operations needed by Arcanum.
"""

from .automation import (
    setup_comfyui,
    get_comfyui_status,
    transform_image,
    batch_transform_images,
    download_flux_models
)

from .transformer import ArcanumStyleTransformer

__all__ = [
    'setup_comfyui',
    'get_comfyui_status',
    'transform_image',
    'batch_transform_images', 
    'download_flux_models',
    'ArcanumStyleTransformer'
]