#!/usr/bin/env python3
"""
Arcanum Integration Module
------------------------
This module handles integration with external systems and services.
"""

# Google 3D Tiles integration
from modules.integration.g3d_tiles import fetch_tiles as fetch_3d_tiles
from modules.integration.g3d_tiles import fetch_city_tiles
from modules.integration.city_tiles import fetch_city_3d_tiles

# Street View integration
from modules.integration.streetview import fetch_street_view
from modules.integration.sv_roads import fetch_street_view_along_roads

# Coverage and textures
from modules.integration.coverage import verify_coverage
from modules.integration.textures import project_textures

__all__ = [
    # Google 3D Tiles
    'fetch_3d_tiles',
    'fetch_city_tiles',
    'fetch_city_3d_tiles',
    
    # Street View
    'fetch_street_view',
    'fetch_street_view_along_roads',
    
    # Coverage and textures
    'verify_coverage',
    'project_textures'
]