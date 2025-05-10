#!/usr/bin/env python3
"""
Authentication wrapper for Google Earth Engine (for backwards compatibility)
"""

import sys
from modules.geo.earth_engine import main

if __name__ == "__main__":
    sys.exit(main())