#!/usr/bin/env python3
"""
GPU check wrapper (for backwards compatibility)
"""

import sys
from modules.utils.gpu_check import main

if __name__ == "__main__":
    sys.exit(main())